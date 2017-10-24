""" Behavior_cloning for selfdriving car project
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import google3
from google3.pyglib import gfile
from google3.util.textprogressbar import pybar
import tensorflow.google as tf
from google3.sstable.python import sstable

from google3.learning.deepmind.python import app
from google3.learning.deepmind.python import flags
from google3.learning.deepmind.python import logging

#mprint = tf.app.logging.info
mprint = logging.info
mprint('google3 imports done')




import keras
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
import keras.callbacks as kcb

mprint('keras imports done')
import pandas as pd
from PIL import Image
import numpy as np

mprint('mlstuff done')

import matplotlib.pyplot as plt
mprint('matplotlib imported')

import csv
import tempfile
from contextlib import contextmanager
import collections
import random
from datetime import datetime
import platform

mprint('allimports done')


FLAGS = flags.FLAGS
flags.DEFINE_string('master', 'local',
                           """BNS name of the TensorFlow runtime to use.""")

flags.DEFINE_string('output_file', 'mandelbrot.png',
                           """Where to write output.""")

flags.DEFINE_integer('num_steps', 100,
                            """Number of steps to run.""")

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tfrecordwriter(tfrecords_path):
  with gfs(tfrecords_path) as fname:
    writer = tf.python_io.TFRecordWriter(fname)
    yield writer
    writer.close()
    sys.stdout.flush()



@contextmanager
def gfsread(name):
  tmpdir = tempfile.mkdtemp()
  tmpfname = tmpdir+'/tmp'
  gfile.Copy(name,tmpfname)
  yield tmpfname

@contextmanager
def gfs(name,suffix='.tmpdata'):
  tmp_file = tempfile.NamedTemporaryFile(mode='w',suffix=suffix)
  mprint('writing '+name+' to tmp file')
  yield tmp_file.name
  gfile.Copy(tmp_file.name,name,overwrite=True)

def Lenet():
  height = 160
  width = 320
  depth = 3
  model=Sequential()
  model.add(Lambda(lambda x:x/255-0.5,input_shape=(height, width, depth)))
  model.add(Cropping2D(cropping=((70,25),(0,0))))
  model.add(Conv2D(20, (5, 5), padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(50, (5, 5), padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(240,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(84,activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse',optimizer='adam')
  return model

DrivingFrame = collections.namedtuple('DrivingFrame', 'center left right steering throttle brake speed')
def fromTFExampleMoreFeatures(rcrd):
  ex = tf.train.Example()
  ex.ParseFromString(rcrd)
  height = 160
  width = 320
  def toimg(s):
    return np.fromstring(ex.features.feature[s].bytes_list.value[0],dtype=np.float32).reshape((height,width,-1))
  def toflt(s):
    return ex.features.feature[s].float_list.value[0]
  return DrivingFrame(center = toimg('center'),
  left = toimg('left'),
  right = toimg('right'),
  steering = toflt('steering'),
  throttle=toflt('throttle'),
  brake = toflt('brake'),
  speed = toflt('speed'))

def train_validate_generators(sstable_path,cross_validation_ratio=0.1,batch_size=256):
  mprint('train_validate_generators')
  table = sstable.SSTable(sstable_path)
  n = len(table)
  cv_start = int(n*(1.0-cross_validation_ratio))
  mprint("number of entries in table : "+str(n))
  num_valid = n-cv_start
  mprint("num_valid : "+str(num_valid*3))
  num_train = cv_start
  mprint("num_train : "+str(num_train*3))
  num_valid_steps = int(num_valid/batch_size)+(1 if num_valid%batch_size != 0 else 0)
  num_train_steps = int(num_train/batch_size)
  cv_start_key = table.iteritems(start_offset=cv_start).next()
  tgen = train_generator(sstable_path,batch_size,0.5,None,cv_start_key,None)
  vgen = valid_generator(sstable_path,batch_size,cv_start_key,None,None)
  return tgen,num_train_steps*3,vgen,num_valid_steps*3

def example_generator(sstable_path,start,stop,start_offset,cycle):
  table = sstable.SSTable(sstable_path)
  while True:
    for k,v in table.iteritems(start_offset=start_offset):
      f=fromTFExampleMoreFeatures(v)
      yield (f.center,f.steering)
      yield (f.right,f.steering-0.2)
      yield (f.left,f.steering+0.2)
    if not cycle:
      mprint('finished non-cyclic example generator')
      break

def weight_fn(batch_labels):
  return np.ones_like(np.squeeze(batch_labels)) #-0.5+2/(1+np.exp((-1.0/3)*np.square(np.squeeze(batch_labels))))

def train_generator(sstable_path,batch_size,reject_prob,start,stop,start_offset):
  mprint('train_generator')
  height = 160
  width = 320
  batch_features = np.zeros((batch_size, height, width, 3))
  batch_labels = np.zeros((batch_size,1))
  yieldid=0
  curid=0
  for img,str_angle in example_generator(sstable_path,start,stop,start_offset,True):
    if random.uniform(0.0,1.0)<reject_prob:
      continue
    batch_features[curid]=img
    batch_labels[curid]=str_angle
    curid+=1
    if curid==batch_size:
      yieldid+=1
      if yieldid%10==0:
        print('yieldid : ',yieldid)
      yield batch_features,batch_labels,weight_fn(batch_labels)
      curid=0

def valid_generator(sstable_path,batch_size,start,stop,start_offset):
  mprint('valid_generator')
  height = 160
  width = 320
  batch_features = np.zeros((batch_size, height, width, 3))
  batch_labels = np.zeros((batch_size,1))
  while True:
    curid=0
    for img,str_angle in example_generator(sstable_path,start,stop,start_offset,False):
      batch_features[curid]=img
      batch_labels[curid]=str_angle
      curid+=1
      if curid==batch_size:
        yield batch_features,batch_labels,weight_fn(batch_labels)
        curid=0
    if curid!=0:
      yield batch_features[0:curid,:,:,:],batch_labels[0:curid,:],weight_fn(batch_labels)

def save_model(m,history_object,path_prefix):
  if False:
    with gfs(path_prefix+'.hd5') as fname1:
      m.save(fname1)
  with gfs(path_prefix+'_weights.hd5') as fname2:
    m.save_weights(fname2)
  if False:
    with gfile.FastGFile(path_prefix+'.json', 'w') as f1:
      mprint('writing model json file')
      f1.write(m.to_json())
  if False:
    with gfile.FastGFile(path_prefix+'.yml','w') as f2:
      mprint('writing model to yaml file')
      f2.write(m.to_yaml())
  plot_history(history_object,path_prefix)

class CSVLogger(kcb.CSVLogger):
  def __init__(self,filename):
    kcb.CSVLogger.__init__(self,filename)

  def on_train_begin(self, logs=None):
    self.csv_file = gfile.FastGFile(self.filename, 'w' + self.file_flags)

class ModelCheckpoint(kcb.Callback):
  """keras.callback.ModelCheckPoint
  """
  def __init__(self, filepath, monitor='val_loss', verbose=0,
               save_best_only=False, save_weights_only=False,
               mode='auto', period=1):
    super(ModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.epochs_since_last_save = 0

    if mode not in ['auto', 'min', 'max']:
      warnings.warn('ModelCheckpoint mode %s is unknown, '
                    'fallback to auto mode.' % (mode),
                    RuntimeWarning)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self.filepath.format(epoch=epoch, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          warnings.warn('Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s'
                    % (epoch, self.monitor, self.best,
                       current, filepath))
            self.best = current
            save_model(self.model,None,filepath)
          else:
            if self.verbose > 0:
              print('Epoch %05d: %s did not improve' %
                    (epoch, self.monitor))
      else:
        if self.verbose > 0:
          print('Epoch %05d: saving model to %s' % (epoch, filepath))
        save_model(self.model,None,filepath)



def main(argv):
  mprint('entered main')
  mprint(platform.python_version())

  wdir = '/cns/is-d/home/sunilsn/carnd/t1/p3/'
  #wdir = '/usr/local/google/home/sunilsn/carnd/t1/p3/'
  sstable_path=wdir+'allfeatures.sstable'
  mprint('working director : '+wdir)

  batch_size=1024
  train_gen,train_steps_per_epoch,valid_gen,valid_steps = train_validate_generators(sstable_path,batch_size=batch_size)
  mprint('train_steps_per_epoch : '+str(train_steps_per_epoch))
  mprint('valid_steps : '+str(valid_steps))
  epochs = 10
  model = Lenet()
  dt = datetime.now()
  model_path_prefix = wdir+'model_spe_{:03d}_epochs_{:03d}_datetime_'.format(train_steps_per_epoch,epochs)+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  checkpoint = ModelCheckpoint(model_path_prefix+'_checkpoint.data')
  csvlogger = CSVLogger(model_path_prefix+'_training_log.csv')
  history_object = model.fit_generator(train_gen,validation_data=valid_gen,
                                validation_steps=valid_steps,
                                steps_per_epoch=train_steps_per_epoch,
                                epochs=epochs,
                                callbacks=[checkpoint,csvlogger])
  save_model(model,history_object,model_path_prefix)

def plot_history(history_object,path_prefix):
  if history_object is not None:
    mprint(history_object.history.keys())
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    with gfs(path_prefix+'_history.png',suffix='.png') as fname:
      plt.savefig(fname)

if __name__ == '__main__':
  mprint('calling tf-app-run')
  app.run(main)

