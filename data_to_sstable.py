from PIL import Image
from google3.pyglib import gfile
import numpy as np
import tensorflow as tf
import pandas as pd
from google3.sstable.python import sstable
from google3.util.textprogressbar import pybar
from random import shuffle
import matplotlib.pyplot as plt
import collections

width = 320
height = 160
def readImage(x):
  with gfile.FastGFile(x,'r') as gf:
    image = Image.open(gf)
    image.load()
    return np.array(image).astype(np.float32)
  
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def toTFExample(row):
  img = lambda x: _bytes_feature(readImage(x).tostring())
  flt = lambda x: _float_feature(x)
  feature = {'center':img(row.center),
             'left':img(row.left),
             'right':img(row.right),
             'steering':flt(row.steering),
             'throttle':flt(row.throttle),
             'brake':flt(row.brake),
             'speed':flt(row.speed)}
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()

DrivingFrame = collections.namedtuple('DrivingFrame', 'center left right steering throttle brake speed')
def fromTFExample(rcrd):
  ex = tf.train.Example()
  ex.ParseFromString(rcrd)
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
          
def readCsv(csvfile,frac=1.0):
  with gfile.FastGFile(csvfile,'r') as gf:
    df = pd.read_csv(gf).sample(frac=frac)
  return df
  
def datasets_to_sstable_shuffled():
  data_root = '/usr/local/google/home/sunilsn/carnd/t1/p3/collect_data/'
  sstable_path = '/usr/local/google/home/sunilsn/carnd/t1/p3/collect_data/allfeatures.sstable'
  def helper(dir):
    dirprefix=data_root+x+'/'
    dfi = readCsv(dirprefix+'driving_log.csv')
    mstrip = lambda x:x.strip()
    dfi['center']=dirprefix+dfi['center'].apply(mstrip)
    dfi['left']=dirprefix+dfi['left'].apply(mstrip)
    dfi['right']=dirprefix+dfi['right'].apply(mstrip)
    return dfi
  df=pd.concat([helper(x) for x in ['00','01','02','03','04','05','06','07']])
  df.sample(frac=1.0)
  with sstable.Builder(sstable_path) as sstable_builder:
    for rowid,row in zip(range(999999),df.itertuples()):
      if(rowid+1)%1000==0:
        print('reading .... : ',rowid)
      tfrecord = toTFExample(row)
      sstable_builder.Add('{:09d}'.format(rowid),tfrecord)    
    
datasets_to_sstable_shuffled()
print('all done')
