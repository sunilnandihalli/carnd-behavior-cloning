import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import keras
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D


from keras.models import load_model,model_from_json,model_from_yaml
import h5py
from keras import __version__ as keras_version
import sys
import traceback

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def my_load_model(prefix):
    try:
        print('trying to load model from hd5 file')
        m = load_model(prefix+'.hd5')
        model_read_successful = True
        weight_read_successful = True
    except :
        print(sys.exc_info())
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        print('loading model from hd5 failed')
        model_read_successful = False
        weight_read_successful = False
    if not model_read_successful:
        try:
            print('trying to load model from json file')
            with open(prefix+'.json','r') as f:
                m = model_from_json(f.read())
            model_read_successful = True
        except :
            print(sys.exc_info())
            print('loading model from json failed')
    if not model_read_successful:
        try:
            print('trying to load model from yaml file')
            with open(prefix+'.yml','r') as f:
                m = model_from_yaml(f.read())
            model_read_successful = True
        except:
            print(sys.exc_info())
            print('loading model from yaml failed')
    if model_read_successful and not weight_read_successful:
        try:
            print('loading weights from _weights.hd5 file')
            m.load_weights(prefix+'_weights.hd5')
            weight_read_successful = True
        except:
            print(sys.exc_info())
            print('loading weights from _weights.hdf file failed')
            return None
    if model_read_successful and weight_read_successful:
        return m
    else:
        return None

def Lenet():
  height = 160
  width = 320
  depth = 3
  model=Sequential()
  model.add(Lambda(lambda x:x/255-0.5,input_shape=(height, width, depth)))
  model.add(Cropping2D(cropping=((70,25),(0,0))))
  model.add(Conv2D(20, 5, 5, border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(50, 5, 5, border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(240,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(84,activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse',optimizer='adam')
  return model

def hacky_load_model(prefix):
    m = Lenet()
    m.load_weights(prefix+'.hd5')
    return m
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model+'.hd5', mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = hacky_load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
