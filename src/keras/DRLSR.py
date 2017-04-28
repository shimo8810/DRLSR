import keras
#from keras.layers
import glob
import numpy as np
from os import path
import sys

APP_ROOT = path.normpath(path.join(path.dirname(path.abspath( __file__ )), '../../'))

#parameter
batch_size = 64
epoch = 2

train_paths = glob.glob(APP_ROOT + '/images/mini_train_dataset/*')
for path in train_paths:
    print(np.load(path).shape)
