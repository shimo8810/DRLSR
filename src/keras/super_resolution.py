from os import path
import sys
import numpy as np
import glob
import cv2
import keras
from keras.models import model_from_json

APP_ROOT = path.normpath(path.join(path.dirname(path.abspath( __file__ )), '../../'))
up_scale = 3
#画像読み込み
image = APP_ROOT + '/images/Set1/t1.bmp'
image = cv2.imread(image).astype(np.float32)
#画像サイズ crop
size = np.array(image.shape) - np.array(image.shape) % up_scale
image = image[0:size[0], 0:size[1], :]
#Y情報
image_y  = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#CbCr情報
image_cbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 1:]
#超解像
height, width = image_y.shape
buf = cv2.resize(image_y, (width//up_scale, height//up_scale), interpolation=cv2.INTER_CUBIC)
image_y_bic = cv2.resize(buf, (width, height), interpolation=cv2.INTER_CUBIC)
image_input = image_y_bic[np.newaxis, np.newaxis, :, :] * (1 / 255.0)
#モデルセッティング
model = model_from_json(open('demo_drlsr_model.json').read())
model.load_weights('drlsr_model_weights.h5')
hoge = model.predict(image_input)
hoge = (hoge*255).reshape(174, 195)
cv2.imwrite('imagey.bmp', hoge)
