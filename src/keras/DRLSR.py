from os import path
import sys
import numpy as np
import glob
import keras
from keras.models import Model
from keras.layers import Conv2D, Input, concatenate
from keras.callbacks import ModelCheckpoint

APP_ROOT = path.normpath(path.join(path.dirname(path.abspath( __file__ )), '../../'))

#parameter
batch_size = 128
epoch = 100

#学習データ読み込み
#Keras のCNNのデータ読み込みは(data, ch, row, col) OR (data, row, col, ch)
train_paths = glob.glob(APP_ROOT + '/images/mini_train_dataset/*')

x_train = None
y_train = None
for path in train_paths:
    x_buf, y_buf = np.load(path)
    x_buf = x_buf[np.newaxis,:,:,:]
    y_buf = y_buf[np.newaxis,:,:,:]
    x_train = np.concatenate((x_train, x_buf), axis=0) if x_train is not None else x_buf
    y_train = np.concatenate((y_train, y_buf), axis=0) if y_train is not None else y_buf

#テストデータ読み込み
test_paths = glob.glob(APP_ROOT + '/images/mini_test_dataset/*')

x_test = None
y_test = None
for path in test_paths:
    x_buf, y_buf = np.load(path)
    x_buf = x_buf[np.newaxis,:,:,:]
    y_buf = y_buf[np.newaxis,:,:,:]
    x_test = np.concatenate((x_test, x_buf), axis=0) if x_test is not None else x_buf
    y_test = np.concatenate((y_test, y_buf), axis=0) if y_test is not None else y_buf

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape = y_test.shape[1:]

print(input_shape)
#モデル準備
input_img = Input(shape=input_shape)
#Inception model
conv1_3 = Conv2D(filters=8, kernel_size=3, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(input_img)

conv1_5 = Conv2D(filters=8, kernel_size=5, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(input_img)

conv1_9 = Conv2D(filters=8, kernel_size=9, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(input_img)

concat1 = concatenate([conv1_3, conv1_5, conv1_9], axis=1)

#conv model
conv2 = Conv2D(filters=16, kernel_size=1, strides=1,
               padding="same",
               data_format="channels_first",
               activation="relu",
               kernel_initializer='he_normal')(concat1)

conv22 = Conv2D(filters=16, kernel_size=3, strides=1,
               padding="same",
               data_format="channels_first",
               activation="relu",
               kernel_initializer='he_normal')(conv2)

conv23 = Conv2D(filters=16, kernel_size=1, strides=1,
               padding="same",
               data_format="channels_first",
               activation="relu",
               kernel_initializer='he_normal')(conv22)

#inception model 2
conv3_3 = Conv2D(filters=8, kernel_size=3, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(conv23)

conv3_5 = Conv2D(filters=8, kernel_size=5, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(conv23)

conv3_9 = Conv2D(filters=8, kernel_size=9, strides=1,
                 padding="same",
                 data_format="channels_first",
                 activation="relu",
                 kernel_initializer='he_normal')(conv23)

concat3 = concatenate([conv3_3, conv3_5, conv3_9], axis=1)

sparse_img = Conv2D(filters=1, kernel_size=1, strides=1,
               padding="same",
               data_format="channels_first",
               activation="relu",
               kernel_initializer='he_normal')(concat3)

output_img = keras.layers.add([input_img, sparse_img])

#モデルセッティング
model = Model(inputs=input_img, outputs=output_img)
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0)
model.compile(optimizer=sgd, loss='mean_squared_error')
#コールバックリスト
callbacks = []
callbacks.append(ModelCheckpoint(filepath="model.ep{epoch:02d}.h5", period=1))
model.fit(x_train, y_train, batch_size=batch_size,epochs=epoch,validation_data=(x_test, y_test), callbacks=callbacks)

model.save('drlsr_model.h5')
model_json_str = model.to_json()
with open('drlsr_model.json', 'w') as f:
    f.write(model_json_str)
model.save_weights('drlsr_model_weights.h5')
