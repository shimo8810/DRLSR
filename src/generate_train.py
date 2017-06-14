#Training 用のデータセットの作成
#保存形式はnpy 形式で保存するつもり
#データ配列の形状は[num_images, 1, 41, 41]になるはず?
import numpy as np
import cv2
import os
import sys
from joblib import Parallel, delayed

#parameter
SIZE_INPUT = 41
SIZE_LABEL = 41
SCALE = 3
STRIDE = 11

#path 情報
image_path = '../images/91_images_aug/'
#image_path = '../images/general_100_aug/'
image_paths = os.listdir(image_path)

#初期化
train = None
label = None
padding = np.abs(SIZE_INPUT - SIZE_LABEL) / 2

#ループ情報
length = len(image_paths)
im_no = 0
#データ生成
#並列化 関数
def gen_train(i):
    c = 0
    #chainer ではfloat32を読み込むがmatlab版では倍精度に変換している
    image = cv2.imread(image_path+i)
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]).astype(np.float64)
    #画像サイズの調整
    size = np.array(image.shape)
    size = size - size % SCALE
    #ラベル用画像
    image_label = image[0:size[0], 0:size[1]]
    height, width = image_label.shape
    #中間縮小画像
    buf = cv2.resize(image_label, (width//SCALE, height//SCALE), \
                     interpolation=cv2.INTER_CUBIC)
    #入力用画像
    image_input = cv2.resize(buf, (width, height), \
                             interpolation=cv2.INTER_CUBIC)
    #各画像を入力サイズに切り分けていく
    for x in range(0, height - SIZE_INPUT + 1, STRIDE):
        for y in range(0, width - SIZE_INPUT + 1, STRIDE):
            subim_input = image_input[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT]/255
            subim_label = image_label[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT]/255

            #雑魚セクション
            _data = np.concatenate([subim_input, subim_label], axis=0)
            np.save('../images/Yang91_npy_float64/' + str(i.split('.')[0]) + '_' + str(c) + '.npy', _data)
            c += 1
#並列処理
Parallel(n_jobs=-1, verbose=5)([delayed(gen_train)(i) for i in image_paths])
print("saved data to npy")
