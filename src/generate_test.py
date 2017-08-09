#Training 用のデータセットの作成
#保存形式はnpy 形式で保存するつもり
#データ配列の形状は[num_images, 1, 41, 41]になるはず?
import numpy as np
import cv2
import os
import sys
from joblib import Parallel, delayed
import json
from os import path

if '__main__' == __name__:
    with open('config.json', 'r') as f:
        config = json.load(f)
    #parameter
    SIZE_INPUT = 41
    SIZE_LABEL = 41
    SCALE = 3
    STRIDE = 11

    #path 情報
    image_path = config['path_general100_aug']
    save_path = config['path_general100_npy']
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
    def gen_test(i):
        c = 0
        #画像読み込み
        image = cv2.imread(path.join(image_path,i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        #画像サイズ調整
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
        #各画像を入力サイズに切り分ける
        for x in range(0, height - SIZE_INPUT + 1, STRIDE):
            for y in range(0, width - SIZE_INPUT + 1, STRIDE):
                subim_input = (image_input[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT].astype(np.float32)) / 255.0
                subim_label = (image_label[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT].astype(np.float32)) / 255.0
                #雑魚セクション
                _data = np.concatenate([subim_input, subim_label], axis=0)
                sp = os.path.join(save_path, "{}{}.npy".format(str(i.split('.')[0]), str(c)))
                np.save(sp, _data)
                c += 1
    #並列処理
    Parallel(n_jobs=-1, verbose=10)([delayed(gen_test)(i) for i in image_paths])
    print("saved data to npy")
