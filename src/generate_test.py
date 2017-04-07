#Training 用のデータセットの作成
#保存形式はnpy 形式で保存するつもり
#データ配列の形状は[num_images, 1, 41, 41]になるはず?
import numpy as np
import cv2
import os

#parameter
SIZE_INPUT = 41
SIZE_LABEL = 41
SCALE = 3
STRIDE = 14

#テストコード
image_path = '../images/91_images_aug/t10_scale_10_angle_0.bmp'
image_paths = list()
image_paths.append(image_path)

#初期化
data = np.zeros((SIZE_INPUT, SIZE_INPUT, 1, 1))
label = np.zeros((SIZE_LABEL, SIZE_LABEL, 1, 1))
padding = np.abs(SIZE_INPUT - SIZE_LABEL) / 2
count = 0

#データ生成
for i in image_paths:
    #chainer ではfloat32を読み込むがmatlab版では倍精度に変換している
    image = cv2.imread(i).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    print(image.shape)

    #ラベル用画像
