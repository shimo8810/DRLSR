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
data = np.zeros((1, 1, SIZE_INPUT, SIZE_INPUT))
hoge = np.zeros(1)
label = np.zeros((1, 1, SIZE_LABEL, SIZE_LABEL))
padding = np.abs(SIZE_INPUT - SIZE_LABEL) / 2
count = 0

#データ生成
for i in image_paths:
    #chainer ではfloat32を読み込むがmatlab版では倍精度に変換している
    image = cv2.imread(i).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #画像サイズの調整
    size = np.array(image.shape)
    size = size - size % SCALE
    #ラベル用画像
    image_label = image[0:size[0], 0:size[1]]
    height, width = image_label.shape

    #中間縮小画像
    buf = cv2.resize(image, (width//SCALE, height//SCALE), \
                     interpolation=cv2.INTER_CUBIC)
    #入力用画像
    image_input = cv2.resize(buf, (width, height), \
                             interpolation=cv2.INTER_CUBIC)

    for x in range(0, height - SIZE_INPUT + 1, STRIDE):
        for y in range(0, width - SIZE_INPUT + 1, STRIDE):
            subim_input = image_input[x:x+SIZE_INPUT, y:y+SIZE_INPUT]
            subim_label = image_label[x:x+SIZE_INPUT, y:y+SIZE_INPUT]

            count += 1
            print(subim_input[np.newaxis, :, :].shape)
