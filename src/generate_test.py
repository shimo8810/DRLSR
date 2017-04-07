#Training 用のデータセットの作成
#保存形式はnpy 形式で保存するつもり
#データ配列の形状は[num_images, 1, 41, 41]になるはず?
import numpy as np
import cv2
import os
import sys
#parameter
SIZE_INPUT = 41
SIZE_LABEL = 41
SCALE = 3
STRIDE = 14

#path 情報
image_path = '../images/91_images_aug/'
image_paths = os.listdir(image_path)

#初期化
train = None
label = None
padding = np.abs(SIZE_INPUT - SIZE_LABEL) / 2

#ループ情報
count = 0
length = len(image_paths)

#データ生成
for i in image_paths:
    count += 1
    sys.stdout.write("\r images :{}/{}, {}％".format(count, length, (count*100)//length))
    sys.stdout.flush()

    #chainer ではfloat32を読み込むがmatlab版では倍精度に変換している
    image = cv2.imread(image_path+i).astype(np.float32)
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

    #各画像を入力サイズに切り分けていく
    for x in range(0, height - SIZE_INPUT + 1, STRIDE):
        for y in range(0, width - SIZE_INPUT + 1, STRIDE):
            subim_input = image_input[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT]
            subim_label = image_label[np.newaxis, np.newaxis, x:x+SIZE_INPUT, y:y+SIZE_INPUT]

            #train, label配列にどんどんデータを追加
            if train is None and label is None:
                train =  subim_input
                label = subim_label
            else:
                train = np.concatenate([train, subim_input], axis=0)
                label = np.concatenate([label, subim_label], axis=0)

    #データのシャッフル（不要？）
np.save('../images/train_data.npy', train)
np.save('../images/label_data.npy', label)
print(train.shape)
print("\nsaved data to npy")
