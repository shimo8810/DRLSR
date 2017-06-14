#画像データセットを増幅させる 回転 縮小
import os
import json
import numpy as np
import cv2
import sys
from os import path

with open('config.json', 'r') as f:
    config = json.load(f)

# 最初の91枚の画像をカサ増しする
#91-iamgeのパス
image_path = config['path_mini']
#保存用ディレクトリのパス
save_path = config['path_mini_aug']
image_names = os.listdir(image_path)
length = len(image_names)
count = 0
for i in image_names:
    count += 1
    sys.stdout.write("\r images :{}/{}, {}％".format(count, length, (count*100)//length))
    sys.stdout.flush()
    #画像パス
    image_name = path.join(image_path,i)
    #画像ファイル
    image = cv2.imread(image_name)
    for scale in range(6, 11):
        scale = scale * 0.1
        size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        scaled_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        i_list = []
        i_list.append(cv2.flip(scaled_image.transpose((1,0,2)), 0))
        i_list.append(cv2.flip(scaled_image, -1))
        i_list.append(cv2.flip(i_list[0], -1))
        i_list.append(scaled_image)
        angle_list = [90,180,270, 0]
        for j in range(len(angle_list)):
                buf_name = "{}_scale_{}_angle_{}.bmp".format(i.split(".")[0], int(scale*10), angle_list[j])
                buf_path = path.join(save_path, buf_name)
                hog = cv2.imwrite(buf_path, i_list[j])
print("\ncomplete!")
