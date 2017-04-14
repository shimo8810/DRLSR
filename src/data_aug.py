#画像データセットを増幅させる 回転 縮小
import os
import numpy as np
import cv2
import sys

# 最初の91枚の画像をカサ増しする
#91-iamgeのパス
image_path = '../images/General-100//'
#保存用ディレクトリのパス
save_path = '../images/general_100_images_aug/'
image_names = os.listdir(image_path)

length = len(image_names)
count = 0
for i in image_names:
    count += 1
    sys.stdout.write("\r images :{}/{}, {}％".format(count, length, (count*100)//length))
    sys.stdout.flush()
    #画像パス
    image_name = image_path + str(i)
    #画像ファイル
    image = cv2.imread(image_name)
    for scale in range(6, 11):
        scale = scale * 0.1
        size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        scaled_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        i1 = cv2.flip(scaled_image.transpose((1,0,2)), 0)
        i2 = cv2.flip(scaled_image, -1)
        i3 = cv2.flip(i1, -1)
        cv2.imwrite(save_path + i.split(".")[0] +\
                        "_scale_" + str(int(scale*10)) + \
                        "_angle_90" + ".bmp", i1)
        cv2.imwrite(save_path + i.split(".")[0] + \
                        "_scale_" + str(int(scale*10)) + \
                        "_angle_180" + ".bmp", i2)
        cv2.imwrite(save_path + i.split(".")[0] +\
                        "_scale_" + str(int(scale*10)) + \
                        "_angle_270" + ".bmp", i3)
        cv2.imwrite(save_path + i.split(".")[0] + \
                        "_scale_" + str(int(scale*10)) + \
                        "_angle_0" + ".bmp", scaled_image)
print("\ncomplete!")
