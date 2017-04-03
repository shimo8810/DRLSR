import os
import numpy as np
import cv2

# 最初の91枚の画像をカサ増しする
image_path = '../images/91_images/'
image_names = os.listdir(image_path)

for i in image_names:
    #画像パス
    image_name = image_path + str(i)
    #画像ファイル
    image = cv2.imread(image_name)
    for angle in range(0, 360, 90):
        #画像中心
        center = tuple(np.array(image.shape[0:2]) / 2)
        for scale in range(6, 10):
            scale = 0.1 * scale
            #アフィン変換用行列
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            rot_image = cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_CUBIC)
            cv2.imwrite(str(i) + str(angle) + str(scale) + ".bmp", rot_image)
