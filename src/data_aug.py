#画像データセットを増幅させる
import os
import numpy as np
import cv2

# 最初の91枚の画像をカサ増しする
#91-iamgeのパス
image_path = '../images/1_images/'
#保存用ディレクトリのパス
save_path = '../images/1_images_aug/'
image_names = os.listdir(image_path)

for i in image_names:
    #画像パス
    image_name = image_path + str(i)
    #画像ファイル
    image = cv2.imread(image_name)
    for angle in range(0, 360, 90):
        #画像中心
        center = tuple(np.array(image.shape[0:2]) / 2)
        for scale in range(6, 11):
            scale = 0.1 * scale
            #アフィン変換用行列
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            size = tuple(map(lambda x : int(x * scale), image.shape[0:2]))
            rot_image = cv2.warpAffine(image, rot_mat, size, flags=cv2.INTER_CUBIC)
            cv2.imwrite(save_path + \
                        i.split('.')[0] + \
                        '_ang' + str(angle) + \
                        '_sca' + str(int(scale * 10)) + \
                        ".bmp", rot_image)
