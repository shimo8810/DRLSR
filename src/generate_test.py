import numpy as np
import cv2
import os

#parameter
SIZE_INPUT = 41
SIZE_LABEL = 41
SCALE = 3
STRIDE = 14

image_path = '../images/1_images_aug/t1_ang0_sca10.bmp'
image_paths = list()
image_paths.append(image_path)
#初期化
data = np.zeros((SIZE_INPUT, SIZE_INPUT, 1, 1))
label = np.zeros((SIZE_LABEL, SIZE_LABEL, 1, 1))
padding = np.abs(SIZE_INPUT - SIZE_LABEL) / 2
count = 0

#データ生成
for i in image_paths:
    image = cv2.imread(i)
