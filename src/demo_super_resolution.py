from chainer import serializers
import numpy as np
import cv2
import glob
import DRLSR

def MSE(image1, image2):
    err = np.sum((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    return err / (image1.shape[0] * image1.shape[1])

def PSNR(image1, image2):
    m = np.max(np.array([np.max(image1), np.max(image2)]))
    psnr = 10 * np.log10(m * m / MSE(image1, image2))
    return psnr


if __name__ == '__main__':
    up_scale = 3
    #モデル読み込み
    model = DRLSR.DRLSRNet()
    serializers.load_npz("result/model_final", model)
    model.is_train = False

    #画像読み込み
    image = '../images/Set16/flowers.bmp'
    image = cv2.imread(image).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    size = np.array(image.shape)
    size = size - size % up_scale
    #ラベル画像
    image_label = image[0:size[0], 0:size[1]]
    height, width = image_label.shape

    buf = cv2.resize(image_label, (width//up_scale, height//up_scale), \
        interpolation=cv2.INTER_CUBIC)
    image_input = cv2.resize(buf, (width, height), \
        interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('result/input.bmp', image_input)
    print(PSNR(image_label, image_input))
    image_input = image_input[np.newaxis, np.newaxis, :, :] / 255.0
    image_output = model(image_input, image_label).data * 255.0
    image_output = image_output.reshape((height, width))
    cv2.imwrite('result/output.bmp', image_output)
    cv2.imwrite('result/label.bmp', image_label)
    print(PSNR(image_label, image_output))
