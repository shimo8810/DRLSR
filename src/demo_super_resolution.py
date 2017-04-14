from chainer import serializers
import numpy as np
import cv2
import glob
import DRLSR

def PSNR(image1, image2):
    mse = np.sum((image1.astype(np.float32) - image2.astype(np.float32)) ** 2) / (image1.shape[0] * image1.shape[1])
    psnr = 10 * np.log10((255.0 **2) / mse)
    return psnr


if __name__ == '__main__':
    up_scale = 3
    #モデル読み込み
    model = DRLSR.DRLSRNet()
    serializers.load_npz("result/model_final_test64.npz", model)
    model.is_train = False

    #画像読み込み
    image = '../images/Set16/lenna.bmp'
    image = cv2.imread(image).astype(np.float32)
    #画像サイズ crop
    size = np.array(image.shape) - np.array(image.shape) % up_scale
    image = image[0:size[0], 0:size[1], :]
    #Y情報
    image_y  = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #CbCr情報
    image_cbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 1:]

    #超解像
    height, width = image_y.shape
    buf = cv2.resize(image_y, (width//up_scale, height//up_scale), interpolation=cv2.INTER_CUBIC)
    image_y_bic = cv2.resize(buf, (width, height), interpolation=cv2.INTER_CUBIC)
    image_input = image_y_bic[np.newaxis, np.newaxis, :, :] * (1 / 255.0)
    image_output = model(image_input, image_y).data * 255.0
    image_y_drlsr = image_output.reshape((height, width))
    #輝度情報画像保存
    cv2.imwrite('result/demo/image_y.bmp', image_y)
    cv2.imwrite('result/demo/image_y_drlsr.bmp', image_y_drlsr)
    cv2.imwrite('result/demo/image_y_bic.bmp', image_y_bic)
    #カラー化
    image_color_drlsr = np.concatenate([image_y_drlsr[:,:,np.newaxis], image_cbcr], axis=2)
    image_color_bic = np.concatenate([image_y_bic[:,:,np.newaxis], image_cbcr], axis=2)
    cv2.imwrite('result/demo/image_color.bmp', image)
    cv2.imwrite('result/demo/image_color_drlsr.bmp', cv2.cvtColor(image_color_drlsr, cv2.COLOR_YCR_CB2BGR))
    cv2.imwrite('result/demo/image_color_bic.bmp', cv2.cvtColor(image_color_bic, cv2.COLOR_YCR_CB2BGR))
