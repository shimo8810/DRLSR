import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, utils, \
                    Link, Chain, ChainList
from chainer import training
from chainer import datasets
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import argparse
import cv2
import sys
import cv2
import glob

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 2
GRADIENT_CLIPPING = 0.1
MAX_EPOCH = 100

class ImageDataset(chainer.dataset.DatasetMixin):
    """docstring forImageDataset."""
    def __init__(self, is_train=True):
        if is_train:
            data_paths = glob.glob('../images/demo_train_dataset/*')
        else:
            data_paths = glob.glob('../images/demo_test_dataset/*')
        data_list = []
        for path in data_paths:
            data_list.append(path)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def get_example(self, i):
        image_input, image_label = np.load(self.data_list[i])
        return image_input/255.0, image_label/255.0

class DRLSRNet(Chain):
    """
    DRLSR network
    入力画像は(41, 41 ,1)の輝度情報画像
    """
    def __init__(self):
        super(DRLSRNet, self).__init__(
            conv1_3 = L.Convolution2D(1, 8, ksize=3, stride=1, pad=1, bias=0),
            conv1_5 = L.Convolution2D(1, 8, ksize=5, stride=1, pad=2, bias=0),
            conv1_9 = L.Convolution2D(1, 8, ksize=9, stride=1, pad=4, bias=0),
            conv2 = L.Convolution2D(24, 16, ksize=1, stride=1, pad=0, bias=0),
            conv22= L.Convolution2D(16, 16, ksize=3, stride=1, pad=1, bias=0),
            conv23= L.Convolution2D(16, 16, ksize=1, stride=1, pad=0, bias=0),
            conv3_3 = L.Convolution2D(16, 8, ksize=3, stride=1, pad=1, bias=0),
            conv3_5 = L.Convolution2D(16, 8, ksize=5, stride=1, pad=2, bias=0),
            conv3_9 = L.Convolution2D(16, 8, ksize=9, stride=1, pad=4, bias=0),
            conv4 = L.Convolution2D(24, 1, ksize=1, stride=1, pad=0, bias=0)
        )
        self.is_train = True

    def __call__(self, x, t):
        #forward network
        #inception layer 1
        h = F.concat((F.relu(self.conv1_3(x)), \
                      F.relu(self.conv1_5(x)), \
                      F.relu(self.conv1_9(x))), axis=1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv22(h))
        h = F.relu(self.conv23(h))
        #inception layer 2
        h = F.concat((F.relu(self.conv3_3(h)), \
                      F.relu(self.conv3_5(h)), \
                      F.relu(self.conv3_9(h))), axis=1)
        h = F.relu(self.conv4(h))
        h = h + x

        if self.is_train:
            #Training Phase
            self.loss = F.mean_squared_error(h, t)
            chainer.report({'loss': self.loss}, self)
            return self.loss
        else:
            print("not train")
            return h

if __name__ == '__main__':
    #メインで呼ばれるときは学習Phaseで
    print("Training Phase ...")

    #引数 読み込み GPU 情報のみ
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    #モデル読み込み
    drlsr = DRLSRNet()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        drlsr.to_gpu()

    #データセット読み込み
    train_data = ImageDataset(is_train=True)
    test_data = ImageDataset(is_train=False)
    #Trianer準備
    train_iter = chainer.iterators.MultiprocessIterator(train_data, TRAIN_BATCH_SIZE)
    test_iter = chainer.iterators.MultiprocessIterator(test_data, TEST_BATCH_SIZE, repeat=False, shuffle=False)

    #optimizer 準備
    optimizer = optimizers.SGD()
    optimizer.setup(drlsr)
    #optimizer.add_hook(chainer.optimizer.GradientClipping(GRADIENT_CLIPPING))

    #Trainer 準備
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    #trainer = training.Trainer(updater, (MAX_EPOCH, 'epoch'), out="result")
    trainer = training.Trainer(updater, (100, 'iteration'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, drlsr, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'iteration'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()

    chainer.serializers.save_npz('result/model_final', drlsr)
    chainer.serializers.save_npz('result/optimizer_final', optimizer)
