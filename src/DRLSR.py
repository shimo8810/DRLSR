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

class ImageDataset(chainer.dataset.DatasetMixin):
    """docstring forImageDataset."""
    def __init__(self):
        data_paths = glob.glob('../images/demo_train_dataset/*')
        data_list = []
        for path in data_paths:
            data_list.append(path)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def get_example(self, i):
        image_input, image_label = np.load(self.data_list[i])
        return image_input, image_label

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
        self.train = True

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

        if self.train:
            #Training Phase
            self.loss = F.mean_squared_error(x, t)
            print("loss",self.loss.dtype)
            self.acc = F.accuracy(x, t)
            print("acc", self.acc)
            return self.loss
        else:
            return h

    def forward(self, x):
        pass


if __name__ == '__main__':
    #メインで呼ばれるときは学習Phaseで
    print("Training Phase ...")

    #引数 読み込み GPU 情報のみ
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    #データセット読み込み
    train_data = ImageDataset()
    #Trianer準備
    train_iter = chainer.iterators.SerialIterator(train_data, 1)
    test_iter = chainer.iterators.SerialIterator(train_data, 1)

    #モデル読み込み
    drlsr = DRLSRNet()
    optimizer = optimizers.SGD()
    optimizer.setup(drlsr)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (5, 'epoch'), out="result")

    trainer.extend(extensions.Evaluator(test_iter, drlsr, device=0))
    #trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(2, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    chainer.serializers.save_npz('model_final', drlsr)
chainer.serializers.save_npz('optimizer_final', optimizer)
