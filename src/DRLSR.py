import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, utils, \
                    Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class DRLSR(Chain):
    """docstring forDRLSR."""
    def __init__(self):
        super(DRLSR, self).__init__(
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

    def __call__(self, x, y):
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    print("Training ...")
