import numpy as np
import chainer
import os
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

#parameter
#Train バッチサイズ
TRAIN_BATCH_SIZE = 64
#Test バッチサイズ
TEST_BATCH_SIZE = 2
GRADIENT_CLIPPING = 0.1
#MAX_EPOCH = 300000
MAX_ITER = 300000
interval =  10000

class ImageDataset(chainer.dataset.DatasetMixin):
    """docstring forImageDataset."""
    def __init__(self, data_set_type='training'):
        '''
        training : pre-training用の画像セット
        tuning : fine-tuning用の画像セット
        test : テスト用の画像セット
        '''
        if 'training' == data_set_type:
            data_paths = glob.glob('../images/demo_train_dataset/*')
        elif 'tuning' == data_set_type:
            data_paths = glob.glob('../images/general_train_dataset/*')
        elif 'test' == data_set_type:
            data_paths = glob.glob('../images/demo_test_dataset/*')

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
    """
    def __init__(self):
        w_init = chainer.initializers.HeNormal()
        super(DRLSRNet, self).__init__(
            conv1_3 = L.Convolution2D(1, 8, ksize=3, stride=1, pad=1, bias=0, initialW=w_init),
            conv1_5 = L.Convolution2D(1, 8, ksize=5, stride=1, pad=2, bias=0, initialW=w_init),
            conv1_9 = L.Convolution2D(1, 8, ksize=9, stride=1, pad=4, bias=0, initialW=w_init),
            conv2 = L.Convolution2D(24, 16, ksize=1, stride=1, pad=0, bias=0, initialW=w_init),
            conv22= L.Convolution2D(16, 16, ksize=3, stride=1, pad=1, bias=0, initialW=w_init),
            conv23= L.Convolution2D(16, 16, ksize=1, stride=1, pad=0, bias=0, initialW=w_init),
            conv3_3 = L.Convolution2D(16, 8, ksize=3, stride=1, pad=1, bias=0, initialW=w_init),
            conv3_5 = L.Convolution2D(16, 8, ksize=5, stride=1, pad=2, bias=0, initialW=w_init),
            conv3_9 = L.Convolution2D(16, 8, ksize=9, stride=1, pad=4, bias=0, initialW=w_init),
            conv4 = L.Convolution2D(24, 1, ksize=1, stride=1, pad=0, bias=0, initialW=w_init)
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
            print("SR Phase")
            return h

if __name__ == '__main__':
    #メインで呼ばれるときは学習Phaseで
    '''
    gpu: 0->gpu, otherwise -> cpu
    phase: 1 -> pre-training, otherwise -> fine-tuning
    snapshot: スナップショットファイル
    model: モデルファイル
    '''
    #引数 読み込み
    #読み込む情報はgpu, 学習フェイズ,
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--phase', type=int, default=1)#1 -> pre-training, 0(not 1) -> fine-tuning
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--model', action='store_true', default=False)

    args = parser.parse_args()

    #モデル読み込み
    drlsr = DRLSRNet()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        drlsr.to_gpu()

    #データセット読み込み
    #学習レートの設定
    if args.phase:
        print("#Pre-Training Phase")
        train_data = ImageDataset(data_set_type='training')
        lr = 0.1
    else:
        print("#Fine-Tuning Phase")
        train_data = ImageDataset(data_set_type='tuning')
        lr = 1.0
    test_data = ImageDataset(data_set_type='test')

    #Trianer準備
    train_iter = chainer.iterators.MultiprocessIterator(train_data, TRAIN_BATCH_SIZE)
    test_iter = chainer.iterators.MultiprocessIterator(test_data, TEST_BATCH_SIZE, repeat=False, shuffle=False)

    #optimizer 準備
    optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(drlsr)
    optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    #Trainer 準備
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (MAX_ITER, 'iteration'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, drlsr, device=args.gpu), trigger=(interval, 'iteration'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(interval, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(interval, 'iteration')))
    trainer.extend(extensions.observe_lr(), trigger=(interval, 'iteration'))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'lr']), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.snapshot:
        if args.phase:
            #pre training
            serializers.load_npz('./result/snapshot_iter_' +str(args.snapshot) , trainer)
        else:
            #fine tuning
            serializers.load_npz('./result/snapshot_iter_' +str(args.snapshot) , trainer)
    elif not args.phase:
        #not snapshot and not pre(finetune) -> プレの最終ファイル読み込みが必要
        serializers.load_npz('./result/model_pre_training.npz', drlsr)

    print("#Start Learning")
    trainer.run()
    print("#Saving model.")

    if args.phase:
        chainer.serializers.save_npz('result/model_pre_training.npz', drlsr)
    else:
        chainer.serializers.save_npz('result/model_fine_tuning.npz', drlsr)

    print('#Completed')
