import os
import sys
import nupy as np
import chainer
from chainer import datasets

class ImageDataset(chainer.dataset.DatasetMixin):
    """docstring forImageDataset."""
    def __init__(self, normalize=True, flatten=True, train=True, max_size=200):
        superImageDataset, self).__init__()
        self._normalize = normalize
        self._flatten = flatten
        self._train = train
        self._max_size = max_size
