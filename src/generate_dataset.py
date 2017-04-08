import sys
import numpy as np
import cv2
import glob
import chainer
from chainer import datasets

class ImageDataset(chainer.dataset.DatasetMixin):
    """docstring forImageDataset."""
    def __init__(self):
        i = 1
        data_paths = glob.glob('../images/demo_train_dataset/*')
        pairs = []
        for path in data_paths:
            pairs.append(np.load(path))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        image_input, image_label = self.pairs[i]
        return image_input, image_label
if __name__ == '__main__':
    image_dataset = ImageDataset()
    print(len(image_dataset))
    image_dataset.get_example(2)
