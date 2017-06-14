'''
    分裂されたnpyファイルを結合するプログラム
    並列化はされていない
    コマンドライン引数
        --data: datasetの名前
'''
import numpy as np
import os
import sys
from os import path
from tqdm import tqdm
import json
import argparse

size = 41
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
args = parser.parse_args()

if not args.data in ['set14', 'yang91', 'general100', 'mini']:
    exit()
else:
    data_name = args.data

with open('config.json', 'r') as f:
    config = json.load(f)

image_path = config['path_{}_npy'.format(data_name)]

path_list = os.listdir(image_path)
length = len(path_list)
x_data = np.zeros((length,1,41,41))
y_data = np.zeros((length,1,41,41))
print(length)

for i, p in enumerate(tqdm(path_list)):
    x, y = np.load(path.join(image_path, p))
    x_data[i, :, :, :] = x
    y_data[i, :, :, :] = y
np.save('../data_set/{}_input.npy'.format(data_name), x_data)
np.save('../data_set/{}_label.npy'.format(data_name), y_data)
print("#complete concat npy files!")
