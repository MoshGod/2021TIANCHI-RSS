#!/user/bin/env python
#-*- coding:utf-8 -*-

'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''

import shutil
import random
import os
import string

def copy_file(file_path, tar_dir):
    # pathDir = os.listdir(fileDir)

    # pathDir = os.listdir(file_dir)
    # sample = random.sample(pathDir, 200)

    for i in range(1, 3001):
        name = str(i).rjust(6, '0') + '.png'
        shutil.copyfile(file_path, tar_dir + name)
        # print(file_path, tar_dir + name)
        # os.remove(file_dir + name)


def origin_data_split():
    """
    将原数据集 image, label 分别存储
    """
    file_path = r'../../dataset/suichang_round1_train_210120'
    img_path = r'../data/train/img'
    label_path = r'../data/train/label'

    file_list = os.listdir(file_path)

    for file in file_list:
        if file[-3:] == 'tif':
            shutil.copyfile(file_path + '/' + file, img_path + '/' + file)
        else:
            shutil.copyfile(file_path + '/' + file, label_path + '/' + file)


def validation_split():
    origin_path = r'../data/train/'

    target_path = r'../data/validation/'

    img_list = os.listdir(origin_path + 'img')
    label_list = os.listdir(origin_path + 'label')

    sample = random.sample(img_list, 4017)  # 大数据集 98:1:1 划分

    for file in sample:

        shutil.copyfile(origin_path + 'img/' + file, target_path + 'img/' + file)
        os.remove(origin_path + 'img/' + file)

        shutil.copyfile(origin_path + 'label/' + file.replace('.tif', '.png'), target_path + 'label/' + file.replace('.tif', '.png'))
        os.remove(origin_path + 'label/' + file.replace('.tif', '.png'))



if __name__ == '__main__':
#     origin_data_split()
    validation_split()
#     copy_file('../test.png', '../data/results/')
    pass