#!/user/bin/env python
#-*- coding:utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import os
import torch
import numpy as np

def make_dataset(root1, root2):
    '''
    @func: 读取数据, 存入列表
    @root1: src 路径
    @root2: label 路径
    '''
    imgs = []                                    #遍历文件夹, 添加图片和标签图片路径到列表
    for i in range(650, 811):
        img = os.path.join(root1, "%s.png" % i)
        mask = os.path.join(root2, "%s.png" % i)
        imgs.append((img, mask))
    return imgs

class SensingDataset(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.ds = make_dataset(img_path, label_path)

    def __getitem__(self, index):
        img_list = os.listdir(self.img_path)
        label_list = os.listdir(self.label_path)
        img = Image.open(self.img_path + '/' + img_list[index])
        label = Image.open(self.label_path + '/' + label_list[index])


        img = self.transform(img)

        label = np.array(label) - 1   # 1-10 => 0-9
        label = torch.from_numpy(label).long()

        return img, label

    def __len__(self):
        return len(os.listdir(self.img_path))


if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # PIL Image → Tensor
    ])
    RSDataset = SensingDataset(r'../data/validation/img', r'../data/validation/label', transform)
    # print(len(RSDataset))

    train_dataset = SensingDataset(r'../data/train/img', r'../data/train/label', transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    channel_total = {
        "r": 0, "g": 0, "b": 0, "n": 0
    }
    channel_min = {
        "r": 0, "g": 0, "b": 0, "n": 0
    }
    channel_max = {
        "r": 0, "g": 0, "b": 0, "n": 0
    }
    for img, _ in train_loader:
        r = img[:, 0, :, :]
        g = img[:, 1, :, :]
        b = img[:, 2, :, :]
        n = img[:, 3, :, :]
        # print(r.size())
        channel_min['r'] = torch.min(r)
        channel_min['g'] = torch.min(g)
        channel_min['b'] = torch.min(b)
        channel_min['n'] = torch.min(n)
        channel_max['r'] = torch.max(r)
        channel_max['g'] = torch.max(g)
        channel_max['b'] = torch.max(b)
        channel_max['n'] = torch.max(n)

    print(channel_min)
