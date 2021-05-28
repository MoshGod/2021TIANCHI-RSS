#!/user/bin/env python    
#-*- coding:utf-8 -*-

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import resnet18


class FCN(nn.Module):
    def __init__(self, in_channels, num_classes, attention = False):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)
        # if attention:
        #     self.conv2 = nn.Sequential(*list(ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes).children())[:-2])
        # else:
        self.conv2 = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        self.conv3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)
        nn.init.xavier_normal_(self.conv1.weight.data, gain=1)
        nn.init.xavier_normal_(self.conv3.weight.data, gain=1)
        self.conv4.weight.data = self.bilinear_kernel(num_classes, num_classes, 64)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        return x1

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size+1)//2
        if kernel_size%2 == 1:
            center = factor-1
        else:
            center = factor-0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1-abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
        weight = np.zeros((in_channels,out_channels, kernel_size,kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        weight = torch.from_numpy(weight)
        weight.requires_grad = True
        return weight


if __name__ == '__main__':
    x = torch.rand((2,4,256,256))
    print(FCN(4, 10, True)(x).shape) # 依然是torch.Size([3, 21, 320, 480])

