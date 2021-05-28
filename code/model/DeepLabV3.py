#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50

def DeepLabV3(in_channels, num_class):
    dpv3 = deeplabv3_resnet50(pretrained = False,
                              progress = True,
                              num_classes = num_class,
                              aux_loss = None)
    model = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=1), dpv3)
#     model = nn.Sequential(dpv3)
    return model

if __name__ == '__main__':
    x = torch.rand((2, 4, 256, 256))
    print(DeepLabV3(4, 10)(x)['out'].shape)
