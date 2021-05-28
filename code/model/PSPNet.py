#!/user/bin/env python    
#-*- coding:utf-8 -*-

import torch.nn as nn
# from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp

model_name = 'resnet34'#xception

class PSPNet(nn.Module):
    def __init__(self, model_name, in_channels, n_class):
        super().__init__()
        self.model = smp.PSPNet(# UnetPlusPlus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=in_channels,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    # @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x

print(smp.MAnet(encoder_weights=None))