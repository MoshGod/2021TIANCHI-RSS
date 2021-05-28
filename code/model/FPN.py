#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
from torch import nn
import segmentation_models_pytorch as smp

model_name = 'efficientnet-b6'#xception

class FPN(nn.Module):
    def __init__(self, model_name, in_channels, n_class):
        super().__init__()
        self.model = smp.FPN(# UnetPlusPlus
                encoder_name=model_name,
                encoder_weights="imagenet",     
                in_channels=in_channels,        
                classes=n_class,
            )
    # @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x


# from pytorch_toolbelt.modules import encoders as E
# from pytorch_toolbelt.modules import decoders as D

# class SEResNeXt50FPN(nn.Module):
#     def __init__(self, num_classes, fpn_channels):
#         super().__init__()
#         self.encoder = E.SEResNeXt50Encoder()
#         self.decoder = D.FPNCatDecoder(self.encoder.channels, fpn_channels)
#         self.logits = nn.Conv2d(self.decoder.channels[0], num_classes, kernel_size=1)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return self.logits(x[0])

if __name__ == '__main__':
    x = torch.rand((2,4,256,256))
    print(SEResNeXt50FPN(10, 4)(x).shape) # 依然是torch.Size([3, 21, 320, 480])