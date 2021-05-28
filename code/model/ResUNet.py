

import torch
from torchvision import models
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_unit(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()

        '''卷积  encoder'''

        self.base_model = models.resnet18(True)
        self.base_layers = list(self.base_model.children())

        # 重写 ResNet 第一层以自定义输入通道数
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self.base_layers[1],
            self.base_layers[2]
        )
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]



        '''上采样 + 卷积 = 反卷积  decoder'''
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(96, 64)

        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        encoder1 = self.layer1(x)  # 256, 256, 64
        encoder2 = self.layer2(encoder1)  # 128, 128, 64
        encoder3 = self.layer3(encoder2)  # 64, 64, 128
        encoder4 = self.layer4(encoder3)  # 32, 32, 256
        f = self.layer5(encoder4)  # 16, 16, 512
        # print(encoder1.shape, encoder2.shape, encoder3.shape, encoder4.shape, f.shape)


        up6 = self.up6(f)  # 32, 32, 256
        merge6 = torch.cat([encoder4, up6], dim=1)  # 32, 32, 512
        conv6 = self.conv6(merge6)  # 32, 32, 256


        up7 = self.up7(conv6)  # 64, 64, 128
        merge7 = torch.cat([encoder3, up7], dim=1)  # 64, 64, 256
        conv7 = self.conv7(merge7)  # 64, 64, 128


        up8 = self.up8(conv7)  # 128, 128, 64
        merge8 = torch.cat([encoder2, up8], dim=1)  # 128, 128, 128
        conv8 = self.conv8(merge8)  # 128, 128, 64

        up9 = self.up9(conv8)  # 256, 256, 32
        merge9 = torch.cat([encoder1, up9], dim=1)  # 256, 256, 96
        conv9 = self.conv9(merge9)  # 256, 256, 32



        conv10 = self.conv10(conv9)

        logits = torch.sigmoid(conv10)

        return logits


if __name__ == '__main__':
    img = torch.rand((1, 4, 256, 256))
    model = ResUNet(4, 10)
    # print(model)
    out = model(img)
    print(type(out), out.shape, torch.squeeze(out).shape)