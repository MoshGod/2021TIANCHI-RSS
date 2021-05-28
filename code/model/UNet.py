

import torch
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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        '''卷积  encoder'''
        self.conv1 = DoubleConv(in_channels, 64)  # 256, 256, 64
        self.pool1 = nn.MaxPool2d(2)  # 128, 128, 64
        self.conv2 = DoubleConv(64, 128)  # 128, 128, 128
        self.pool2 = nn.MaxPool2d(2)  # 64, 64, 128
        self.conv3 = DoubleConv(128, 256)  # 64, 64, 256
        self.pool3 = nn.MaxPool2d(2)  # 32, 32, 256
        self.conv4 = DoubleConv(256, 512)  # 32, 32, 512
        self.pool4 = nn.MaxPool2d(2)  # 16, 16, 512
        self.conv5 = DoubleConv(512, 1024)  # 16, 16, 1024

        '''上采样 + 卷积 = 反卷积  decoder'''
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 32, 32, 512
        # self.merge6 = torch.cat([self.conv4, self.up6], dim=1) # 32, 32, 512 + 512 = 1024
        self.conv6 = DoubleConv(1024, 512)  # 32, 32, 512

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 64, 64, 256
        # self.merge7 = torch.cat((self.conv3, self.up7), dim=1)  # 64, 64, 256 + 256 = 512
        self.conv7 = DoubleConv(512, 256)  # 64, 64, 256

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 128, 128, 128
        # self.merge8 = torch.cat((self.conv2, self.up8), dim=1)  # 128, 128, 128 + 128 = 256
        self.conv8 = DoubleConv(256, 128)  # 128, 128, 128

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 256, 256, 64
        # self.merge9 = torch.cat((self.conv1, self.up9), dim=1)  # 256, 256, 64 + 64 = 128
        self.conv9 = DoubleConv(128, 64)  # 256, 256, 64

        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)

        logits = nn.Sigmoid()(conv10)

        return logits


if __name__ == '__main__':
    img = torch.rand((1, 4, 256, 256))
    model = UNet(4, 10)
    # print(model)
    out = model(img)
    print(type(out), out.shape, torch.squeeze(out).shape)