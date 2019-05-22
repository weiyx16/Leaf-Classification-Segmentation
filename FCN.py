
'''

Leaf Vein Segmentation based on U-Net or Resnetbased-FCN
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# FCN Network Structure.function

'''

import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad

class FCN(nn.Module):
    def __init__(self, ResNet, n_classes, output_size):
        super(FCN, self).__init__()
        #取掉model的后两层
        in_channel = 2048
        self.resnet_layer = nn.Sequential(*list(ResNet.children())[:-2])
        self.up_1 = nn.ConvTranspose2d(in_channel, 512, 2, stride=2)
        self.up_2 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.up_3 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.conv = nn.Conv2d(32, n_classes, 1)
        self.upsample = nn.Upsample(scale_factor=output_size/(7*2*2*2), mode='bilinear', align_corners=True)
        
    def forward(self, in_put):
        x = self.resnet_layer(in_put)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.conv(x)
        output = self.upsample(x)
        
        return torch.sigmoid(output)