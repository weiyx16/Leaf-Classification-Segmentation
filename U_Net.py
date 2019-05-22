'''

Leaf Vein Segmentation based on U-Net or FCN
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# Network Structure.function
Adapted from https://github.com/milesial/Pytorch-UNet/
'''

import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad

class U_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(U_Net, self).__init__()
        self.conv_encode_l1 = double_conv(n_channels, 64)
        self.down_maxpooling_l1 = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode_l2 = double_conv(64, 128)
        self.down_maxpooling_l2 = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode_l3 = double_conv(128, 256)
        self.down_maxpooling_l3 = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode_l4 = double_conv(256, 512)
        self.down_maxpooling_l4 = nn.MaxPool2d(kernel_size = 2)
        self.bottle_neck_encode = double_conv(512, 1024)

        self.conv_decode_l1 = Up_Sample(1024, 512)
        self.conv_decode_l2 = Up_Sample(512, 256)
        self.conv_decode_l3 = Up_Sample(256, 128)
        self.conv_decode_l4 = Up_Sample(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, in_put):
        l1 = self.conv_encode_l1(in_put)
        l2_in = self.down_maxpooling_l1(l1)
        l2 = self.conv_encode_l2(l2_in)
        l3_in = self.down_maxpooling_l2(l2)
        l3 = self.conv_encode_l3(l3_in)
        l4_in = self.down_maxpooling_l1(l3)
        l4 = self.conv_encode_l4(l4_in)
        l5_in = self.down_maxpooling_l4(l4)

        bottle = self.bottle_neck_encode(l5_in)

        up = self.conv_decode_l1(bottle, l4)
        up = self.conv_decode_l2(up, l3)
        up = self.conv_decode_l3(up, l2)
        before_out = self.conv_decode_l4(up, l1)
        output = self.outc(before_out)

        return torch.sigmoid(output)


class double_conv(nn.Module):
    '''(conv => BN => ReLU => BN => ReLU
       `one layer of down-sample in U_Net` 
    '''
    def __init__(self, in_channels, out_channels, is_Upsample = False):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3), #, padding=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3), #, padding=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            #Or nn.LeakyReLU # inplace True or not
        )

    def forward(self, in_put):
        output = self.conv(in_put)
        return output


class Up_Sample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_Sample, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # interpolate()
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, up_layer, skip_layer):
        up_layer = self.up(up_layer)
        # input is CHW
        # or you can pad the up_layer...

        diffY = skip_layer.size()[2] - up_layer.size()[2]
        diffX = skip_layer.size()[3] - up_layer.size()[3]
        skip_layer = pad(skip_layer, ( - diffX // 2, - (diffX - diffX//2), - diffY // 2, - (diffY - diffY//2)))
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        cat_out = torch.cat((skip_layer, up_layer), dim=1)
        output = self.conv(cat_out)
        return output


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, in_put):
        output = self.conv(in_put)
        return output