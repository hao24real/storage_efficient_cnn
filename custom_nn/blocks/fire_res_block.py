
import torch.nn as nn
from .skip_conn import SkipConn2d
import torch

class FireResBlock(nn.Module):

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_conv1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn1 = nn.BatchNorm2d(squeeze_channels)
        self.squeeze_act1 = nn.ReLU()
        self.expand_conv1 = nn.Conv2d(squeeze_channels, expand1x1_channels,
                                   kernel_size=1)
        self.expand_bn1 = nn.BatchNorm2d(expand1x1_channels)
        self.expand_conv2 = nn.Conv2d(squeeze_channels, expand3x3_channels,
                                   kernel_size=3, padding=1)
        self.expand_bn2 = nn.BatchNorm2d(expand3x3_channels)
        self.shortcut = SkipConn2d(in_channels, expand1x1_channels + expand3x3_channels, 1)

    def forward(self, x):
        
        sc = self.shortcut(x)
        
        # squeeze layer
        x = self.squeeze_conv1(x)
        x = self.squeeze_act1(x)
        
        # expand layer
        x_1x1 = self.expand_conv1(x)
        x_1x1 = self.expand_bn1(x_1x1)
        
        x_3x3 = self.expand_conv2(x)
        x_3x3 = self.expand_bn2(x_3x3)
        
        # not using activation here for densenet implementation
        return torch.cat([x_1x1, x_3x3], 1) + sc
