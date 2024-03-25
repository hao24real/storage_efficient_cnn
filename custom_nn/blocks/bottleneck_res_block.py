import torch.nn as nn
from .skip_conn import SkipConn2d

""" the purpose of BottleneckResBlock is to reduce the number of parameters while 
    increasing the depth of the network. This is done by using 1x1 convolutions to
    reduce the number of channels before the 3x3 convolution and then using another
    1x1 convolution to increase the number of channels back to the original number.
"""
class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, bn_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, bn_channels, 
                                kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bn_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_channels, bn_channels, 
                                kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bn_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(bn_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = SkipConn2d(in_channels, out_channels, stride)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x + shortcut