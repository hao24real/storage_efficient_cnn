import torch.nn as nn

# if the number of output feature map and the size are the same as input
# then identity take precedence over the usage of this class in model definition
class SkipConn2d(nn.Module):
    # by official resnet paper, the skip connection is always a 1x1 convolution
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # enable skip connection to do projection onto feature map with lower resolution
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    