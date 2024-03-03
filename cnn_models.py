import torch
import torch.nn as nn
from cnn_blocks import SkipConn2d, ResBlock

class DenseMax(nn.Module):
    def __init__(self):
        super(DenseMax, self).__init__()
        
         ### RES 1
         
        # 3x32x32 to 10x32x32
        self.rb1 = ResBlock(3, 10, 1)
        self.act1 = nn.ReLU()
        # 10x32x32 to 10x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        ### RES 2
        
        # 10x16x16 to 20x16x16
        self.rb2 = ResBlock(10, 20, 1)
        # 3x32x32 to 20x16x16 skip connection
        self.res1_sc1 = SkipConn2d(3, 20, 2)
        self.act2 = nn.ReLU()
        # 20x16x16 to 20x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                
        # 20x8x8 to 20x8x8
        self.rb3 = ResBlock(20, 20, 1)
        # 3x32x32 to 20x8x8 skip connection
        self.res1_sc2 = SkipConn2d(3, 20, 4)
        # 10x16x16 to 20x8x8 skip connection
        self.res2_sc1 = SkipConn2d(10, 20, 2)
        self.act3 = nn.ReLU()
        # 20x8x8 to 20x4x4
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten(1)
        # 20x4x4 affine operation to hidden layer with 100 neurons
        self.fc1 = nn.Linear(20 * 4 * 4, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.act4 = nn.ReLU()
        
        # affine operation to output layer with 10 neurons
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
       
        res1_sc1 = self.res1_sc1(x)
        res1_sc2 = self.res1_sc2(x)
        
        x = self.rb1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        res2_sc1 = self.res2_sc1(x)

        x = self.rb2(x)
        x = self.act2(x + res1_sc1)
        x = self.pool2(x)
        
        x = self.rb3(x)
        x = self.act3(x + res1_sc2 + res2_sc1)
        x = self.pool3(x)
                
        # flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x