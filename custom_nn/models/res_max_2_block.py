import torch.nn as nn
from ..blocks import SkipConn2d, ResBlock

class ResMaxTwoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
         ### RES 1
         
        # 3x32x32 to 10x32x32
        self.rb1 = ResBlock(3, 10, 1)
        self.act1 = nn.ReLU()
        # 10x32x32 to 10x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        ### RES 2
        
        # 10x16x16 to 20x16x16
        self.rb2 = ResBlock(10, 20, 1)
        self.act2 = nn.ReLU()
        # 20x16x16 to 20x4x4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                
        self.flatten = nn.Flatten(1)
        # 20x8x8 affine operation to hidden layer with 100 neurons
        self.fc1 = nn.Linear(20 * 8 * 8, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.act4 = nn.ReLU()
        
        # affine operation to output layer with 10 neurons
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        
        x = self.rb1(x)
        x = self.act1(x)
        x = self.pool1(x)
    
        x = self.rb2(x)
        x = self.act2(x)
        x = self.pool2(x)
                
        # flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x