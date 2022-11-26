import torch
import torch.nn as nn
import torch.nn.functional as F
# Simple MultyLayer Network from Torch
# It has Convolutional and Dense layers
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(3,6,5)
        self.max_pooling = nn.MaxPool2d(2,2)
        self.conv_layer_2 = nn.Conv2d(6,16,5)
        self.dense_1 = nn.Linear(16*5*5,120)
        self.dense_2 = nn.Linear(120,84)
        self.dense_3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.max_pooling(F.relu(self.conv_layer_1(x)))
        x = self.max_pooling(F.relu(self.conv_layer_2(x)))
        x = F.relu(self.dense_1)
        x = F.relu(self.dense_2)
        x = F.softmax(self.dense_2)
        return x