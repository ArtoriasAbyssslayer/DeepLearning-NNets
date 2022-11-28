import torch
import torch.nn 
import torch.nn.functional as F
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
NUM_CLASSES = 10
n_hidden_1 = 1024
class DenseMLP(torch.nn.Module):
    
    def __init__(self,activation):
        self.activation = activation
        super(DenseMLP,self).__init__()
        self.dense_1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,n_hidden_1)
        self.fc1_drop = torch.nn.Dropout(0.2)
        self.dense_2 = torch.nn.Linear(n_hidden_1,n_hidden_1//8)
        self.fc2_drop = torch.nn.Dropout(0.2)
        self.dense_3 = torch.nn.Linear(128, 32)
        self.dense_4 = torch.nn.Linear(32, NUM_CLASSES)
        
    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        if self.activation == 'relu':
            x = F.relu(self.dense_1(x))
            x = self.fc1_drop(x)
            x = F.relu(self.dense_2(x))
            x = self.fc2_drop(x)
            x = F.relu(self.dense_3(x))
            x = F.softmax(self.dense_4(x))
        elif self.activation == "sigmoid":
            x = F.sigmoid(self.dense_1(x))
            x = self.fc1_drop(x)
            x = F.sigmoid(self.dense_2(x))
            x = self.fc2_drop(x)
            x = F.sigmoid(self.dense_3(x))
            x = F.sigmoid(self.dense_4(x))
        return x