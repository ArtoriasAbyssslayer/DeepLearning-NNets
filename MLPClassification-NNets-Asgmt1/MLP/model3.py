import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = nn.Linear(1024, 512)
        self.dropout_1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(512, 256)
        self.dropout_2 = nn.Dropout(0.2)
        self.dense_3 = nn.Linear(256, 128)
        self.dense_4 = nn.Linear(128, 64)
        self.dense_5 = nn.Linear(64, 10)
        
    def forward(self, x):
        layer_1_out = F.relu(self.dense_1(x))
        layer_2_out = F.relu(self.dropout_1(layer_1_out))
        layer_3_out = F.relu(self.dense_2(layer_2_out))
        layer_4_out = F.relu(self.dropout_2(layer_3_out))
        layer_5_out = F.relu(self.dense_3(layer_4_out))
        layer_6_out = F.relu(self.dense_4(layer_5_out))
        layer_7_out = F.softmax(self.dense_5(layer_6_out))
        
        hidden_layers = [layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out]
        return layer_7_out
    