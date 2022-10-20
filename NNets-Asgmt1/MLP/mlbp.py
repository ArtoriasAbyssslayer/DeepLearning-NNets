import os
import torch
from torch import  nn as nn
from nnets-asgmt1.cifar10.py import load_cifar10

# Simple MultyLayer Network from Torch
# with back propagation segment

# Module == the base class of all neural network modules (so I paas the base class args)
class MLPNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(MLPNeuralNetwork)).__init__()
        self.flatten = nn.Flatten()

