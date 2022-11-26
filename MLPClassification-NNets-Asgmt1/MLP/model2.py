import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutMLP(nn.Module):
    def __init__ (self):