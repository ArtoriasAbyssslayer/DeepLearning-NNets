import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutNet(nn.Module):
    def __init__ (self):