import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Pick Device to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
from BasicLayers import DenseLayerPT


# This refers to the latent size 
encoding_dim  = 32
# Define AutoEncoder
class AutoEncoder(input_dim,encoding_dim):
    def __init__(self,input_dim,encoding_dim,activation=F.relu):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = DenseLayerPT(input_dim,encoding_dim,activation)
        self.decoder = DenseLayerPT(encoding_dim,input_dim,activation)
    def forward(self,X):
        self.X = X
        self.Z = self.encoder.forward(X)
        self.X_hat = self.decoder.forward(self.Z)
        return self.X_hat
    def backward(self,dX_hat):
        self.dZ = self.decoder.backward(dX_hat)
        self.dX = self.encoder.backward(self.dZ)
        return self.dX
    def update(self,lr):
        self.encoder.update(lr)
        self.decoder.update(lr)
    def encode(self,X):
        enc =
    def rec_loss(self):
        return tf.reduce_mean(tf.square(self.X - self.X_hat))

    
    

