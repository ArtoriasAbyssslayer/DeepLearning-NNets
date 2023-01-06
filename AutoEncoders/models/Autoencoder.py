import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Pick Device to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
from BasicLayers import DenseLayerPT


# This refers to the latent size 
encoding_dim  = 32

# Define Encoder part
class Encoder(nn.Module):
    '''
        Inference Encoding Network implemented with MLP. 
    '''
# Define Decoder part


# Define AutoEncoder Model
class AutoEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,latent_dim,activation=F.relu):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        # input dim 784 we gradually descent this with 3 layers
        self.encoder_l1 = DenseLayerPT(input_dim,hidden_dim,activation)
        self.encoder_l2 = DenseLayerPT(hidden_dim,)
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
        # Simple Autoencoder reconstruction Loss  L_2 norm of generated and input image
        return tf.reduce_mean(tf.square(self.X - self.X_hat))

    
    

