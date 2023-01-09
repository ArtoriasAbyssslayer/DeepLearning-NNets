import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Pick Device to train the model
import sys
sys.path.append('models')
import BasicLayers

device = "cuda" if torch.cuda.is_available() else "cpu"



# Define Simple AutoEncoder Model
class AutoEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim=300,latent_dim=64,activation=F.relu,bernoulli_input=None,gaussian_blurred_input=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bernoulli_input = bernoulli_input
        self.gaussian_blurred_input = gaussian_blurred_input
        # input dim 784 we gradually descent this with 2 Dense Linear layers
        self.encoder_l1 = BasicLayers.DenseLayerPT(input_dim,hidden_dim,activation)
        self.encoder_l2 = BasicLayers.DenseLayerPT(hidden_dim,latent_dim,activation)
        self.decoder = BasicLayers.DenseLayerPT(latent_dim,input_dim,activation)
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
        x = X.view(-1,self.input_dim)
        self.Z = self.encoder.forward(x)
        return self.Z
    def generate(self,Z):
        self.X_hat = self.decoder.forward(Z)
        return self.X_hat

    def tensor_to_numpyImg(self,img):
        bin = self.bernoulli_input
        gas = self.gaussian_blurred_input

        if bin:
            img = img.view(bin.shape)
            return (img > 0.5).to(img.dtype).reshape(img.size(0), 28, 28).cpu().detach().numpy()
        elif gas:
            return img.reshape(img.size(0), 28, 28).cpu().detach().numpy()
        else:
            return img.reshape(img.size(0), 28, 28).cpu().detach().numpy()
    def rec_loss(self):
        # Simple Autoencoder reconstruction Loss  L_2 norm of generated and input image
        return tf.reduce_mean(tf.square(self.X - self.X_hat))

    
    

