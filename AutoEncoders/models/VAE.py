import torch
import torch.nn as nn
import torch.nn.functional as F



# Pick Device to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
"""
	Variation AutoEncoder Class
	inputs:
		- input_dim := The size of the flatten image vector
		- e_hidden1 := The size of the hidden dim of the 1st hidden Encoder Layer Compress the image
		- latent_dim :=  The size of the "code" image that is produced by Autoencoder
		- bernoulli_input :=  Noisy image input
"""
class VAE(nn.Module):
	def __init__(self,input_dim = 784, e_hidden1=300,latent_dim-20,bernoulli_input):
		super(VAE,self).__init__()
		self.encoder_inputHidden2 = nn.Linear(in_dim=input_dim,out_features=e_hidden1)
		self.encoder_hidden2mean =  nn.Linear(in_features=e_hidden1, out_features=latent_dim)
		self.encoder_hidden2logvar = nn.Linear(in_features=e_hidden1,out_features=latent_dim)

	def encoder(self,X):
		# Flatten input image to pass it to the encoder (batch_size, input_features)
		x = X.view(-1,784)
		# Feet x into Encoder to obtain mean and logvar
		x = F.relu(self.encoder_inputHidden2(x))
		return self.encoder_input2mean, self