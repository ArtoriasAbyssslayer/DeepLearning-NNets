import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Variational Inference Utils made according to the 
import vi_utils

# Pick Device to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

	Variational AutoEncoder Class

	Variational Inference Formulation given by MLP Autoencoder.
	In case of a Gaussian MLP, we will have to output log_var and mean.
	In case of a Bernoulli MLP, we will have to output log_var and mean.

	inputs:
		- input_dim := The size of the flatten image vector
		- e_hidden_dim := The size of the hidden dim of the 1st hidden Encoder Layer Compress the image
		- latent_dim :=  The size of the "code" image that is produced by Autoencoder
		- bernoulli_input :=  Bin masked input 
		- gaussian_blurred_input := Noisy input with gaussian Noise 
	outputs:
		- log_var := log(sigma^2)
		- mu := mean of the reparametrized 
		- dec := decoded z distribution
"""

# Main Variational AE - Variational Inference Formulation Pipeline for Autoencoder
# Input image -> Hidden Dim -> mean,std ->  Parametrization Trick -> Decoder -> Output Image
class VAE(nn.Module):
	def __init__(self,input_dim = 784,e_hidden_dim=400,latent_dim=20,bernoulli_input=None,gaussian_blurred_input=None):
		# Initialize Model and parameters
		super(VAE,self).__init__()
		self.input_dim = input_dim
		self.e_hidden_dim = e_hidden_dim
		self.latent_dim = latent_dim
		self.bernoulli_input = bernoulli_input
		self.gaussian_blurred_input = gaussian_blurred_input
	
		
		# Create the architecture
		neurons = [input_dim, *e_hidden_dim]

		
		# Encoder Layers Definition
		self.encoder_inputHidden2 = nn.Linear(in_dim=input_dim,out_features=e_hidden_dim)
		self.encoder_hidden2mu =  nn.Linear(in_features=e_hidden_dim, out_features=latent_dim)
		self.encoder_hidden2logvar = nn.Linear(in_features=e_hidden_dim,out_features=latent_dim)

		# Decoder Layer Definition
		self.decoder_z2hidden = nn.Linear(latent_dim,e_hidden_dim)
		self.decoder_hidden2img = nn.Linear(e_hidden_dim,input_dim)
		self.kl_divergence = 0
		# Initialize the weights of the model 
		self.init_weights()

	def init_weights(self):
		# Initialize the weights of the model
		for m in self.modules():
			if isinstance(m,nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias,0)

	def reparametrize(self,mu,logvar):
		# Reparametrization Trick using vi_utils
		 return vi_utils.reparametrization_trick(mu,logvar)

	def forward(self,X):
		z,z_mu,z_log_var = self.encode(self,X)
		kl_divergence

	# Encoder process 
	def encode(self,X):
		# Flatten input image to pass it to the encoder (batch_size, input_features)
		x = X.view(-1,self.input_dim)
		# Fit x into Encoder to obtain compressed imgae
		h = self.encoder_inputHidden2(x)
		# Fit x into Encoder layer for mu to obtain mean 
		mu = self.encoder_hidden2mu(h)
		# Fit x into 3rd layer to obtain logvar
		logvar =self.encoder_hidden2logvar(h)
		return mu,logvar
	
	# Decoder process 
	def decode(self,z):
		# Fit x into Encoder to obtain mean and logvar
		z = F.relu(self.decoder_z2hidden(x))
		# Add sigmoid activation to obtain the image (sigmoid function add the threshold we need)
		return torch.sigmoid(self.decoder_hidden2img(x))
	
	def tensor_to_numpyImg(self,img):
		bin = self.bernoulli_input
		gas = self.gaussian_blurred_input
		if gaussian_blurred_input:
			return img.reshape(img.size(0), 28, 28).cpu().detach().numpy()
		elif bin:
			return (img > 0.5).to(img.dtype).reshape(img.size(0), 28, 28).cpu().detach().numpy()
		else:
			return img.reshape(img.size(0), 28, 28).cpu().detach().numpy()

	def generate(self,mu,log_var):
		with torch.no_grad():
			z = vi_utils.reparametrization_trick(mu, log_var)
			dec = self.decode(z)
			return self.tensor_to_numpyImg(dec)
	# sample is the function that generates random samples of images -- generator
	def sample(self, n_images):
        '''
        Method to sample from our generative model
        :return: a sample starting from z ~ N(0,1)
        '''

        z = torch.randn((n_images, self.z_dims), dtype = torch.float)
        # print(z)
        samples =  self.decoder(z)

        return samples



