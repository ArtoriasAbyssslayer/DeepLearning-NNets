import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import vi_utils
import loss 

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAE_cnn(nn.Module):
    def __init__(self,latent_size,bernoulli):
        super(VAE_cnn,self).__init__()
        self.latent_size = latent_size
        self.bernoulli = bernoulli
        if bernoulli:
            print("Constructing VAE basic model with bernoulli masked images")
        else:  
            print("Reconstructon ")


        # Convolutional Part 
        # encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, [2,2], stride=2, padding=1), # [8, 14, 14]
            nn.ReLU(),
            nn.Conv2d(8, 16, [5,5], stride=2, padding=1), # [16, 7, 7]
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Conv2d(16, 32, [3,3], stride=2, padding=0),# [32, 3, 3]
            nn.ReLU(),
            nn.Conv2d(32, 64, [3,3], stride=2, padding=2), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3,3)),# [64, 1, 1]
            
        )


        self.flatten = nn.Flatten()


        # VI formulation supplementary layers
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logs2 = nn.Linear(64, latent_size)
        # Linear Section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Decoder
        # linear part
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # unflatten part
        self.unflatten = nn.Sequential(
            # nn.Linear(latent_size, 64),
            nn.Unflatten(dim=1, unflattened_size=(64, 1, 1))
        )
        

        #Conv Part
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3,3)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, [3,3], stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, [5,5], stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, [2,2], stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.init_weights()


    def init_weights(self):
        # Initialize the weights of the model
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    

    def encode(self,X):
        enc = self.encoder_cnn(X)
        x_flat = self.flatten(enc)
        h = self.encoder_lin(x_flat)
        mu = self.fc_mu(h)
        logs2 = self.fc_logs2(h)
        z = self.reparameterize(mu, logs2)
        return mu,logs2


    def decode(self,z):
        x_hat = self.decoder_lin(z)
        x_hat_unflat = self.unflatten(x_hat)
        decoded = self.decoder_cnn(x_hat_unflat)
        return decoded

    def forward(self,X):
        # encode 
        enc = self.encoder_cnn(X)
        x_flat = self.flatten(enc)
        h = self.encoder_lin(x_flat)
        mu = self.fc_mu(h)
        logs2 = self.fc_logs2(h)
        # decode 
        z = self.reparameterize(mu, logs2)
        x_hat = self.decoder_lin(z)
        x_hat_unflat = self.unflatten(x_hat)
        decoded = self.decoder_cnn(x_hat_unflat)
        return decoded,mu,logs2


    def reparameterize(self, mu, logvar):
        z = vi_utils.reparametrization_trick(mu=mu,log_var=logvar)
        return z


    def generate(self,mu,log_var):
        with torch.no_grad():
            z = vi_utils.reparametrization_trick(mu, log_var)
            dec = self.decode(z)
            return self.tensor_to_numpyImg(dec)


    def generate_random_sample(self, n_images):
        #    Method that generates random samples from the latent space
        #    :return: a sample starting from z ~ N(0,1) converted to img 
        with torch.no_grad():
                
            z = torch.randn((n_images, self.latent_size), dtype = torch.float,device=device)
            samples =  self.decode(z)
            return self.tensor_to_numpyImg(samples)
    
    def generate_next_sample(self,n_images):
        with torch.no_grad():
            z = torch.randn((n_images+1),self.latent_size, dtype = torch.float,device=device)
            samples =  self.decode(z)
            return self.tensor_to_numpyImg(samples)

    def tensor_to_numpyImg(self,tensor):
        bin = self.bernoulli
        if bin:
            recon_bin = (tensor > 0.5).to(tensor.dtype).reshape(tensor.size(0),28,28).cpu().detach().numpy()
            return recon_bin
        else:
            recon_gas = tensor.reshape(tensor.size(0),28,28).cpu().detach().numpy()
            return recon_gas

    def loss_function(self,recon_x,x,mu,logvar):
        if self.bernoulli:
            return loss.recon_kld(recon_x,x,mu,logvar,input_type='bernoulli_input')
        else:
            return loss.recon_kld(recon_x, x, mu, logvar,input_type='gaussian_blurred_input')
    

if __name__ == "__main__":
    # NOTE THIS IS JUST A TEST CODE
    # NOTE CNN NEED 4D INPUT 
    x = torch.randn(1,1,28,28)
    model = VAE_cnn(input_dim=(1,28,28,1),latent_size=20,bernoulli=True)
    recon,mu,logvar = model(x)
    print(recon.shape)
    print(mu.shape)
    print(logvar.shape)