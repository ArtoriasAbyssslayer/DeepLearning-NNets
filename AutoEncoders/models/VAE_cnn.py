import torch 
import torch.nn as nn
import torch.nn.functional as F 


device = "cuda" if torch.cuda.is_available() else "cpu"

class VAECnn(nn.Module):
    def __init__(self,latent_size,input):
        super(VAEBasic,self).__init__()
        self.latent_size = latent_size
        self.input = input
        if input:
            print("Constructing VAE basic model with bern ")

        # Convolutional Part 

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear Section
        self.encoder_lin = nn.Sequential(
            n.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_size)
        )
        ### Decoder Part
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )


        def forward(self,x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            return x

        def eikonal_loss(self,dec)

        def 