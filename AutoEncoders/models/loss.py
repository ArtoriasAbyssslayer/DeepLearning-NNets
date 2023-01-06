"""
    This function is dedicated to calculate the loss of the vae model.
    Based on Variational Inference Formulation practice
    The selected loss is a combination of sum of Reconstruction Loss and KLD loss 
    KLD loss refers to KullBack-Leibler Divergence between the latent distribution and the prior distribution of generated images
"""

import torch 
import torch.nn.functional as F 


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KL

