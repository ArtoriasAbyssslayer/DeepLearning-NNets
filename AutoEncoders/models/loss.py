"""
    This function is dedicated to calculate the loss of the vae model.
    Based on Variational Inference Formulation practice
    The selected loss is a combination of sum of Reconstruction Loss and KLD loss 
    KLD loss refers to KullBack-Leibler Divergence between the latent distribution and the prior distribution of generated images
"""

import torch 
import torch.nn.functional as F 


# Reconstruction + KL divergence losses summed over all elements and batch
def recon_kld(recon_x, x, mu, logvar, input_type='None'):
    # Reconstruction Loss 
    if input_type == 'bernoulli_input':
        # BCE loss
        # recon_loss = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')
        if(recon_x.shape == (32,1,28,28)):
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x)
        else:
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784))
        
    elif input_type == 'gaussian_blurred_input':
        # Gaussian Reconstruction Loss 
        # recon_loss = F.mse_loss(recon_x,x.view(-1, 784),reduction='sum')
        if(recon_x.shape == (32,1,28,28)):
            recon_loss = F.mse_loss(recon_x,x)
        else:
            recon_loss = F.mse_loss(recon_x,x.view(-1, 784))  # without reduction to let gradients handle the loss 
    else:
        recon_loss = 0
        raise ValueError('Input type not supported')
    
    # KL Divergence 
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    sum_loss = recon_loss + KLD
    return recon_loss, KLD, sum_loss

