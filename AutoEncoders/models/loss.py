"""
    This function is dedicated to calculate the loss of the vae model.
    Based on Variational Inference Formulation practice
    The selected loss is a combination of sum of Reconstruction Loss and KLD loss 
    KLD loss refers to KullBack-Leibler Divergence between the latent distribution and the prior distribution of generated images
"""