import torch 
import numpy as np
def log_standard_gaussian(x):
    """
    Computes the log pdf of a standard Gaussian distribution at each data point in x. Also
    know as (Univariate Distribution)
    
    Args:
        x: torch.Tensor: point at which to evaluate the log pdf of the standard Gaussian
        
    Returns:
        log_prob: torch.Tensor, shape (batch_size, )  
    """
    return torch.sum(-0.5 * torch.log(2 * np.pi) - x ** 2 / 2, dim=-1)

def log_gaussian(x, mu, log_var):
    """
    Computes the log pdf of a Gaussian distribution at each data point in x.
    *Parameterized*

    Args:
        x: torch.Tensor: point at which to evaluate the log pdf of the Gaussian
        mu: torch.Tensor: mean of the Gaussian Distribution
        log_var: torch.Tensor: log variance of the Gaussian Distribution
        
    Returns:
        log_prob: torch.Tensor, shape (batch_size, )  
    """
    # return torch.sum(log_pdf, dim=-1)
    log_pdf = torch.sum(-0.5 * torch.log(2 * np.pi) - log_var / 2 - (x-mu)**2 / (2 * torch.exp(log_var)), dim=-1)
    return log_pdf



def reparametrization_trick(mu,log_var):
    '''
        Function that given the mean(mu) and logarithmic variance(log_var)
        computes the latent variables using the reparametrization trick.
            z = mu + sigma * noise, where noise is sample
    
        :param mu: mean of the z_variables
        :param log_var: variance of the latent variables
        :return: z = mu + sigma * noise
    '''
    # Get var/2 using logarithm property
    std = torch.expo(log_var*0.5)

    # Sample the noise(we dont have to keep gradients with respect to the noise)

    eps = Variable(torch.randn_like(std), requires_grad = False)
    z = mu*addcmul(std,eps)
    return z


