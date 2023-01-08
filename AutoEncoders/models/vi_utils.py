import torch 
import numpy as np
from torch.autograd import Variable
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
    std = torch.exp(log_var*0.5)

    # Sample the noise(we dont have to keep gradients with respect to the noise)

    epsilon = Variable(torch.randn_like(std), requires_grad = False)
    z = torch.addcmul(mu,std,epsilon)
    return z


# Functions bellow are not used but were written for reference purposes

def log_bernoulli(x, p):
    """
    Computes the log pdf of a Bernoulli distribution at each data point in x.
    *Parameterized*

    Args:
        x: torch
        p: torch
        
    Returns:
        log_prob: torch.Tensor, shape (batch_size, )  
    """
    # return torch. 
    return torch.sum(x * torch.log(p) + (1 - x) 
                       * torch.log(1 - p), dim=-1)

def _analytical_kl_bernoulli(p, q):
    """
    Computes the analytical KL divergence between two Bernoulli)
    """
    return torch.sum(p * torch.log(p / q), dim=-1)


def _kl_divergence(self, z, q_params, p_params = None):
    '''
    The function compute the KL divergence between the distribution q_phi(z|x) and the prior p_theta(z)
    of a sample z.
    KL(q_phi(z|x) || p_theta(z))  = -∫ q_phi(z|x) log [ p_theta(z) / q_phi(z|x) ]
                                    = -E[log p_theta(z) - log q_phi(z|x)]
    :param z: sample from the distribution q_phi(z|x)
    :param q_params: (mu, log_var) of the q_phi(z|x)
    :param p_params: (mu, log_var) of the p_theta(z)
    :return: the kl divergence KL(q_phi(z|x) || p_theta(z)) computed in z
    '''

    ## we have to compute the pdf of z wrt q_phi(z|x)
    (mu, log_var) = q_params
    qz = log_gaussian(z, mu, log_var)
    # print('size qz:', qz.shape)
    ## we should do the same with p
    if p_params is None:
        pz = log_standard_gaussian(z)
    else:
        (mu, log_var) = p_params
        pz = log_gaussian(z, mu, log_var)
        # print('size pz:', pz.shape)

    kl = qz - pz

    return kl

## in case we are using a gaussian prior and a gaussian approximation family
def _analytical_kl_gaussian(self, q_params):
    '''
    Way for computing the kl in an analytical way. This works for gaussian prior
    and gaussian density family for the approximated posterior.
    :param q_params: (mu, log_var) of the q_phi(z|x)
    :return: the kl value computed analytically
    '''

    (mu, log_var) = q_params
    # print(mu.shape)
    # print(log_var.shape)
    # prova = (log_var + 1 - mu**2 - log_var.exp())
    # print(prova.shape)
    # print(torch.sum(prova, 1).shape)
    # kl = 0.5 * torch.sum(log_var + 1 - mu**2 - log_var.exp(), 1)
    kl = 0.5 * torch.sum(log_var + 1 - mu.pow(2) - log_var.exp(), 1)

    return kl

