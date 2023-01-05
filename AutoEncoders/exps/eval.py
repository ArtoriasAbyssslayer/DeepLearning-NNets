import argparse
import torch
import numpy as np 
import matplotlib.pyplot as plt

import sys 
sys.path.append('..')
# Import models
from models import AutoEncoder
from models import VAE
from models import PCA_Compression

import utils
import visualization_utils



''' Select running device '''
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Load checkpoint netowork
    checkpoint = torch.load(args.model_name)
    model = 
    model.load_state_dict(checkpoint['state_dict'])

    train_Loader,test_Loader = utils.load_mnist(batch_size = args.batch_size,
                                                masking = args.data_masking)
                                                    

    # TODO











if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model_name","-m",default="VAE.pt")
    parser.add_argument("--latent_size","-L",default=85,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument("--batch_size","-b")
    parser.add_argument("--ep")
   