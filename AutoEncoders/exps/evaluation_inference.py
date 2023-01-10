import argparse
import torch
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os 
sys.path.append(os.getcwd())
# Import model
from models import Autoencoder,VAE,VAE_cnn,PCA_denoise
# Import utils')
import utils
import visualization_utils
from train import net_builder
from tqdm import tqdm

''' Select running device '''
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Load checkpoint netowork
    checkpoint = torch.load(args.model_name)
    config = {
        'batch_size':args.batch_size,
        'data_masking': args.data_masking,
        'model_name':args.model_name,
        'latent_size':args.latent_size,
        'samples_num':args.samples_num
    }
    
    if os.path.exists(args.model_name):

        model = checkpoint['model']
        # model.load_state_dict(checkpoint['state'],strict=False)
        model.to(device)
        model.eval()
        # Load data
    else :
        print("No pretrained model checkpoint found")
        return
    
    # Load Dataset and Generate selected sample of MNIST Images with model function call 
    _,testDataLoader = utils.load_mnist(batch_size=config['batch_size'],masking=config['data_masking'],workers=4)
    generated_imgs = np.zeros(((config['samples_num']+1),args.batch_size,28,28),dtype=float)
    for data,_ in testDataLoader:
        data = data.to(device)
        data_reshaped = data.reshape((config['batch_size'],28,28)).cpu().detach().numpy()
        generated_imgs[0] = data_reshaped
        mu,logvar = model.encode(data)
        # Generate samples of dataset
        for i in tqdm(range(args.samples_num)):
            generated_imgs[i+1]=model.generate(mu,logvar)
        break


    #random_image = model.generate_random_sample(args.samples_num)
    #random_image_2 = model.generate_next_sample(args.samples_num)
    #generated_samples = [random_image,random_image_2]

    # visualization_utils.make_gif(generated_samples)
    #generated_imgs = np.concatenate((generated_imgs,random_image,random_image_2),axis=2)
    # print(generated_imgs.shape)
    # Save generated images
    visualization_utils.plot_ae_outputs(model,model.encode, model.decode, testDataLoader.dataset, config['samples_num'])
    visualization_utils.plot_image(generated_imgs)
    visualization_utils.gif(config['model_name']+"generated_imgs.gif",generated_imgs)


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model_name","-m",default="./saved_models/checkpoint_VAE.ptr")
    parser.add_argument("--latent_size","-L",default=85,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument("--batch_size","-b",default=32,type=int)
    parser.add_argument("--samples_num","-s",default=10,type=int)
    args = parser.parse_args()
    main(args)
    
   