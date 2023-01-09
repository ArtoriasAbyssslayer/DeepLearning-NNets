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


''' Select running device '''
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Load checkpoint netowork
    checkpoint = torch.load(args.model_name)
    config = {
        'batch_size':args.batch_size,
        'data_masking': args.data_masking,
        'save_name':args.save_name,
        'model_name':args.model_name,
        'latent_size':args.latent_size,
        'one_hot':args.one_hot_labels,
        'hidden_size':args.hidden_size
    }
    if os.path.exists(checkpoint):
        model = net_builder(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        # Load data

        train_Loader,test_Loader = utils.load_mnist(config.get('batch_size'),masking=config['data_masking'],one_hot_labels=['one_hot'],workers=4)
        generated_imgs = np.zeros((args.sample))
    else :
        print("No pretrained model checkpoint found")
        return
    
    trainDataLoader,testDataLoader = utils.load_mnist(config.get('batch_size'),masking=config['data_masking'],one_hot_labels=['one_hot'],workers=4)
    
    generated_imgs = np.zeros((args.samples_num+1),args.batch_size,28,28)
    for data,_ in tqdm(test_DataLoader):
        data = data.to(device)
        generated_imgs[0] = data.reshape((args.batch_size,28,28)).cpu().detach().numpy()
        mu,logvar = model.encode(data)
        # Generate samples of dataset
        for i in tqdm(range(args.samples_num)):
            generated_images[i+1]=model.generate(mu,logvar)
        break

    random_image = model.generate_random_sample(args.samples_num)
    random_image_2 = model.generate_next_sample(args.samples_num)
    generated_samples = [random_image,random_image_2]
    visualization_utils.numpy_to_gif(generated_samples)
    generated_imgs = np.concatenate((generated_imgs,random_image,random_image_2),axis=0)
    # Save generated images

    visualization_utils.plot_ae_outputs(model.encode, model.decode, test_Loader.dataset.targets, config['batch_size'])
    


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model_name","-m",default="VAE.ptr")
    parser.add_argument("--latent_size","-L",default=85,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument("--batch_size","-b")
    parser.add_argument("--samples_num","-s",default=10,type=int)
    args = parser.parse_args()
    main(args)
    
   