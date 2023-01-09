import sys
import os 
sys.path.append(os.getcwd())
import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import stats
import torch
from sklearn.decomposition import PCA
import utils
import visualization_utils
import time 

# Get the DataLoaders and load the images 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main(args):
    trainLoader,testLoader =  utils.load_mnist(batch_size=args.batch_size,masking='gaussian',workers=4)
    #
    n_components = args.n_components
    plt.figure(figsize=(5,len(n_components)))
    plt.title("PCA autoencoded images with different number of principal Components - Train")
    

    for batch,(traindata,_) in enumerate(trainLoader):
        traindata = data.to(DEVICE)
        for i,components in enumearte(n_components):
            # Start timer
            start_time = time.time()

            # Initialize PCA model
            pca = PCA(n_components=components)

            # Fit model to traindata

            pca = pca.fit(traindata)
            
            # Encode using pca 
            pca_tranformed = pca.transform(traindata)
            # Decode using pca
            pca_inverse = pca.inverse_transform(pca_transformed)
            # End timer
            end_time = time.time()
            # Print time taken
            print("Time taken for PCA with {} components: {}".format(components,end_time-start_time))
            # Plot original and reconstructed images
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(traindata[0].reshape(28,28),cmap='gray')
            # or use function from visualiztion utils to plot reconstructed
            fig,ax = plt.subplots(2,2)
            visualization_utils.plot_pca_components(pca_inverse)

            plt.show()
    # do the same for the test set
    plt.title("PCA autoencoded images with different number of principal Components - Train")

    for batch,(testdata,_) in enumerate(testLoader):
        testdata = testdata.to(DEVICE)
        # Start timer
        start_time = time.time()
        # Encode using pca 
        pca_tranformed = pca.transform(testdata)
        # Decode using pca
        pca_inverse = pca.inverse_transform(pca_transformed)
        # End timer
        end_time = time.time()
        # Print time taken
        print("Time taken for PCA with {} components: {}".format(components,end_time-start_time))
        # Plot original and reconstructed images
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(testdata[0].reshape(28,28),cmap='gray')
        # or use function from visualiztion utils to plot reconstructed
        fig,ax = plt.subplots(2,2)
        visualization_utils.plot_pca_components(pca_inverse)
    '''save pca model'''
    utils.save_pca_model(pca, 'PCA')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_components', type=int, default=[100,0.9,0.95], help='latent dimension')
    args = parser.parse_args()
    main(args)

