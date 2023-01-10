import sys
import os 
sys.path.append(os.getcwd())
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import torch
import utils
import visualization_utils
from sklearn.decomposition import PCA

import time 
from sklearn.metrics import mean_squared_error

# Get the DataLoaders and load the images 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main(args):
    trainLoader,testLoader =  utils.load_mnist(batch_size=args.batch_size,masking='bin_masking',workers=4)
    #
    n_components = args.n_components
    plt.figure(figsize=(5,len(n_components)))
    plt.title("PCA autoencoded images with different number of principal Components - Train")
    traindata =trainLoader.dataset.data
    testdata  =trainLoader.dataset.data
    print(traindata.shape)
    print(testdata.shape)
    plt.figure(figsize=(5, len(n_components)))
    plt.title("Autoencoded images for different number of components")
    images = traindata[:5, :].reshape(-1, 28, 28)
    for j in range(5):
        plt.subplot(len(n_components) + 1, 5, j + 1)
        plt.imshow(images[j, :, :], cmap="gray")
        plt.axis("off")
    for i,components in enumerate(n_components):
        # Start timer
        traindata = traindata.view(-1,784)
        testdata = testdata.view(-1,784)
        start_time = time.time()

        # Initialize PCA model
        pca = PCA(n_components=components)

        # Fit model to traindata
        pca_fitted = pca.fit(traindata)
        # AE with pca
        # Encode
        pca_transformed_train = pca.transform(traindata)
        # Decode
        pca_inverse = pca.inverse_transform(pca_transformed_train)
        # End timer
        end_time = time.time()
        pca_testdata = pca.fit(testdata)
        pca_transformed_test = pca.transform(testdata)
        pca_inverse_test = pca.inverse_transform(pca_transformed_test)
        # Print time taken
        print("Time taken for PCA with {} components: {}".format(components,end_time-start_time))
    
        # Print Errors 
        train_rmse = mean_squared_error(traindata, pca_inverse, squared=False)
        test_rmse = mean_squared_error(testdata, pca_inverse_test, squared=False)
        print(f"For {pca.n_components_} components")
        print(f"The RMSE of the train set is: {train_rmse}")
        print(f"The RMSE of the test set is: {test_rmse}")
        print("Time passed: %s seconds." % (time.time() - start_time))
        reconstructed = pca_inverse[:5,:].reshape(-1,28,28)
        for j in range(5):
            plt.subplot(len(n_components) + 1, 5, (i + 1) * 5 + j + 1)
            plt.imshow(reconstructed[j, :, :], cmap="gray")
            plt.axis("off")
        plt.show()
    utils.save_pca_model(pca, 'PCA')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size','-b',type=int, default=64)
    parser.add_argument('--n_components', type=int, default=[0.8,0.9,0.95], help='latent dimension')
    args = parser.parse_args()
    main(args)

