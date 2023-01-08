import imageio
import os
import numpy as np
import cv2 as cv
def outputs_to_gif(filenames):
     filenames=os.listdir()
     images = []
     for filename in filenames:
          images.append(imageio.imread(filename))
     imageio.mimsave('movie.gif', images)
def numpy_to_gif(array):
     images = []
     for i in range(array.shape[0]):
          images.append(array[i])
     imageio.mimsave('movie.gif', images)

def plot_image(img):
     fig = plt.figure()
     plt.subplots_adjust(wspace=0, hspace=0)
     cv.imshow('image',img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def plot_pca_components(pca):
     fig = plt.figure()
     fig.subtitle('MNIST PRINCIPAL COMPONENTS')
     feature_dim = int(np.ceil(np.sqrt(pca.n_components_)))
     i = 1
     for c in pca.components_:
          img = 255*np.reshape(c,(28,28))
          subplot = fig.add_subplot(feature_dim,feature_dim,i)
          ax.get_yaxis().set_visible(False)
          ax.get_xaxis().set_visible(False)
          subplot.imshow(img)
          i=i+1
     
def plot_loss_curves(kld_losses,recon_losses,train_losses,num_epochs,type,isTrain):
     fig = plt.figure()
     if isTrain: 
          fig.subtitle('TRAINING LOSSES')
     else:
          fig.subtitle('TESTING LOSSES')
     if type == 'Autoencoder' or type == 'PCA':
          ax2.set_title('Accumulated Loss')
          ax2.set_xlabel('Epochs')
          ax2.set_ylabel('Sum Loss')
          plt.plot(np.arange(1,num_epochs+1),train_losses)
          plt.show()
     else:
          ax = plt.subplot(1,figsize(10,10))
          ax.set_title('Kullback–Leibler divergence Loss')
          ax.set_xlabel('Epochs')
          ax.set_ylabel('KLD')
          plt.plot(np.arange(1,num_epochs+1),kld_losses)
          ax2 = plt.subplot(2,figsize(10,10))
          ax2.set_title('Reconstruction Loss')
          ax2.set_xlabel('Epochs')
          ax2.set_ylabel('Recon')
          plt.plot(np.arange(1,num_epochs+1),recon_losses)
          ax3 = plt.subplot(3,figsize(10,10))
          ax2 = plt.subplot(2,figsize(10,10))
          ax2.set_title('Accumulated Loss')
          ax2.set_xlabel('Epochs')
          ax2.set_ylabel('Sum Loss')
          plt.plot(np.arange(1,num_epochs+1),train_losses)
          plt.show()

def plot_ae_outputs(encoder,decoder,test_dataset,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()   



#TODO CREATE THE FUNC FOR BELLOW TASK -- Left for future work
"""
    Visualize the latent space with t-SNE
"""
# import tqdm
# import torch
# encoded_samples = []
# for sample in tqdm(test_dataset):
#     img = sample[0].unsqueeze(0).to(device)
#     label = sample[1]

    

# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2)
# tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
# fig = px.scatter(tsne_results, x=0, y=1,
#                  color=encoded_samples.label.astype(str),
#                  labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
# fig.show()