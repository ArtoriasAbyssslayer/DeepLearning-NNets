import imageio
import os
from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
def outputs_to_gif(filenames):
     if filenames==os.listdir():
          images = []
          for filename in filenames:
               images.append(imageio.imread(filename))
          imageio.mimsave('movie.gif', images)
     else:
          imageio.mimsave('movie.gif',filenames)

def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip
def make_gif(imgs):
     # frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*")] 
     # frames = np.zeros((imgs.shape[0]*imgs.shape[1]+imgs.shape[1],(28,28)))
     for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
               img = imgs[i,j]
               idx = i*imgs.shape[1]+j 
               frames = Image.fromarray((img* 255).astype(np.uint8))
               frame_one = frames[0]
               frame_one.save("gen.gif", format="GIF", append_images=frames,save_all=True, duration=1000, loop=1)

def plot_image(imgs):
     fig = plt.figure()
     plt.subplots_adjust(wspace=0, hspace=0)
     for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            idx = i*imgs.shape[1]+j
            ax = fig.add_subplot(imgs.shape[0], imgs.shape[1], idx+1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_axis_off()
            ax.imshow(255*img, cmap='gray_r')
     plt.show()


def plot_pca_components(pca):
     fig = plt.figure()
     fig.suptitle('MNIST PRINCIPAL COMPONENTS', fontsize=16)
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
     if type == 'Autoencoder' or type == 'PCA':
          fig,ax1 = plt.subplots(1)
          if isTrain: 
               fig.suptitle('TRAINING LOSSES', fontsize=16)
          else:
               fig.suptitle('TESTING LOSSES', fontsize=16)
          
          ax1.set_title('Accumulated Loss')
          ax1.set_xlabel('Epochs')
          ax1.set_ylabel('Sum Loss')
          ax1.plot(np.arange(1,num_epochs+1),train_losses)
          plt.show()
     else:
          fig,(ax1,ax2,ax3) = plt.subplots(1,3)
          if isTrain: 
               fig.suptitle('TRAINING LOSSES', fontsize=16)
          else:
               fig.suptitle('TESTING LOSSES', fontsize=16)
          ax1.set_title('Kullback–Leibler divergence Loss')
          ax1.set_xlabel('Epochs')
          ax1.set_ylabel('KLD')
          ax1.plot(np.arange(1,num_epochs+1),kld_losses)
          ax2.set_title('Reconstruction Loss')
          ax2.set_xlabel('Epochs')
          ax2.set_ylabel('Recon')
          ax2.plot(np.arange(1,num_epochs+1),recon_losses)
          ax3.set_title('Accumulated Loss')
          ax3.set_xlabel('Epochs')
          ax3.set_ylabel('Sum Loss')
          ax3.plot(np.arange(1,num_epochs+1),train_losses)
          plt.show()

def plot_ae_outputs(model,encoder,decoder,test_dataset,n=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plt.figure()
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      with torch.no_grad():
          rec_img,_,_  = model(img) # instead of running decode(encode(img)) run forward call
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      rec_img = rec_img.cpu()
      rec_img=np.reshape(rec_img,(1,28,28))
      plt.imshow(rec_img.squeeze().numpy(), cmap='gist_gray')  
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