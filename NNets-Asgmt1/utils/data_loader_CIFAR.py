import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch
import numpy as np
'''
    This script is preparing dataset in order to use it on an NNet
    (split dataset in test and training set and shuffle it to create
    entropy result to a more robust and accurate model)
    
    Data iterator is an object in pytorch that preprocess the dataset,
    suffles it and operates as an iterator to begin the training process
    imediatelly after loading dataset.
'''


def load_cifar10_iterators():
       
        # The output of torchvision datasets are PILImage images of range[0,1]. 
        # below is a transformer that torchvision provides and in general form is this:

        mean, std = (0.5,), (0.5,)
        transformation = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                              ])
        # transformation = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # Batch_size is defined to set the number of data that DataLoader brings in each iteration
        batch_size = 256
        # torchvision has its own functions to load data and torch has random split inorder to get a validation set.
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformation)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transformation)
        val_size = 1000
        train_size = len(train_set) - val_size
        torch.manual_seed(32)
        train_ds, val_ds = random_split(train_set, [train_size, val_size])
        # load dataset with parrallel use of cpu and mix up images to create noise
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size*2, num_workers=4, pin_memory=True)
        return train_loader, test_loader, val_loader
def imshow(img):
    img = img/2 + 0.5  # image unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
