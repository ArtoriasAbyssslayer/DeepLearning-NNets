# TODO:  CLEAN UP CODE AND WRITE SMALLER COMMENTS IMPORT DETAILS IN assignment report

'''
    This script is preparing dataset in order to use it on an NNet
    (split dataset in test and training set and shuffle it to create
    entropy and noise)
'''
# Imported CIFAR10 dataset  
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import tv.transforms as transforms
# this is a python lib so the images that are imported are in PIL (Python Imaging Library)
import cifar10


def load_cifar10():
       
        # The output of torchvision datasets are PILImage images of range[0,1]. We transform them to Tensors of normalized
        # range [-1,1]
        # below is a transformer that torchvision provides and in general form is this:

        # TRam
        transform = transforms.Compose(
        [transform.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )


        batch_size = 4

        # torchvision has its own functions to load data and it has split the CIFAR10 dataset to
        trainset = tv.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
        # load dataset with parallel use of cpu and mix up images to create noise
        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
        # load dataset with parrallel use of cpu and mix up images to create noise
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        shuffle=False, num_workers=2)


        return trainset,testset
