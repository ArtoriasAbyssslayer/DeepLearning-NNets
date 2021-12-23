# TODO:  CLEAN UP CODE AND WRITE SMALLER COMMENTS IMPORT DETAILS IN assignment report

'''
    This script is preparing dataset in order to use it on an NNet
    (split dataset in test and training set and shuffle it to create
    entropy and noise)
'''
# *** Disclaimers ***
# Imported CIFAR10 dataset that way because i wanted first to create noise
# in order my initial data to not be biased and the learning to be as unbiased as possible
# that was based also on entropy's theory.
# Also the data load throught torchvision is faster and I have transforms library
# so I can compose the transform i need sus

from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import tv.transforms as transforms
# this is a python lib so the images that are imported are in PIL (Python Imaging Library)
import cifar10


# The output of torchvision datasets are PILImage images of range[0,1]. We transform them to Tensors of normalized
# range [-1,1]
# below is a transformer that torchvision provides and in general form is this:


# from torchvision import transforms as transforms
# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  #crops a 32x32 patch (if it is an image) with 4 padding in the crop algorithm
#     transforms.RandomHorizontalFlip(),  #makes an horizontal flip with a kernel
#     transforms.RandomRotation((-45,45)), #makes a rotation to the image or signal
#     transforms.ToTensor(), #makes the array that the image is stored a Tensor
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B normalization
# ])

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


# class definition

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'dog', 'frog', 'horse', 'ship', 'truck')
