from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transforms
'''
    This script is preparing dataset in order to use it on an NNet
    (split dataset in test and training set and shuffle it to create
    entropy result to a more robust and accurate model)
'''
def load_cifar10():
       
        # The output of torchvision datasets are PILImage images of range[0,1]. We transform them to Tensors of normalized
        # range [-1,1]
        # below is a transformer that torchvision provides and in general form is this:

        # TRam
        transform = transforms.Compose([transform.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        batch_size = 4

        # torchvision has its own functions to load data and it has split the CIFAR10 dataset to
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # load dataset with parallel use of cpu and mix up images to create noise
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
        # load dataset with parrallel use of cpu and mix up images to create noise
        testloader = DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)


        return trainloader,testloader
