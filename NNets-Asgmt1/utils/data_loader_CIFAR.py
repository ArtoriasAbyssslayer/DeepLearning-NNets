from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transforms
'''
    This script is preparing dataset in order to use it on an NNet
    (split dataset in test and training set and shuffle it to create
    entropy result to a more robust and accurate model)
    
    Data iterator is an object in pytorch that preprocess the dataset,
    suffles it and operates as an iterator to begin the training process
    imediatelly after loading dataset.
'''


def load_cifar10_iterators():
       
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
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
        # load dataset with parrallel use of cpu and mix up images to create noise
        testloader = DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=4)


        return trainloader,testloader

import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img/2 + 0.5  # image unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
def plot_random_images(trainloader):
    dataiter= iter(trainloader)
    images,labels = next(dataiter)
    

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


## Another versions of loading CIFAR10 without pytorch this uses already downloaded Dataset in ../datasets/cifar-10-python/cifar-10-batches.py where cifar is saved as batches
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y
def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = '../input/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test
