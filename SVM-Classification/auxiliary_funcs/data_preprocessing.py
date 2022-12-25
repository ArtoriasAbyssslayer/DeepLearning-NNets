import os
import tarfile 
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The features of the dataset.
    y : array-like, shape (n_samples,)
        The labels of the dataset.
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test set.
    random_state : int, optional (default=42)
        The seed for the random number generator.

    Returns
    -------
    X_train : array-like, shape (n_samples_train, n_features)
        The features of the training set.
    X_test : array-like, shape (n_samples_test, n_features)
        The features of the test set.
    y_train : array-like, shape (n_samples_train,)
        The labels of the training set.
    y_test : array-like, shape (n_samples_test,)
        The labels of the test set.
    """
    # Set the random seed
    np.random.seed(random_state)

    # Generate a boolean mask for the test set
    mask = np.random.rand(len(X)) < (1 - test_size)

    # Split the dataset into training and test sets
    X_train = X[mask]
    X_test = X[~mask]
    y_train = y[mask]
    y_test = y[~mask]

    return X_train, X_test, y_train, y_test

def load_label_names():
    """
        Returns the array with classes names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
def _onehot(integer_labels):
        """
            Returns matrix whose rows are onehot encodings of integers.
        """
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot


def cifar10_load(data_path=None):
# - Return:
#     (train_images, train_labels, test_images, test_labels).
# - Args:
#     data_path (str): Directory containing CIFAR-10. Default is
#     {workspacefolder}/SVMClassification/Dataset/data/cifar10 or C:\Users\USER\data\cifar10.
#     Create if nonexistant. Download CIFAR-10 if missing.
    url  = 'https://www.cs.toronto.edu/~kriz/'
    tar  = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    if data_path is None:
        # Set path to Dataset path
        data_path = os.path.join("/home/harry/Documents/DeepLearning-NNets/Dataset",'data','cifar10')
    # Create path if it doesn't exist
    os.makedirs(data_path,exist_ok=True)
    # Download tarfile if missing
    if tar not in os.listdir(data_path):
        urlretrieve(''.join((url, tar)), os.path.join(data_path, tar))
        print("Downloaded %s to %s" % (tar, data_path))
    # Load data from tarfile 
    with tarfile.open(os.path.join(data_path,tar)) as tar_object:
        # Each file contains 10.000 RGB images and 10.000 labels
        file_size =  10000 * (32*32*3) + 10000

        # There are 6 files (5 train and 1 test) split in batches of 10.000 each.
        # Test-Train split is done in the way we have  0.8 train 0.2 test approximately

        buffer = np.zeros(file_size*6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other irrelevant stuff

        members = [file for file in tar_object if file.name in files]
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            file = tar_object.extractfile(member)
            # Read bytes from that file object into buffer array
            buffer[i * file_size:(i + 1) * file_size] = np.frombuffer(file.read(), 'B')
    # Labels are the first byte of every chunk of data
    labels = buffer[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffer,np.arange(0,buffer.size,3073))
    images = pixels.reshape(-1,3072).astype('float32')
    # Normalization will take place in the next function
    train_images,test_images,train_labels,test_labels = train_test_split(images, labels)
    return (train_images, train_labels), (test_images, test_labels)
    

def data_preproc(data_path,scale=True,pca=False,num_components=0.95):
    # Load the CIFAR-10 data   splitted in above method
    (X_train,y_train),(X_test,y_test) = cifar10_load(data_path)
    
    # Normalize and flatten the data  
    # Using sklearn.preprocessing.StandardScaler in order to normalize the dataset based on 
    # its mean and standard deviaion values for each pixel so I have an equal distribution and avoid 
    # outliers that are made through image capture progress 
    if scale:
        train_images = StandardScaler().fit_transform(X_train)
        test_images = StandardScaler().fit_transform(X_test)

    if pca:
        clf = PCA(n_components=num_components)
        train_features = clf.fit_transform(train_images)
        test_features = clf.fit_transform(test_images)
    
    # one_hot encode train labels and test labels
    y_trainEnc = _onehot(y_train)
    y_testEnc = _onehot(y_test)
    # Import the data into a pandas dataframe and return it to the user
    df_train_images = pd.DataFrame(train_images)
    df_test_images = pd.DataFrame(test_images)
    df_train_labels = pd.DataFrame(data=y_trainEnc,columns=load_label_names())
    df_test_labels = pd.DataFrame(data=y_testEnc , columns=load_label_names())
    y_train_df = pd.DataFrame({"label": y_train})
    return (df_train_images, df_train_labels),(df_test_images,df_test_labels), y_train_df

def reduce_dataset_size(n_samples,labels,reduce_factor=10):
    '''
        Reduce the size of the dataset according to reduce_factor percent
        If reduce_factor is 10 and we have 50000 samples then we will have
        40000 samples with the same labels.
:

    '''
    reduction_rate = reduce_factor // 100
    reduced_size = labels.shape[0] * reduction_rate

    # Value that checks if the new dataset is balanced

    even_labels_percenatge = 1.0

    while even_labels_percenatge < 0.4 or even_labels_percenatge > 0.6:
        # Generate new indicies
        indices = np.random.randint(0,labels.shape[0], reduced_size)
        # new_labels = argmax(labels.loc[indices],1)
        new_labels = labels.loc[indices]
        # Sum the values with label 1 and devide by the total number of labels
        even_labels_percenatge =  new_labels.value_counts()[1] / reduced_size
    new_sampless = samples.loc[indices,:]

    return new_samples, new_labels
    