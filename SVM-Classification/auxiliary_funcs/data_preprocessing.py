import tensorflow as tf
import numpy as np
def one_hot_encode(x):
    """
        argument:
        - x: a list of labels
        return:
        - one hot encoding matrix(number of labels,number of class)
    """
    encoded = np.zeros(len(x),10)
    for idx,val in enumerate(x):
        encoded[idx][val] = 1
    return encoded
def data_load_preproc():
    # Load the CIFAR-10 data    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Make the assertions needed
    assert X_train.shape == (50000, 32, 32, 3)
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    # Preprocess the data   
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    # one_hot_encode train_labels
    # now with tensorflow func
    X_train_enc = tf.one_hot(X_train)
    # now with the custom function
    Y_train_enc = one_hot_encode(y_train)
    return X_train,y_train,X_test,y_test