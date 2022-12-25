import numpy as np

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
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']