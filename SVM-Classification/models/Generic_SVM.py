import numpy as np

class Generic_SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.W = None
        self.b = None

    def train(self, X, y):
        n_samples, n_features = X.shape

        # Initialize the weights and biases to zeros
        self.W = np.zeros(n_features)
        self.b = 0

        # Initialize the optimization parameters
        eta = 1
        epochs = 1000

        # Iterate over the number of epochs
        for epoch in range(1, epochs+1):
            # Iterate over all samples in the dataset
            for i in range(n_samples):
                # If the sample is misclassified, update the weights and biases
                if y[i] * (np.dot(self.W, X[i]) + self.b) < 1:
                    self.W = self.W + eta * (self.C * y[i] * X[i] - 2 * (1/epoch) * self.W)
                    self.b = self.b + eta * (self.C * y[i] - 2 * (1/epoch) * self.b)
                # If the sample is correctly classified, do not update the weights and biases
                else:
                    self.W = self.W - eta * (2 * (1/epoch) * self.W)
                    self.b = self.b - eta * (2 * (1/epoch) * self.b)

    def predict(self, X):
        y_pred = []
        # Iterate over all samples in the dataset
        for i in range(X.shape[0]):
            # Predict the label for the sample
            y = np.dot(self.W, X[i]) + self.b
            if y >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred
