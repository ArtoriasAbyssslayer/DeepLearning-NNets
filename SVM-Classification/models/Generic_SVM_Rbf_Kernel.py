import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.W = None
        self.b = None

    def train(self, X, y, learning_rate=1.0, batch_size=128, epochs=1000):
        n_samples, n_features = X.shape

        # Initialize the weights and biases to zeros
        self.W = np.zeros(n_features)
        self.b = 0

        # Calculate the number of batches
        n_batches = n_samples // batch_size

        # Iterate over the number of epochs
        for epoch in range(1, epochs+1):
            # Shuffle the data at the start of each epoch
            shuffle_index = np.random.permutation(n_samples)
            X, y = X[shuffle_index], y[shuffle_index]

            # Iterate over all batches
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]

                # Iterate over all samples in the batch
                for j in range(batch_size):
                    # If the sample is misclassified, update the weights and biases
                    if y_batch[j] * (np.dot(self.W, X_batch[j]) + self.b) < 1:
                        self.W = self.W + learning_rate * (self.C * y_batch[j] * X_batch[j] - 2 * (1/epoch) * self.W)
                        self.b = self.b + learning_rate * (self.C * y_batch[j] - 2 * (1/epoch) * self.b)
                    # If the sample is correctly classified, do not update the weights and biases
                    else:
                        self.W = self.W - learning_rate * (2 * (1/epoch) * self.W)
                        self.b = self.b - learning_rate * (2 * (1/epoch) * self.b)

    def predict(self, X):
        y_pred = []
        # Iterate over all samples in the dataset
        for i in range(X.shape[0]):
            # Predict the label for the sample
            if self.kernel == 'linear':
                y = np.dot(self.W, X[i]) + self.b
            elif self.kernel == 'rbf':
                y = np.sum(np.exp(-self.gamma * np.sum((X[i] - self.X)**2, axis=1)))
            if y >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred
