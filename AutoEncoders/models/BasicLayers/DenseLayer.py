import numpy as np
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim)
    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        if self.activation is not None:
            self.Z = self.activation(self.Z)
        return self.Z
    def backward(self, dZ):
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0)
        self.dX = np.dot(dZ, self.W.T)
        return self.dX
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db 

# Path: AutoEncoders/models/Sequential.py


        