import numpy as np
import tensorflow as tf
from BasicLayers import DenseLayerTF,DenseLayer

# Load the MNIST dataset

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Apply basic Normalization range 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Reshape the input data  so they are 2D
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod[x_test.shape[1:]]))


# Define the input placeholder

input_dim  = x_train.shape[1]
inputs = tf.placeholder(dtype=tf.float32,shape=(None,input_dim))

encoding_dim  = 32
# Define AutoEncoder
class AutoEncoder(input_dim,encoding_dim,activation=tf.nn.relu):
    def __init__(self,input_dim,encoding_dim,activation=tf.nn.relu):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.encoder = DenseLayerTF(input_dim,encoding_dim,activation)
        self.decoder = DenseLayerTF(encoding_dim,input_dim,activation)
    def forward(self,X):
        self.X = X
        self.Z = self.encoder.forward(X)
        self.X_hat = self.decoder.forward(self.Z)
        return self.X_hat
    def backward(self,dX_hat):
        self.dZ = self.decoder.backward(dX_hat)
        self.dX = self.encoder.backward(self.dZ)
        return self.dX
    def update(self,lr):
        self.encoder.update(lr)
        self.decoder.update(lr)
    def loss(self):
        return tf.reduce_mean(tf.square(self.X - self.X_hat))
    

# dEF
NUM_EPOCHS = 10
BATCH_SIZE = 128
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    autoencoder = AutoEncoder(input_dim,encoding_dim)
    for epoch in range(NUM_EPOCHS):
        for i in range(0,len(x_train),BATCH_SIZE):
            X = x_train[i:i+BATCH_SIZE]
            X_hat = autoencoder.forward(X)
            dX_hat = autoencoder.backward(X_hat)
            autoencoder.update(0.01)
        print("Epoch: {} Loss: {}".format(epoch,autoencoder.loss()))
    X_hat = autoencoder.forward(x_test)
    print("Test Loss: {}".format(autoencoder.loss()))
    
    
    
    
# TEST MODEL
#     
# Use the encoder model to encode the input data
encoder = autoencoder
encoded_inputs = encoder.forward(x_test)

# Use the decoder model to decode the encoded data
decoded_outputs = autoencoder.forward(encoded_inputs)

# Print the original and decoded data
print(x_test[0])
print(decoded_outputs[0])
