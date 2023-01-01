import tensorflow as tf
class DenseLayer:
    def __init__(self, in_dim, out_dim, activation=None):
        self.W = tf.Variable(tf.random_normal(shape=(in_dim, out_dim)))
        self.b = tf.Variable(tf.zeros(shape=(out_dim)))
        self.activation = activation

    def __call__(self, inputs):
        output = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output