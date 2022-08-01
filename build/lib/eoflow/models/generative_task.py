

##########################################################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, \
    Layer, Activation, Dense, Dropout, BatchNormalization, RNN, GRU

from tensorflow.keras.regularizers import l2

tf.keras.backend.set_floatx('float32')



#######################################################################################################################


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GRUVariationalAutoencoder(tf.keras.Model):
    '''
    Variational Autoencoder using GRU cells
    '''
    def __init__(self,
                 ndim = 64,
                 output_shape,
                 hidden_activation='tanh',
                 kernel_initializer = 'glorot_uniform'):

        super(GRUVariationalAutoencoder, self).__init__(name='GRUVariationalAutoencoder')

        self.enc1 = GRU(ndim, activation=hidden_activation, return_sequences=True,
                        kernel_initializer=kernel_initializer)
        self.bottleneck = GRU(ndim//4, activation=hidden_activation, return_sequences=False,
                              kernel_initializer=kernel_initializer)
        self.dense_mean = tf.keras.layers.Dense(units=ndim//4) #,activation=embedding_activation
        self.dense_log_var = tf.keras.layers.Dense(units=ndim//4) #,activation=embedding_activation
        self.sampling = Sampling()
        self.embedding = tf.keras.layers.RepeatVector(output_shape[0])
        self.dec1 = GRU(ndim//4, activation=hidden_activation, return_sequences=True,
                        kernel_initializer=kernel_initializer)
        self.dec2 = GRU(ndim, activation=hidden_activation, return_sequences=True,
                        kernel_initializer=kernel_initializer)
        self.output_layer = tf.keras.layers.TimeDistributed(Dense(output_shape[1]))

    def call(self, inputs):
        encode = self.enc1(inputs)
        bottleneck = self.bottleneck(encode)
        z_mean = self.dense_mean(bottleneck)
        z_log_var = self.dense_log_var(bottleneck)
        z = self.sampling((z_mean, z_log_var))
        embedding = self.embedding(z)
        decoder = self.dec1(embedding)
        decoder2 = self.dec2(decoder)
        return self.output_layer(decoder2)