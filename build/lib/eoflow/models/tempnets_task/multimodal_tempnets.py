import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.layer_utils import print_summary

from eoflow.models.layers import ResidualBlock
from eoflow.models.tempnets_task.tempnets_base import BaseTempnetsModel, BaseCustomTempnetsModel, \
    BaseModelAdapt, BaseModelAdaptV2, BaseModelAdaptV3, BaseModelAdaptCoral, BaseModelMultiview
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class MultimodalModel(BaseCustomTempnetsModel):
    """
    Implementation of the TempCNN network taken from the temporalCNN implementation
    https://github.com/charlotte-pel/temporalCNN
    """

    class MultimodalModelSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.', example=0.5)
        keep_prob_conv = fields.Float(missing=0.8, description='Keep probability used in dropout tf.keras.layers.')
        kernel_size = fields.List(missing=[5, 2, 2], description='Size of the convolution kernels.')
        nb_conv_filters = fields.List(missing=[16, 32, 64], description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        n_strides = fields.List(missing=[1, 1, 1], description='Value of convolutional strides.')
        nb_fc_neurons = fields.List(missing=[256, 128], description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=2, description='Number of fully connected tf.keras.layers.')
        fc_activation = fields.Str(missing='relu', description='Activation function used in final FC tf.keras.layers.')

        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        n_classes = fields.Int(missing=1, description='Number of classes')
        output_activation = fields.String(missing='linear', description='Output activation')

        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=0.0, description='L2 regularization parameter.')

        ema = fields.Bool(missing=True, description='Apply EMA')

        batch_norm = fields.Bool(missing=True, description='Whether to use batch normalisation.')

    def _cnn_layer(self, net, filters, n_strides, kernel_size):

        dropout_rate = 1 - self.config.keep_prob_conv

        layer = tf.keras.layers.Conv1D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=n_strides,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)

        return layer


    def _embeddings(self,net):
        name = "embedding"
        if self.config.emb_layer == 'Flatten':
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.emb_layer == 'GlobalAveragePooling1D':
            net = tf.keras.layers.GlobalAveragePooling1D(name=name)(net)
        elif self.config.emb_layer == 'GlobalMaxPooling1D':
            net = tf.keras.layers.GlobalMaxPooling1D(name=name)(net)
        return net


    def _fcn_layer(self, net,nb_neurons):
        dropout_rate = 1 - self.config.keep_prob

        layer_fcn = Dense(units=nb_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn


    def build(self, inputs_shape, lambda_ = 1.0):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        conv = self._cnn_layer(net)
        for i, _ in enumerate(range(self.config.nb_conv_stacks-1)):
            conv = self._cnn_layer(conv, i+1)

        embedding = self._embeddings(conv)

        net_mean_emb = self._fcn_layer(embedding)
        for i in range(1, self.config.nb_fc_stacks):
            net_mean_emb = self._fcn_layer(net_mean_emb, i)

        output = Dense(units = self.config.n_classes,
                       activation = self.config.output_activation,
                       kernel_initializer=self.config.kernel_initializer,
                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net_mean_emb)

        if self.config.multioutput or self.config.loss in ['gaussian', 'laplacian']:
            net_aux = self._fcn_layer(embedding)
            for i in range(1, self.config.nb_fc_stacks):
                net_aux = self._fcn_layer(net_aux, i)

            output_sigma = Dense(units=1,
                                 activation=self.config.output_activation,
                                 kernel_initializer=self.config.kernel_initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net_aux)
            self.net = tf.keras.Model(inputs=x, outputs=[output, output_sigma, embedding])
        else:
            self.net = tf.keras.Model(inputs=x, outputs=[output, embedding])

        print(self.net.summary())

    def call(self, inputs, training=None):
        return self.net(inputs, training)

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
