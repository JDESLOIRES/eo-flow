import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.layer_utils import print_summary

from eoflow.models.layers import ResidualBlock
from eoflow.models.tempnets_task.tempnets_base import BaseTempnetsModel, BaseCustomTempnetsModel, BaseModelAdapt, BaseModelAdaptV2
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class TCNModel(BaseCustomTempnetsModel):
    """ Implementation of the TCN network taken form the keras-TCN implementation

        https://github.com/philipperemy/keras-tcn
    """

    class TCNModelSchema(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.', example=0.5)

        kernel_size = fields.Int(missing=2, description='Size of the convolution kernels.')
        nb_filters = fields.Int(missing=64, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=1)
        dilations = fields.List(fields.Int, missing=[1, 2, 4, 8, 16, 32], description='Size of dilations used in the '
                                                                                      'covolutional tf.keras.layers')
        padding = fields.String(missing='CAUSAL', validate=OneOf(['CAUSAL', 'SAME']),
                                description='Padding type used in convolutions.')
        use_skip_connections = fields.Bool(missing=True, description='Flag to whether to use skip connections.')
        return_sequences = fields.Bool(missing=False, description='Flag to whether return sequences or not.')
        activation = fields.Str(missing='linear', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=0, description='L2 regularization parameter.')

        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        layer_norm = fields.Bool(missing=False, description='Whether to use layer normalisation.')

    def _cnn_layer(self, net):

        dropout_rate = 1 - self.config.keep_prob

        layer = tf.keras.layers.Conv1D(filters= self.config.nb_filters,
                                       kernel_size=self.config.kernel_size,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)
        return layer

    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        dropout_rate = 1 - self.config.keep_prob

        net = x

        net = self._cnn_layer(net)

        # list to hold all the member ResidualBlocks
        residual_blocks = []
        skip_connections = []

        total_num_blocks = self.config.nb_conv_stacks * len(self.config.dilations)
        if not self.config.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for _ in range(self.config.nb_conv_stacks):
            for d in self.config.dilations:
                net, skip_out = ResidualBlock(dilation_rate=d,
                                              nb_filters=self.config.nb_filters,
                                              kernel_size=self.config.kernel_size,
                                              padding=self.config.padding,
                                              activation=self.config.activation,
                                              dropout_rate=dropout_rate,
                                              use_batch_norm=self.config.batch_norm,
                                              use_layer_norm=self.config.layer_norm,
                                              kernel_initializer=self.config.kernel_initializer,
                                              last_block=len(residual_blocks) + 1 == total_num_blocks,
                                              name=f'residual_block_{len(residual_blocks)}')(net)
                residual_blocks.append(net)
                skip_connections.append(skip_out)


        # Author: @karolbadowski.
        output_slice_index = int(net.shape.as_list()[1] / 2) \
            if self.config.padding.lower() == 'same' else -1
        lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])

        if self.config.use_skip_connections:
            net = tf.keras.layers.add(skip_connections)

        if not self.config.return_sequences:
            net = lambda_layer(net)

        net = tf.keras.layers.Dense(1, activation='linear')(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        #print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)



class TempCNNModel(BaseCustomTempnetsModel,BaseModelAdaptV2):
    """ Implementation of the TempCNN network taken from the temporalCNN implementation

        https://github.com/charlotte-pel/temporalCNN
    """

    class TempCNNModelSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.',
                                 example=0.5)
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        n_strides = fields.Int(missing=1, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected tf.keras.layers.')
        fc_activation = fields.Str(missing='relu', description='Activation function used in final FC tf.keras.layers.')

        emb_layer = fields.String(missing='Flatten', validate=OneOf(['Flatten', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']),
                                  description='Final layer after the convolutions.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        n_classes = fields.Int(missing=1, description='Number of classes')
        output_activation = fields.String(missing='linear', description='Output activation')
        residual_block = fields.Bool(missing=False, description= 'Add residual block')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=0.0, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        str_inc = fields.Bool(missing=False, description='Increase strides')
        ker_inc = fields.Bool(missing=False, description='Increase kernels')
        ker_dec = fields.Bool(missing=False, description='Decrease kernels')
        fc_dec = fields.Bool(missing=False, description='Decrease dense neurons')
        multioutput = fields.Bool(missing=False, description='Decrease dense neurons')
        batch_norm = fields.Bool(missing=True, description='Whether to use batch normalisation.')
        gradient_reversal = fields.Bool(missing=True, description='Whether to use gradient reversal.')

    def _cnn_layer(self, net, i = 0, first = False):

        dropout_rate = 1 - self.config.keep_prob
        filters = self.config.nb_conv_filters
        kernel_size = self.config.kernel_size
        n_strides = self.config.n_strides

        if self.config.enumerate:
            filters = filters * (2**i)
        if self.config.ker_inc:
            kernel_size = kernel_size * (i+1)

        if self.config.ker_dec:
            kernel_size = self.config.kernel_size // (i+1)
            if kernel_size ==0: kernel_size += 1

        if self.config.str_inc:
            n_strides = 1 if first else 2

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

    def _fcn_layer(self, net, i=0):
        dropout_rate = 1 - self.config.keep_prob
        nb_neurons = self.config.nb_fc_neurons
        if self.config.fc_dec:
            nb_neurons /= 2**i
        layer_fcn = Dense(units=nb_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn


    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        conv = self._cnn_layer(net, 0, first = True)
        for i, _ in enumerate(range(self.config.nb_conv_stacks-1)):
            conv = self._cnn_layer(conv, i+1)

        embedding = self._embeddings(conv)

        net_mean = self._fcn_layer(embedding)

        for i in range(1, self.config.nb_fc_stacks):
            net_mean = self._fcn_layer(net_mean, i)

        output = Dense(units = self.config.n_classes,
                       activation = self.config.output_activation,
                       kernel_initializer=self.config.kernel_initializer,
                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net_mean)

        if self.config.multioutput or self.config.loss in ['gaussian', 'laplacian']:
            net_std = self._fcn_layer(embedding)
            for i in range(1, self.config.nb_fc_stacks):
                net_std = self._fcn_layer(net_std, i)

            output_sigma = Dense(units=self.config.n_classes,
                                 activation=self.config.output_activation,
                                 kernel_initializer=self.config.kernel_initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net_std)

            self.net = tf.keras.Model(inputs=x, outputs=[output, output_sigma, embedding])
        else:
            self.net = tf.keras.Model(inputs=x, outputs=[output, embedding])

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)



class BayesTempCNNModel(BaseCustomTempnetsModel):
    """ Implementation of the TempCNN network taken from the temporalCNN implementation

        https://github.com/charlotte-pel/temporalCNN
    """

    class BayesTempCNNModel(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.', example=0.5)
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        n_strides = fields.Int(missing=1, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected tf.keras.layers.')
        fc_activation = fields.Str(missing='relu', description='Activation function used in final FC tf.keras.layers.')

        emb_layer = fields.String(missing='Flatten', validate=OneOf(['Flatten', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']),
                                  description='Final layer after the convolutions.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        n_classes = fields.Int(missing=1, description='Number of classes')
        output_activation = fields.String(missing='linear', description='Output activation')
        residual_block = fields.Bool(missing=False, description= 'Add residual block')
        activity_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        str_inc = fields.Bool(missing=False, description='Increase strides')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        fc_dec = fields.Bool(missing=False, description='Decrease dense neurons')
        ker_inc = fields.Bool(missing=False, description='Increase kernels')
        ker_dec = fields.Bool(missing=False, description='Decreae kernels')

    def _cnn_layer(self, net, i = 0, first = False):

        dropout_rate = 1 - self.config.keep_prob
        filters = self.config.nb_conv_filters
        kernel_size = self.config.kernel_size
        n_strides = self.config.n_strides

        if self.config.enumerate:
            filters = filters * (2**i)
        if self.config.ker_inc:
            kernel_size = kernel_size * (i+1)

        if self.config.ker_dec:
            kernel_size = self.config.kernel_size // (i+1)
            if kernel_size ==0: kernel_size += 1

        if self.config.str_inc:
            n_strides = 1 if first else 2

        layer = tfp.layers.Convolution1DReparameterization(filters=filters,
                                                           kernel_size=kernel_size,
                                                           strides=n_strides,
                                                           activation = self.config.activation,
                                                           activity_regularizer=tf.keras.regularizers.l2(self.config.activity_regularizer),
                                                           padding=self.config.padding)(net)
        if self.config.batch_norm:
            layer = tfp.bijectors.BatchNormalization()(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)

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

    def _fcn_layer(self, net, i=0):
        dropout_rate = 1 - self.config.keep_prob
        nb_neurons = self.config.nb_fc_neurons

        if self.config.fc_dec:
            nb_neurons //= 2**i

        layer_fcn = tfp.layers.DenseReparameterization(units=nb_neurons,
                                                       activation=self.config.activation,
                                                       activity_regularizer=tf.keras.regularizers.l2(self.config.activity_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tfp.bijectors.BatchNormalization()(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)

        #if self.config.fc_activation: layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn


    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        net = self._cnn_layer(net, 0, first = True)
        for i, _ in enumerate(range(self.config.nb_conv_stacks-1)):
            net = self._cnn_layer(net, i+1)

        embedding = self._embeddings(net)
        net_mean = self._fcn_layer(embedding)
        net_std = self._fcn_layer(embedding)

        for i in range(1, self.config.nb_fc_stacks):
            net_mean = self._fcn_layer(net_mean, i)

        output = tfp.layers.DenseReparameterization(units = self.config.n_classes,
                                                    activity_regularizer=tf.keras.regularizers.l2(
                                                        self.config.activity_regularizer),
                                                    activation = self.config.output_activation)(net_mean)

        if self.config.loss in ['gaussian', 'laplacian']:
            for i in range(1, self.config.nb_fc_stacks):
                net_std = self._fcn_layer(net_std, i)

            output_sigma = tfp.layers.DenseReparameterization(units=self.config.n_classes,
                                                   activity_regularizer=tf.keras.regularizers.l2(
                                                       self.config.activity_regularizer),
                                                   activation=self.config.output_activation)(net_std)
            self.net = tf.keras.Model(inputs=x, outputs=[output, output_sigma])
        else:
            self.net = tf.keras.Model(inputs=x, outputs=output)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)




class HistogramCNNModel(BaseCustomTempnetsModel):
    """ Implementation of the CNN2D with histogram time series

        https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf
        https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/3%20model/nnet_for_hist_dropout_stride.py
    """

    class HistogramCNNModel(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.', example=0.5)
        kernel_size = fields.List(fields.Int, missing=[3,3], description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        n_strides = fields.List(fields.Int, missing=[1, 1], description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected tf.keras.layers.')
        emb_layer = fields.String(missing='Flatten',
                                  validate=OneOf(['Flatten', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D']),
                                  description='Final layer after the convolutions.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        fc_activation = fields.Str(missing='relu', description='Activation function used in final FC tf.keras.layers.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        fc_dec = fields.Bool(missing=False, description='Decrease dense neurons')
        ker_inc = fields.Bool(missing=False, description='Increase kernels')
        ker_dec = fields.Bool(missing=False, description='Decrease kernels')

    def _cnn_layer(self, net, i = 1, last = False):

        dropout_rate = 1 - self.config.keep_prob
        filters = self.config.nb_conv_filters
        kernel_size = np.array(self.config.kernel_size)
        n_strides = self.config.n_strides

        if self.config.enumerate:
            filters = filters * (2**i)
            n_strides = 1 if not last else 2

        if self.config.ker_inc:
            kernel_size = kernel_size * (i+1)

        if self.config.ker_dec:
            kernel_size = kernel_size // (i+1)
            if kernel_size[0] == 0:
                kernel_size += 1

        layer = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=list(kernel_size),
                                       strides=n_strides,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)
        return layer


    def _fcn_layer(self, net):
        dropout_rate = 1 - self.config.keep_prob
        layer_fcn = Dense(units=self.config.nb_fc_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn


    def _embeddings(self,net):

        name = "embedding"
        if self.config.emb_layer == 'Flatten':
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.emb_layer == 'GlobalAveragePooling2D':
            net = tf.keras.layers.GlobalAveragePooling2D(name=name)(net)
        elif self.config.emb_layer == 'GlobalMaxPooling2D':
            net = tf.keras.layers.GlobalMaxPooling2D(name=name)(net)
        return net


    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        for i, _ in enumerate(range(self.config.nb_conv_stacks-1)):
            net = self._cnn_layer(net, i)
            net = self._cnn_layer(net, i, True)

        #net = self._cnn_layer(net, self.config.nb_conv_stacks-1)
        #net = self._cnn_layer(net, self.config.nb_conv_stacks-1)
        net = self._cnn_layer(net, self.config.nb_conv_stacks-1, True)

        net = self._embeddings(net)

        for _ in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net)

        net = Dense(units = 1,
                    activation = 'linear',
                    kernel_initializer=self.config.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)

    def get_feature_map(self, inputs, training=None):
        return self.backbone(inputs, training)






class InceptionCNN(BaseCustomTempnetsModel):
    '''
    https://github.com/hfawaz/InceptionTime
    '''

    class InceptionCNN(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.', example=0.5)
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=32, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        n_stride = fields.Int(missing=1, description='Value of convolutional strides.')
        bottleneck_size = fields.Int(missing=32, description='Bottleneck size.')
        use_residual = fields.Bool(missing=False,description='Use residuals.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected tf.keras.layers.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        fc_activation = fields.Str(missing='relu', description='Activation function used in final FC tf.keras.layers.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        use_bottleneck = fields.Bool(missing=True, description='use_bottleneck')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        nb_class = fields.Int(missing=1, description='Number of class.')
        output_activation = fields.Str(missing='linear', description='Output activation.')
        n_stride = fields.Int(missing=1, description='Number of strides.')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.config.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.config.bottleneck_size,
                                                     kernel_size=1,
                                                     padding='SAME',
                                                     activation=activation,
                                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [1,2,3]
        #kernel_size_s = [5 // (2 ** i) for i in range(3)]

        conv_list = [
            tf.keras.layers.Conv1D(
                filters=self.config.nb_conv_filters,
                kernel_size=kernel_size_,
                strides=stride,
                padding='SAME',
                activation=activation,
                use_bias=False,
            )(input_inception)
            for kernel_size_ in kernel_size_s
        ]

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=self.config.n_strides, padding='SAME')(input_tensor)

        conv_6 = tf.keras.layers.Conv1D(filters=self.config.nb_conv_filters,
                                        kernel_size=1,
                                        padding='SAME',
                                        activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(1-self.config.keep_prob)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
                                            kernel_size=1,
                                            padding='SAME', use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        shortcut_y = tf.keras.layers.Dropout(1 - self.config.keep_prob)(shortcut_y)
        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build(self, input_shape):
        input_layer = tf.keras.layers.Input(input_shape[1:])
        input_res = input_layer
        x = input_layer

        for d in range(self.config.nb_conv_stacks):

            x = self._inception_module(x)

            if self.config.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(self.config.nb_class,
                                             activation=self.config.output_activation)(gap_layer)

        self.net = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)



class TransformerCNN(BaseCustomTempnetsModel):
    """ Implementation of the Pixel-Set encoder + Temporal Attention Encoder sequence classifier

    Code is based on the Pytorch implementation of V. Sainte Fare Garnot et al. https://github.com/VSainteuf/pytorch-psetae
    """

    class TransformerCNN(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.',
                                 example=0.5)
        num_heads = fields.Int(missing=4, description='Number of Attention heads.')
        head_size = fields.Int(missing=64, description='Size Attention heads.')
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        num_transformer_blocks = fields.Int(missing=4, description='Number of transformer blocks.')
        ff_dim = fields.Int(missing=4, description='Number of feed-forward neurons in point-wise CNN.')
        batch_norm = fields.Bool(missing=False,description='Use batch normalisation.')
        n_conv = fields.Int(missing=3, description='Number of Attention heads.')
        d_model = fields.Int(missing=None, description='Depth of model.')

        mlp_units = fields.List(fields.Int, missing=[64], description='Number of units for each layer in mlp.')
        mlp_dropout = fields.Float(required=True, description='Keep probability used in dropout MLP layers.',
                                   example=0.5)
        emb_layer = fields.Str(missing='Flatten', description='Embedding layer.')
        output_activation = fields.Str(missing='linear', description='Output activation.')
        n_classes = fields.Int(missing=1, description='# Classes.')
        n_strides = fields.Int(missing=1, description='# strides.')

    def transformer_encoder(self, inputs):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=self.config.head_size,
            num_heads=self.config.num_heads,
            dropout=1-self.config.keep_prob)(x, x)
        x = tf.keras.layers.Dropout(1-self.config.keep_prob)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        for _ in range(self.config.n_conv):
            x = tf.keras.layers.Conv1D(filters=self.config.ff_dim,
                                       padding='SAME',
                                       strides=self.config.n_strides,
                                       kernel_size=self.config.kernel_size)(x)
            #if self.config.batch_norm: x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(1 - self.config.keep_prob)(x)
            x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(padding='SAME',
                                   filters=inputs.shape[-1],
                                   kernel_size=1)(x)
        return x, res

    def _embeddings(self,net):

        name = "embedding"
        if self.config.emb_layer == 'Flatten':
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.emb_layer == 'GlobalAveragePooling1D':
            net = tf.keras.layers.GlobalAveragePooling1D(name=name,data_format="channels_first")(net)
        elif self.config.emb_layer == 'GlobalMaxPooling1D':
            net = tf.keras.layers.GlobalMaxPooling1D(name=name,data_format="channels_first")(net)
        return net
    
    def build(self, inputs_shape):
        input_layer = tf.keras.layers.Input(inputs_shape[1:])
        x = input_layer
        for _ in range(self.config.num_transformer_blocks):
            lay, res = self.transformer_encoder(x)
            x = tf.keras.layers.Add()([lay, res])
            x = tf.keras.layers.Activation('relu')(x)

        x = self._embeddings(x)

        for dim in self.config.mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            if self.config.batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.config.mlp_dropout)(x)
            x = tf.keras.layers.Activation('relu')(x)

        outputs = tf.keras.layers.Dense(self.config.n_classes, activation=self.config.output_activation)(x)
        self.net = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class CNNTaeSchema(BaseCustomTempnetsModel):
    """ Implementation of the Pixel-Set encoder + Temporal Attention Encoder sequence classifier

    Code is based on the Pytorch implementation of V. Sainte Fare Garnot et al. https://github.com/VSainteuf/pytorch-psetae
    """

    class CNNTaeSchema(BaseCustomTempnetsModel._Schema):

        num_heads = fields.Int(missing=4, description='Number of Attention heads.')
        num_dff = fields.Int(missing=32, description='Number of feed-forward neurons in point-wise MLP.')
        d_model = fields.Int(missing=None, description='Depth of model.')
        mlp3 = fields.List(fields.Int, missing=[512, 128, 128], description='Number of units for each layer in mlp3.')
        dropout = fields.Float(missing=0.2, description='Dropout rate for attention encoder.')
        T = fields.Float(missing=1000, description='Number of features for attention.')
        len_max_seq = fields.Int(missing=24, description='Number of features for attention.')
        mlp4 = fields.List(fields.Int, missing=[128, 64, 32], description='Number of units for each layer in mlp4. ')

    def init_model(self):
        # TODO: missing features from original PseTae:
        #   * spatial encoder extra features (hand-made)
        #   * spatial encoder masking

        self.spatial_encoder = pse_tae_tf.keras.layers.PixelSetEncoder(
            mlp1=self.config.mlp1,
            mlp2=self.config.mlp2,
            pooling=self.config.pooling)

        self.temporal_encoder = pse_tae_tf.keras.layers.TemporalAttentionEncoder(
            n_head=self.config.num_heads,
            d_k=self.config.num_dff,
            d_model=self.config.d_model,
            n_neurons=self.config.mlp3,
            dropout=self.config.dropout,
            T=self.config.T,
            len_max_seq=self.config.len_max_seq)

        mlp4_tf.keras.layers = [pse_tae_tf.keras.layers.LinearLayer(out_dim) for out_dim in self.config.mlp4]
        # Final layer (logits)
        mlp4_tf.keras.layers.append(pse_tae_tf.keras.layers.LinearLayer(1, batch_norm=False, activation='linear'))

        self.mlp4 = tf.keras.Sequential(mlp4_tf.keras.layers)

    def call(self, inputs, training=None, mask=None):

        out = self.spatial_encoder(inputs, training=training, mask=mask)
        out = self.temporal_encoder(out, training=training, mask=mask)
        out = self.mlp4(out, training=training, mask=mask)

        return out