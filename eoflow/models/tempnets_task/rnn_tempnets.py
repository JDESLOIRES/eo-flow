import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from keras.layers import TimeDistributed
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.python.keras.utils.layer_utils import print_summary


from eoflow.models.tempnets_task.tempnets_base import (
    BaseTempnetsModel,
    BaseCustomTempnetsModel,
)
from eoflow.models.layers import Sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

rnn_layers = dict(rnn=SimpleRNN, gru=GRU, lstm=LSTM)


class BiRNN(BaseCustomTempnetsModel):
    """Implementation of a Bidirectional Recurrent Neural Network

    This implementation allows users to define which RNN layer to use, e.g. SimpleRNN, GRU or LSTM
    """

    class BiRNNModelSchema(BaseCustomTempnetsModel._Schema):
        rnn_layer = fields.String(
            required=True,
            validate=OneOf(["rnn", "lstm", "gru"]),
            description="Type of RNN layer to use",
        )

        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )

        rnn_units = fields.Int(
            missing=64, description="Size of the convolution kernels."
        )
        rnn_blocks = fields.Int(missing=1, description="Number of LSTM blocks")

        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=1e-6, description="L2 regularization parameter."
        )
        nb_fc_stacks = fields.Int(
            missing=0, description="Number of fully connected layers."
        )
        nb_fc_neurons = fields.Int(
            missing=0, description="Number of fully connected neurons."
        )
        fc_activation = fields.Str(
            missing=None, description="Activation function used in final FC layers."
        )

        batch_norm = fields.Bool(
            missing=False, description="Whether to use batch normalisation."
        )
        multioutput = fields.Bool(missing=False, description="Decrease dense neurons")

    def _rnn_layer(self, net, last=False):
        """Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer."""
        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob_conv

        layer = RNNLayer(
            units=self.config.rnn_units,
            dropout=dropout_rate,
            return_sequences=not last,
        )

        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer(net)

    def _fcn_layer(self, net):
        dropout_rate = 1 - self.config.keep_prob_conv
        layer_fcn = Dense(
            units=self.config.nb_fc_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)
        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """Creates the RNN model architecture."""

        x = tf.keras.layers.Input(inputs_shape[1:])
        net = x

        if self.config.layer_norm:
            net = tf.keras.layers.LayerNormalization(axis=-1)(net)

        for _ in range(self.config.rnn_blocks - 1):
            net = self._rnn_layer(net)
        net = self._rnn_layer(net, last=True)

        if self.config.layer_norm:
            net = tf.keras.layers.LayerNormalization(axis=-1)(net)

        for _ in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net)

        net = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class MultiBranchRNN(BaseCustomTempnetsModel):
    """
    Implementation of the TempCNN network taken from the temporalCNN implementation
    https://github.com/charlotte-pel/temporalCNN
    """

    class MultiBranchRNN(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )

        rnn_units = fields.Int(
            missing=16, description="Number of convolutional filters."
        )
        rnn_blocks = fields.Int(
            missing=3, description="Number of convolutional blocks."
        )

        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=1e-6, description="L2 regularization parameter."
        )

        bidirectional = fields.Bool(
            missing=False, description="Whether to use a bidirectional layer"
        )
        layer_norm = fields.Bool(
            missing=False,
            description="Whether to apply layer normalization in the encoder.",
        )

        rnn_layer = fields.String(
            required=True,
            validate=OneOf(["lstm", "gru"]),
            description="Type of RNN layer to use",
        )
        multibranch = fields.Bool(missing=False, description="Multibranch model")
        multioutput = fields.Bool(missing=False, description="Decrease dense neurons")
        finetuning = fields.Bool(
            missing=False, description="Unfreeze layers after patience"
        )

        nb_fc_neurons = fields.Int(
            missing=128, description="Number of Fully Connect neurons."
        )
        static_fc_neurons = fields.Int(
            missing=10, description="Number of Fully Connect neurons."
        )
        nb_fc_stacks = fields.Int(
            missing=2, description="Number of fully connected tf.keras.layers."
        )
        fc_activation = fields.Str(
            missing="relu",
            description="Activation function used in final FC tf.keras.layers.",
        )
        dims = fields.Int(missing=6, description="Number of  dimensions.")
        batch_norm = fields.Bool(
            missing=True, description="Whether to use batch normalisation."
        )

        activation = fields.Str(
            missing="relu", description="Activation function used in final filters."
        )
        n_classes = fields.Int(missing=1, description="Number of classes")
        output_activation = fields.String(
            missing="linear", description="Output activation"
        )

        ema = fields.Bool(missing=True, description="Apply EMA")

    def _rnn_layer(self, net, i, last=False):
        """Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer."""
        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob

        denom = self.config.factor * (i - 1)
        if denom == 0:
            denom += 1

        layer = RNNLayer(
            units=self.config.rnn_units // denom,
            dropout=dropout_rate,
            return_sequences=not last,
        )
        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer(net)

    def _fcn_layer(self, net, nb_neurons):
        dropout_rate = 1 - self.config.keep_prob

        layer_fcn = Dense(
            units=nb_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(self.config.fc_activation)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """

        list_inputs = []
        list_submodels = []

        for input in inputs_shape[0]:
            x = tf.keras.layers.Input(input[1:])
            net = x
            if self.config.layer_norm:
                net = tf.keras.layers.LayerNormalization(axis=-1)(net)

            for i in range(1, self.config.rnn_blocks):
                net = self._rnn_layer(net, i)

            list_submodels.append(
                self._rnn_layer(net, self.config.rnn_blocks, last=True)
            )
            list_inputs.append(x)

        if len(inputs_shape) > 1:
            x = tf.keras.layers.Input(inputs_shape[1][1:])
            net = x
            fc_net = self._fcn_layer(net, self.config.static_fc_neurons)
            fc_net = self._fcn_layer(fc_net, self.config.static_fc_neurons // 2)
            list_submodels.append(fc_net)
            list_inputs.append(x)

        data_fusion = tf.keras.layers.Concatenate(axis=1)(list_submodels)

        fc = self._fcn_layer(data_fusion, nb_neurons=self.config.nb_fc_neurons)
        if self.config.reduce:
            fc = self._fcn_layer(fc, nb_neurons=self.config.nb_fc_neurons // 4)
        else:
            fc = self._fcn_layer(fc, nb_neurons=self.config.nb_fc_neurons // 2)

        output = tf.keras.layers.Dense(
            units=self.config.n_classes,
            activation=self.config.output_activation,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(fc)

        self.net = tf.keras.Model(inputs=[list_inputs], outputs=[output, data_fusion])
        print(self.net.summary())

    def call(self, inputs, training=None):
        return self.net(inputs, training)


# https://www.sciencedirect.com/science/article/pii/S0034425721003205


class VAERNN(BaseCustomTempnetsModel):
    """Implementation of a Bidirectional Recurrent Neural Network

    This implementation allows users to define which RNN layer to use, e.g. SimpleRNN, GRU or LSTM
    """

    class VAERNN(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )

        rnn_units = fields.Int(
            missing=16, description="Number of convolutional filters."
        )
        rnn_blocks = fields.Int(
            missing=3, description="Number of convolutional blocks."
        )
        factor = fields.Int(missing=2, description="Number of convolutional blocks.")

        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=1e-6, description="L2 regularization parameter."
        )

        bidirectional = fields.Bool(
            missing=False, description="Whether to use a bidirectional layer"
        )
        layer_norm = fields.Bool(
            missing=False,
            description="Whether to apply layer normalization in the encoder.",
        )

        rnn_layer = fields.String(
            required=True,
            validate=OneOf(["lstm", "gru"]),
            description="Type of RNN layer to use",
        )
        multibranch = fields.Bool(missing=False, description="Multibranch model")
        multioutput = fields.Bool(missing=False, description="Decrease dense neurons")
        finetuning = fields.Bool(
            missing=False, description="Unfreeze layers after patience"
        )

        variational = fields.Bool(
            missing=True, description="Unfreeze layers after patience"
        )
        output_shape = fields.List(
            fields.Int, description="Unfreeze layers after patience"
        )

    def _rnn_layer(self, net, i, last=False):
        """Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer."""
        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob

        denom = self.config.factor * (i - 1)
        if denom == 0:
            denom += 1

        layer = RNNLayer(
            units=self.config.rnn_units // denom,
            dropout=dropout_rate,
            return_sequences=not last,
        )
        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer(net)

    def build(self, inputs_shape):
        """Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """

        x = tf.keras.layers.Input(shape=inputs_shape[1:])
        print(x.shape)
        net = x

        if self.config.layer_norm:
            net = tf.keras.layers.LayerNormalization(axis=-1)(net)

        for i in range(1, self.config.rnn_blocks):
            net = self._rnn_layer(net, i)

        enc = self._rnn_layer(net, self.config.rnn_blocks, last=True)

        if self.config.variational:
            dense_mean = tf.keras.layers.Dense(
                units=self.config.rnn_units
                // (self.config.factor * (self.config.rnn_blocks - 1))
            )(
                enc
            )  # ,activation=embedding_activation
            dense_log_var = tf.keras.layers.Dense(
                units=self.config.rnn_units
                // (self.config.factor * (self.config.rnn_blocks - 1))
            )(
                enc
            )  # ,activation=embedding_activation
            sampling = Sampling()((dense_mean, dense_log_var))
            net = tf.keras.layers.RepeatVector(self.config.output_shape[0])(sampling)
        else:
            net = tf.keras.layers.RepeatVector(self.config.output_shape[0])(enc)

        for i in range(self.config.rnn_blocks, 0, -1):
            net = self._rnn_layer(net, i)
        output_layer = tf.keras.layers.TimeDistributed(
            Dense(self.config.output_shape[1])
        )(net)

        if self.config.multioutput:
            labels = tf.keras.layers.Dense(
                units=1,
                activation="linear",
                kernel_initializer=self.config.kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(
                    self.config.kernel_regularizer
                ),
            )(enc)

            self.net = tf.keras.Model(inputs=x, outputs=[output_layer, labels])

        else:
            self.net = tf.keras.Model(inputs=x, outputs=output_layer)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)
