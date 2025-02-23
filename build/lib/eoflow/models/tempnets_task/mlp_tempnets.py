import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from keras.layers import TimeDistributed
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.python.keras.utils.layer_utils import print_summary

from eoflow.models.layers import ResidualBlock
from eoflow.models.tempnets_task.tempnets_base import (
    BaseTempnetsModel,
    BaseCustomTempnetsModel,
)

from eoflow.models import transformer_encoder_layers
from eoflow.models import pse_tae_layers
from eoflow.models.layers import Sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class MLP(BaseCustomTempnetsModel):
    """
    Implementation of the mlp network
    """

    class MLPSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )
        nb_fc_neurons = fields.Int(
            missing=256, description="Number of Fully Connect neurons."
        )
        nb_fc_stacks = fields.Int(
            missing=1, description="Number of fully connected layers."
        )
        activation = fields.Str(
            missing="relu", description="Activation function used in final filters."
        )
        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=1e-6, description="L2 regularization parameter."
        )
        batch_norm = fields.Bool(
            missing=False, description="Whether to use batch normalisation."
        )

        reduce = fields.Bool(missing=False, description="reduce number neurons")
        increase = fields.Bool(missing=False, description="increase number neurons")

        multibranch = fields.Bool(missing=False, description="Multibranch model")
        multioutput = fields.Bool(missing=False, description="Decrease dense neurons")
        finetuning = fields.Bool(
            missing=False, description="Unfreeze layers after patience"
        )

        adaptative = fields.Bool(missing=False, description="increase lr")
        factor = fields.Float(missing=1.0, description="increase lr")
        layer_before = fields.Int(missing=1, description="increase lr")
        variational = fields.Bool(
            missing=False, description="variational encoder for reconstruction purposes"
        )
        n_output = fields.Int(missing=1, description="Number of output neurons.")

    def _fcn_layer(self, net, i):

        dropout_rate = 1 - self.config.keep_prob
        nb_neurons = self.config.nb_fc_neurons
        if i == 1:
            if self.config.reduce:
                nb_neurons = nb_neurons // 2
            elif self.config.increase:
                nb_neurons = int(nb_neurons * 2)

        layer_fcn = Dense(
            units=nb_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(self.config.activation)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """Build TCN architecture

        The `inputs_shape` argument is a `(N, T*D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])
        net = x

        for i in range(self.config.nb_fc_stacks):
            if (self.config.nb_fc_stacks - (i + 1)) == self.config.layer_before:
                if self.config.variational:
                    dense_mean = self._fcn_layer(net, i)
                    dense_log_var = self._fcn_layer(net, i)
                    net = Sampling()((dense_mean, dense_log_var))
                else:
                    net = self._fcn_layer(net, i)
                enc = net
            else:
                net = self._fcn_layer(net, i)

        output = tf.keras.layers.Dense(
            units=self.config.n_output,
            activation="linear",
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)

        if self.config.multioutput or self.config.loss in ["gaussian", "laplacian"]:
            output_sigma = Dense(
                units=1,
                activation="linear",
                kernel_initializer=self.config.kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(
                    self.config.kernel_regularizer
                ),
            )(net)
            self.net = tf.keras.Model(inputs=x, outputs=[output, output_sigma])

        elif self.config.adaptative:
            self.net = tf.keras.Model(inputs=x, outputs=[output, enc])
        else:
            self.net = tf.keras.Model(inputs=x, outputs=output)

        self.encoder = tf.keras.Model(inputs=x, outputs=enc)

        print_summary(self.net)
        print_summary(self.encoder)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class MLPDANN(BaseCustomTempnetsModel):
    """
    Implementation of the mlp network
    """

    class MLPSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )
        nb_fc_neurons = fields.Int(
            missing=256, description="Number of Fully Connect neurons."
        )
        nb_fc_stacks = fields.Int(
            missing=1, description="Number of fully connected layers."
        )
        activation = fields.Str(
            missing="relu", description="Activation function used in final filters."
        )
        kernel_initializer = fields.Str(
            missing="he_normal", description="Method to initialise kernel parameters."
        )
        kernel_regularizer = fields.Float(
            missing=1e-6, description="L2 regularization parameter."
        )
        batch_norm = fields.Bool(
            missing=False, description="Whether to use batch normalisation."
        )

        reduce = fields.Bool(missing=False, description="reduce number neurons")
        increase = fields.Bool(missing=False, description="increase number neurons")

        multibranch = fields.Bool(missing=False, description="Multibranch model")
        multioutput = fields.Bool(missing=False, description="Decrease dense neurons")
        finetuning = fields.Bool(
            missing=False, description="Unfreeze layers after patience"
        )

        adaptative = fields.Bool(missing=False, description="increase lr")
        factor = fields.Float(missing=1.0, description="increase lr")
        layer_before = fields.Int(missing=1, description="increase lr")

    def _fcn_layer(self, net, i, batch_norm=False, sublayer=False):

        dropout_rate = 1 - self.config.keep_prob
        nb_neurons = self.config.nb_fc_neurons

        if self.config.reduce or sublayer:
            nb_neurons = nb_neurons // (2**i)
        elif self.config.increase:
            nb_neurons = int(nb_neurons * 2**i)

        layer_fcn = Dense(
            units=nb_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(self.config.activation)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """Build TCN architecture

        The `inputs_shape` argument is a `(N, T*D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])
        net = x

        for i in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net, i)

        discriminator = self._fcn_layer(
            net, self.config.nb_fc_stacks - 2, batch_norm=False, sublayer=True
        )
        discriminator = self._fcn_layer(
            discriminator, self.config.nb_fc_stacks - 1, batch_norm=False, sublayer=True
        )

        output_disc = tf.keras.layers.Dense(
            units=2,
            activation="softmax",
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(discriminator)

        task = self._fcn_layer(net, self.config.nb_fc_stacks - 2, sublayer=True)
        task = self._fcn_layer(task, self.config.nb_fc_stacks - 1, sublayer=True)
        output_task = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(task)
        #######################################################################################################################
        self.encoder = tf.keras.Model(inputs=x, outputs=net)
        self.discriminator = tf.keras.Model(inputs=net, outputs=output_disc)
        self.task = tf.keras.Model(inputs=net, outputs=output_task)
        self.net = tf.keras.Model(inputs=x, outputs=[output_task, net])

        print_summary(self.encoder)
        print_summary(self.discriminator)
        print_summary(self.task)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class TransformerEncoder(BaseTempnetsModel):
    """Implementation of a self-attention classifier
    Code is based on the Pytorch implementation of Marc Russwurm https://github.com/MarcCoru/crop-type-mapping
    """

    class TransformerEncoderSchema(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(
            required=True,
            description="Keep probability used in dropout layers.",
            example=0.5,
        )

        num_heads = fields.Int(missing=8, description="Number of Attention heads.")
        num_layers = fields.Int(missing=4, description="Number of encoder layers.")
        num_dff = fields.Int(
            missing=512, description="Number of feed-forward neurons in point-wise MLP."
        )
        d_model = fields.Int(missing=128, description="Depth of model.")
        max_pos_enc = fields.Int(
            missing=24, description="Maximum length of positional encoding."
        )
        layer_norm = fields.Bool(
            missing=True,
            description="Whether to apply layer normalization in the encoder.",
        )

        activation = fields.Str(
            missing="linear",
            description="Activation function used in final dense filters.",
        )

    def init_model(self):

        self.encoder = transformer_encoder_layers.Encoder(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            dff=self.config.num_dff,
            maximum_position_encoding=self.config.max_pos_enc,
            layer_norm=self.config.layer_norm,
        )

        self.dense = tf.keras.layers.Dense(
            units=self.config.n_classes, activation=self.config.activation
        )

    def build(self, inputs_shape):
        """Build Transformer encoder architecture
        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        seq_len = inputs_shape[1]

        self.net = tf.keras.Sequential(
            [
                self.encoder,
                self.dense,
                tf.keras.layers.MaxPool1D(pool_size=seq_len),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.backend.squeeze(x, axis=-2), name="squeeze"
                ),
                tf.keras.layers.Softmax(),
            ]
        )
        # Build the model, so we can print the summary
        self.net.build(inputs_shape)

        print_summary(self.net)

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training, mask)


########################################################################################################################################################


class PseTae(BaseTempnetsModel):
    """Implementation of the Pixel-Set encoder + Temporal Attention Encoder sequence classifier

    Code is based on the Pytorch implementation of V. Sainte Fare Garnot et al. https://github.com/VSainteuf/pytorch-psetae
    """

    class PseTaeSchema(BaseTempnetsModel._Schema):
        mlp1 = fields.List(
            fields.Int,
            missing=[10, 32, 64],
            description="Number of units for each layer in mlp1.",
        )
        pooling = fields.Str(
            missing="mean_std",
            description="Methods used for pooling. Seperated by underscore. (mean, std, max, min)",
        )
        mlp2 = fields.List(
            fields.Int,
            missing=[132, 128],
            description="Number of units for each layer in mlp2.",
        )

        num_heads = fields.Int(missing=4, description="Number of Attention heads.")
        num_dff = fields.Int(
            missing=32, description="Number of feed-forward neurons in point-wise MLP."
        )
        d_model = fields.Int(missing=None, description="Depth of model.")
        mlp3 = fields.List(
            fields.Int,
            missing=[512, 128, 128],
            description="Number of units for each layer in mlp3.",
        )
        dropout = fields.Float(
            missing=0.2, description="Dropout rate for attention encoder."
        )
        T = fields.Float(missing=1000, description="Number of features for attention.")
        len_max_seq = fields.Int(
            missing=24, description="Number of features for attention."
        )
        mlp4 = fields.List(
            fields.Int,
            missing=[128, 64, 32],
            description="Number of units for each layer in mlp4. ",
        )

    def init_model(self):
        # TODO: missing features from original PseTae:
        #   * spatial encoder extra features (hand-made)
        #   * spatial encoder masking

        self.spatial_encoder = pse_tae_layers.PixelSetEncoder(
            mlp1=self.config.mlp1, mlp2=self.config.mlp2, pooling=self.config.pooling
        )

        self.temporal_encoder = pse_tae_layers.TemporalAttentionEncoder(
            n_head=self.config.num_heads,
            d_k=self.config.num_dff,
            d_model=self.config.d_model,
            n_neurons=self.config.mlp3,
            dropout=self.config.dropout,
            T=self.config.T,
            len_max_seq=self.config.len_max_seq,
        )

        mlp4_layers = [
            pse_tae_layers.LinearLayer(out_dim) for out_dim in self.config.mlp4
        ]
        # Final layer (logits)
        mlp4_layers.append(
            pse_tae_layers.LinearLayer(1, batch_norm=False, activation="linear")
        )

        self.mlp4 = tf.keras.Sequential(mlp4_layers)

    def call(self, inputs, training=None, mask=None):

        out = self.spatial_encoder(inputs, training=training, mask=mask)
        out = self.temporal_encoder(out, training=training, mask=mask)
        out = self.mlp4(out, training=training, mask=mask)

        return out
