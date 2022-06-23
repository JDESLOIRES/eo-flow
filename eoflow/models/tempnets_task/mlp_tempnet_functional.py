import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from eoflow.models.tempnets_task.tempnets_base import BaseTempnetsModel, BaseCustomTempnetsModel, BaseModelAdapt, \
    BaseModelAdaptV2, BaseModelAdaptV3
from marshmallow import fields
from marshmallow.validate import OneOf

tf.keras.backend.set_floatx('float32')


class FC(Layer):
    def __init__(self, nb_units, drop_val, **kwargs):
        super(FC, self).__init__(**kwargs)
        self.dense = Dense(nb_units, kernel_initializer='he_normal')
        self.batch_norm = BatchNormalization(axis=-1)
        self.dropout = Dropout(drop_val)
        self.act = Activation('relu')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        dense = self.dense(inputs)
        batch_norm = self.batch_norm(dense, training=training)
        drop = self.dropout(batch_norm, training=training)
        return self.act(drop)


class Task(Layer):
    def __init__(self, fc_units, nb_class, output_activation, drop_val=0.5, name=None, **kwargs):
        super(Task, self).__init__(**kwargs, name=name)

        self.dense_1 = FC(fc_units, drop_val)
        self.dense_2 = FC(fc_units // 2, drop_val)
        self.output_ = Dense(nb_class, activation=output_activation,
                             kernel_initializer='he_normal')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        dense_1 = self.dense_1(inputs, training=training)
        dense_2 = self.dense_2(dense_1, training=training)
        return self.output_(dense_2)


@tf.custom_gradient
def gradient_reverse(x, lambda_=1.0):
    y = tf.identity(x)

    def custom_grad(dy):
        return lambda_ * -dy, None

    return y, custom_grad


class GradReverse(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, lambda_=1.0):
        return gradient_reverse(x, lambda_)


class TempDANN(BaseCustomTempnetsModel, BaseModelAdaptV3):
    class TempDANNSchema(BaseCustomTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout tf.keras.layers.',
                                 example=0.5)
        nb_conv_filters = fields.Int(missing=64, description='Number of convolutional filters.')
        factor = fields.Float(missing=1.0, description='Keep probability used in dropout tf.keras.layers.')
        nb_fc_neurons = fields.Int(missing=64, description='Number of convolutional filters.')
        adaptative = fields.Bool(missing=True, description='Adaptative lambda for DANN')
        ema = fields.Bool(missing=True, description='Apply EMA')

    def __init__(self, config, **kwargs):
        BaseCustomTempnetsModel.__init__(self, config, **kwargs)
        drop_out = 1 - self.config.keep_prob_conv
        self.encoder = TempCNNEncoder(filters=self.config.nb_conv_filters,
                                      drop_val=drop_out)

        self.task = Task(fc_units=self.config.nb_fc_neurons, nb_class=1,
                         drop_val=drop_out,
                         output_activation='linear',
                         name='Task')

        self.grl = GradReverse()

        self.discriminator = Task(fc_units=self.config.nb_fc_neurons, nb_class=2,
                                  drop_val=drop_out,
                                  output_activation='softmax',
                                  name='Discriminator')

    @tf.function
    def call(self, inputs, training=False, lambda_=1.0):
        enc_out = self.encoder(inputs, training=training)
        grl = self.grl(enc_out, lambda_)
        return self.task(enc_out, training=training), self.discriminator(grl, training=training)

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
