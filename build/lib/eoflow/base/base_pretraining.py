import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle

import os

import tensorflow as tf

from . import Configurable
from eoflow.base.base_callbacks import CustomReduceLRoP
from eoflow.models.data_augmentation import timeshift, feature_noise, noisy_label
from keras.models import Sequential
from keras.layers import Dense


class BaseModelPretraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.init_model()

    def init_model(self):
        """ Called on __init__. Keras self initialization. Create self here if does not require the inputs shape """
        pass

    def build(self, inputs_shape):
        """ Keras method. Called once to build the self. Build the self here if the input shape is required. """
        pass

    def call(self, inputs, training=False):
        pass

    def prepare(self,
                optimizer=None, loss=None, metrics=None,
                epoch_loss_metric=None,
                epoch_val_metric=None,
                reduce_lr=False,
                **kwargs):
        """ Prepares the self for training and evaluation. This method should create the
        optimizer, loss and metric functions and call the compile method of the self. The self
        should provide the defaults for the optimizer, loss and metrics, which can be overriden
        with custom arguments. """

        raise NotImplementedError

    @tf.function
    def train_step(self,
                   train_ds,
                   mask = None):
        # pb_i = Progbar(len(list(train_ds)), stateful_metrics='acc')
        for x_batch_train, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:
                y_preds = self.call(x_batch_train,
                                    training=True)

                cost = self.loss(y_batch_train, y_preds)
                if mask: cost *= mask

            print(y_preds)
            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_metric.update_state(cost)
            with tf.control_dependencies([opt_op]):
                self.ema.apply(self.trainable_variables)


    def _define_auxilliary_model(self, n_dim):
        model = Sequential()
        for layer in self.layers[:-1]:
            model.add(layer)
        model.add(Dense(units=n_dim, activation='linear'))

        return model

    def pretraining(self, x_train, model_directory, batch_size=8, num_epochs=100):

        train_loss = []
        model = self._define_auxilliary_model(x_train.shape[1])

        for epoch in range(num_epochs):
            x_train = shuffle(x_train)
            ts_masking, mask = feature_noise(x_train, value=0.5, proba=0.15)
            print(mask.shape)
            train_ds = tf.data.Dataset.from_tensor_slices((ts_masking, mask))
            train_ds = train_ds.batch(batch_size)
            self.train_step(train_ds,mask)
            loss_epoch = self.loss_metric.result().numpy()

            if epoch%5==0:
                print("Epoch {0}: Train loss {1}".format(str(epoch), str(loss_epoch)))

            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

        self.save_weights(os.path.join(model_directory, 'pretrained_model'))

    def _init_weights_pretrained(self, model_directory, n_freeze =3):


        self.load_weights(os.path.join(model_directory, 'pretrained_model'))

        #Freeze the first n layers
        for i in range(len(self.layers) - 1):
            self.layers[i].set_weights(self.layers[i].get_weights())
            if i<=n_freeze:
                self.layers[i].trainable = False

    def _set_trainable(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].trainable = True



