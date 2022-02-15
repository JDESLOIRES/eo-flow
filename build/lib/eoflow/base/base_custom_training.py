import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from . import Configurable
from eoflow.base.base_callbacks import CustomReduceLRoP
from eoflow.models.data_augmentation import timeshift, feature_noise, noisy_label
from tensorflow.keras.layers import Dense, RepeatVector


class BaseModelCustomTraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)

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
                   train_ds):
        # pb_i = Progbar(len(list(train_ds)), stateful_metrics='acc')
        for x_batch_train, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:
                y_preds = self.call(x_batch_train,
                                    training=True)

                cost = self.loss(y_batch_train, y_preds)

            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_metric.update_state(cost)
            with tf.control_dependencies([opt_op]):
                self.ema.apply(self.trainable_variables)

    # Function to run the validation step.
    @tf.function
    def val_step(self, val_ds):
        for x, y in val_ds:
            y_preds = self.call(x, training=False)
            cost = self.loss(y, y_preds)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y + 1, y_preds + 1)

    @staticmethod
    def _data_augmentation(x_train_, y_train_, shift_step, feat_noise, sdev_label):
        if shift_step:
            x_train_, _ = timeshift(x_train_, shift_step)
        if feat_noise:
            x_train_, _ = feature_noise(x_train_, feat_noise)
        if sdev_label:
            y_train_ = noisy_label(y_train_, sdev_label)
        return x_train_, y_train_

    def _reduce_lr_on_plateau(self, patience=30,
                              factor=0.1,
                              reduce_lin=False):

        return CustomReduceLRoP(patience=patience,
                                factor=factor,
                                verbose=1,
                                optim_lr=self.optimizer.learning_rate,
                                reduce_lin=reduce_lin)


    def pretraining(self,  x, model_directory, batch_size=8, num_epochs=100, shift = 0):
        _ = self(tf.zeros([k for k in x.shape]))
        n_layers = len(self.layers[0].layers)

        top_model = self.layers[0].layers[-2].output
        output_layer = Dense(units = x.shape[-1])(top_model)
        output_layer = RepeatVector(n=x.shape[1])(output_layer)
        model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer)
        print(model.summary())

        for epoch in range(num_epochs):
            x = shuffle(x)
            if shift:
                x, _ = timeshift(x, shift)
            ts_masking, mask = feature_noise(x, value=0.5, proba=0.15)

            train_ds = tf.data.Dataset.from_tensor_slices((x, ts_masking, mask))
            train_ds = train_ds.batch(batch_size)

            for x_batch, ts_masking_batch, mask_batch in train_ds:  # tqdm
                with tf.GradientTape() as tape:
                    x_preds = model.call(ts_masking_batch, training=True)
                    cost = self.loss(x_batch, x_preds)
                    cost = tf.reduce_mean(cost* mask_batch)

                grads = tape.gradient(cost, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                self.loss_metric.update_state(cost)

            if epoch%5==0: print("Epoch {0}: Train loss {1}".format(str(epoch), str(cost)))

            self.loss_metric.reset_states()

        for i in range(n_layers-1):
            self.layers[0].layers[i].set_weights(model.layers[i].get_weights())

        self.save_weights(os.path.join(model_directory, 'pretrained_model'))

    def _set_trainable(self, bool = True):
        n_layers = len(self.layers[0].layers)
        n_outputs = self.config.nb_fc_stacks * 4 + 1

        for i in range(n_layers-n_outputs):
            self.layers[0].layers[i].trainable = bool


    def _init_weights_pretrained(self, model_directory):
        self.load_weights(os.path.join(model_directory, 'pretrained_model'))
        self._set_trainable(False)


    def _allow_training(self):
        self._set_trainable(True)

    def fit(self,
            train_dataset,
            val_dataset,
            batch_size,
            num_epochs,
            model_directory,
            save_steps=10,
            shift_step=0,
            feat_noise=0,
            sdev_label=0,
            reduce_lr = False,
            pretraining = False,
            patience = 30,
            function=np.min):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = train_dataset
        y_train = y_train.astype('float32')
        x_val, y_val = val_dataset
        y_val = y_val.astype('float32')

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience*2)
        wait = 0

        _ = self(tf.zeros([k for k in x_train.shape]))

        if pretraining:
            self._init_weights_pretrained(model_directory)
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))

        for epoch in range(num_epochs + 1):

            x_train_, y_train_ = shuffle(x_train, y_train)
            x_train_, y_train_ = self._data_augmentation(x_train_, y_train_,
                                                         shift_step, feat_noise, sdev_label)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(batch_size)

            self.train_step(train_ds)

            # End epoch
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch == patience and pretraining:
                self._allow_training()

            if epoch % save_steps == 0:
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch), str(loss_epoch), str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                    ))

                wait += 1

                if (
                        function is np.min
                        and val_loss_epoch < function(val_loss)
                        or function is np.max
                        and val_loss_epoch > function(val_loss)
                ):
                    wait = 0
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.save_weights(os.path.join(model_directory, 'best_model'))

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)
                self.loss_metric.reset_states()
                self.metric.reset_states()

                #if wait >= patience: break

            if reduce_lr:
                reduce_rl_plateau.on_epoch_end(epoch, val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))

        # History of the training
        losses = dict(train_loss_results=train_loss,
                      val_loss_results=val_acc
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)


    def train_and_evaluate(self,
                           train_dataset,
                           val_dataset,
                           num_epochs,
                           iterations_per_epoch,
                           model_directory,
                           **kwargs):

        return self.fit(train_dataset,
                        val_dataset,
                        num_epochs=num_epochs,
                        model_directory=model_directory,
                        save_steps=iterations_per_epoch,
                        **kwargs)

