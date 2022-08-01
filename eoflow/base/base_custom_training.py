from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from . import Configurable
from eoflow.models.callbacks import CustomReduceLRoP
from eoflow.models.data_augmentation import data_augmentation
from tsaug import TimeWarp, AddNoise, Pool, Convolve, Drift


class BaseModelCustomTraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
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
                   n_forget = 0,
                   size_batch = 8,
                   lambda_ = 1):

        for x_batch_train, y_batch_train, y_batch_aux in train_ds:
            if self.config.multibranch:
                x_batch_train = [x_batch_train[...,i] for i in range(x_batch_train.shape[-1])]
            with tf.GradientTape() as tape:
                if self.config.multioutput:
                    if np.any(['conv' in x.split('_') for x in self.config.keys()]):
                        y_preds, y_aux, _ = self.call(x_batch_train, training=True)
                    else:
                        y_preds, y_aux = self.call(x_batch_train, training=True)
                    y_true, y_aux_true = y_batch_train, y_batch_aux
                    cost = tf.reduce_mean(self.loss(y_true, y_preds)) \
                           + lambda_ * tf.reduce_mean(self.loss(y_aux_true, y_aux))
                elif self.config.loss in ['gaussian', 'laplacian']:
                    y_preds, sigma_, _ = self.call(x_batch_train, training=True)
                    cost = self.loss(y_preds, sigma_, y_batch_train)
                else:
                    if np.any(['conv' in x.split('_') for x in self.config.keys()]):
                        y_preds, _ = self.call(x_batch_train, training=True)
                    else:
                        y_preds = self.call(x_batch_train, training=True)

                    cost = self.loss(y_batch_train, y_preds)
                    cost += sum(self.losses)

                if n_forget and tf.greater(tf.shape(x_batch_train)[0], size_batch -1):
                    cost = tf.sort(cost, direction='DESCENDING')
                    cost = cost[n_forget:]

                cost = tf.reduce_mean(cost)

            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_metric.update_state(cost)

            if self.config.ema:
                with tf.control_dependencies([opt_op]):
                    self.ema.apply(self.trainable_variables)

    #@tf.function
    def val_step(self, val_ds, lambda_ = 1):
        for x_batch_train, y_batch_train, y_batch_aux in val_ds:
            if self.config.multibranch:
                x_batch_train = [x_batch_train[...,i] for i in range(x_batch_train.shape[-1])]
            if self.config.multioutput:
                if np.any(['conv' in x.split('_') for x in self.config.keys()]):
                    y_preds, y_aux, _ = self.call(x_batch_train, training=False)
                else:
                    y_preds, y_aux = self.call(x_batch_train, training=False)
                y_true, y_aux_true = y_batch_train, y_batch_aux
                cost = self.loss(y_true, y_preds) \
                       + lambda_ * self.loss(y_aux_true, y_aux)
            elif self.config.loss in ['gaussian', 'laplacian']:
                y_preds, sigma_, _ = self.call(x_batch_train, training=False)
                cost = self.loss(y_preds, sigma_, y_batch_train)
            else:
                if np.any(['conv' in x.split('_') for x in self.config.keys()]):
                    y_preds, _ = self.call(x_batch_train, training=False)
                else:
                    y_preds = self.call(x_batch_train, training=False)
                cost = self.loss(y_batch_train, y_preds)

            self.loss_metric.update_state(tf.reduce_mean(cost))
            self.metric.update_state(y_batch_train.numpy().flatten(), y_preds.numpy().flatten())

    def _reduce_lr_on_plateau(self, patience=30,
                              factor=0.1,
                              reduce_lin=False):

        return CustomReduceLRoP(patience=patience,
                                factor=factor,
                                verbose=1,
                                optim_lr=self.optimizer.learning_rate,
                                reduce_lin=reduce_lin)

    def _set_trainable(self, bool = True):
        n_layers = len(self.layers[0].layers)
        fc_sublayers = 4 if self.config.fc_activation else 3
        n_outputs = self.config.nb_fc_stacks * fc_sublayers + 1

        for i in range(n_layers-n_outputs):
            self.layers[0].layers[i].trainable = bool


    def fit(self,
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            num_epochs,
            model_directory,
            save_steps=10,
            patience = 30,
            lambda_ = 1,
            reduce_lr = False,
            forget = 0,
            function=np.min):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))
        x_train, y_train = train_dataset
        x_val, y_val = val_dataset
        x_test, y_test = test_dataset

        if self.config.multioutput:
            y_train, y_train_aux = y_train
            y_val, y_val_aux = y_val
            y_test, y_test_aux = y_test
        else:
            y_train_aux, y_val_aux, y_test_aux = np.zeros(y_train.shape[0]), \
                                           np.zeros(y_val.shape[0]), \
                                           np.zeros(y_test.shape[0])

        if not self.config.multibranch:
            _ = self(tf.zeros(list(x_train.shape)))
        else:
            x_dyn_train, x_dyn_val, x_dyn_test = x_train[0],x_val[0],x_test[0]
            shapes = [tf.zeros(list((x_dyn_train.shape[0], x_dyn_train.shape[1], 1)))
                      for i in range(x_dyn_train.shape[-1])]
            if len(x_train)>1:
                x_static_train, x_static_val, x_static_test = x_train[1], x_val[1], x_test[1]
                _ = self([shapes, tf.zeros(list(x_static_train.shape))])
            else:
                _ = self([shapes])

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val, y_val_aux)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test, y_test_aux)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience//4, factor=0.5)
        wait = 0
        n_forget = 0

        for epoch in range(num_epochs + 1):
            x_train_, y_train_, y_train_aux_ = shuffle(x_train, y_train, y_train_aux)
            if patience and epoch >= patience:
                if self.config.finetuning:
                    for i in range(len(self.layers[0].layers)):
                        self.layers[0].layers[i].trainable = True
                if forget:
                    n_forget = forget

            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_, y_train_aux_)).batch(batch_size)

            self.train_step(train_ds, n_forget, batch_size, lambda_)
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait +=1
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()
                self.val_step(test_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}, Test loss {4}, Test acc {5}".format(
                        str(epoch), str(loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                        str(round(test_loss_epoch, 4)), str(round(test_acc_result, 4)),
                    ))

                if (function is np.min and val_loss_epoch < function(val_loss)
                        or function is np.max and val_loss_epoch > function(val_loss)):
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.save_weights(os.path.join(model_directory, 'best_model'))

                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

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

