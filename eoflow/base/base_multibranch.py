from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation
from tsaug import TimeWarp, AddNoise, Pool, Convolve, Drift
from eoflow.models import data_augmentation as dat_aug


class BaseModelMultibranch(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def train_step_mb(self, train_ds, tabular_data = False):

        for x_dyn_batch_train, x_static_batch_train, y_batch_train in train_ds:
            with tf.GradientTape() as tape:
                if not tabular_data:
                    x_dyn_batch_train = [x_dyn_batch_train[..., i] for i in range(x_dyn_batch_train.shape[-1])]
                    y_preds, _ = self.call([x_dyn_batch_train, x_static_batch_train], training=True)
                else:
                    x_train = self._reshape_array(x_dyn_batch_train.numpy(), x_static_batch_train.numpy())
                    y_preds = self.call(x_train, training=True)

                cost = self.loss(y_batch_train, y_preds)
                cost += sum(self.losses)
                cost = tf.reduce_mean(cost)

            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_metric.update_state(cost)

            if self.config.ema:
                with tf.control_dependencies([opt_op]):
                    self.ema.apply(self.trainable_variables)

    def val_step_mb(self, val_ds, tabular_data = False):

        for x_dyn_batch_train, x_static_batch_train, y_batch_train in val_ds:
            if not tabular_data:
                x_dyn_batch_train = [x_dyn_batch_train[..., i] for i in range(x_dyn_batch_train.shape[-1])]
                y_preds, _ = self.call([x_dyn_batch_train, x_static_batch_train], training=False)
            else:
                x_train = self._reshape_array(x_dyn_batch_train.numpy(), x_static_batch_train.numpy())
                y_preds = self.call(x_train, training=False)

            cost = self.loss(y_batch_train, y_preds)

            self.loss_metric.update_state(tf.reduce_mean(cost))
            self.metric.update_state(y_batch_train.numpy().flatten(), y_preds.numpy().flatten())

    @staticmethod
    def _reshape_array(x_dyn, x_static):
        x_dyn_flatten = x_dyn.reshape(x_dyn.shape[0], x_dyn.shape[1] * x_dyn.shape[2])
        return np.concatenate([x_dyn_flatten, x_static], axis=1)

    def predict_mb(self, x_dyn_batch_train, x_static_batch_train, tabular_data):
        if not tabular_data:
            x_dyn_batch_train = [x_dyn_batch_train[..., i] for i in range(x_dyn_batch_train.shape[-1])]
            y_preds, _ = self.call([x_dyn_batch_train, x_static_batch_train], training=False)
        else:
            x_train = self._reshape_array(x_dyn_batch_train.numpy(), x_static_batch_train.numpy())
            y_preds = self.call(x_train, training=False)
            _ = None

        return y_preds, _

    def fit_mb(self, train_dataset, val_dataset, test_dataset,
               batch_size, num_epochs,
               model_directory, save_steps=10,
               patience=30, reduce_lr=False,  function=np.min,
               data_augmentation = False,
               tabular_data = False,
               **kwargs):

        global val_acc_result

        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_dyn_train, x_static_train, y_train = train_dataset
        x_dyn_val, x_static_val, y_val = val_dataset
        x_dyn_test, x_static_test, y_test = test_dataset

        if not tabular_data:
            shapes = [tf.zeros(list((x_dyn_train.shape[0], x_dyn_train.shape[1], 1)))
                      for i in range(x_dyn_train.shape[-1])]
            _ = self([shapes, tf.zeros(list(x_static_train.shape))])
        else:
            x_train = self._reshape_array(x_dyn_train, x_static_train)
            _ = self(tf.zeros(list(x_train.shape)))

        val_ds = tf.data.Dataset.from_tensor_slices((x_dyn_val, x_static_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_dyn_test, x_static_test,  y_test)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        for epoch in range(num_epochs + 1):
            x_dyn_train_, x_static_train_, y_train_ = shuffle(x_dyn_train, x_static_train, y_train)

            if data_augmentation and epoch>patience:
                my_augmenter = (
                        AddNoise(scale=0.05) @ 0.5
                        + Drift(max_drift=(0.025, 0.1)) @ 0.5
                        + Pool(size=1) @ 0.5
                )

                x_dyn_train_ = my_augmenter.augment(x_dyn_train)
                y_train_ = dat_aug.noisy_label(y_train_, proba=0.5, stdev=0.025)

            train_ds = tf.data.Dataset.from_tensor_slices((x_dyn_train_, x_static_train_, y_train_)).batch(batch_size)

            self.train_step_mb(train_ds, tabular_data)
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.val_step_mb(val_ds, tabular_data)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()
                self.val_step_mb(test_ds, tabular_data)
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

