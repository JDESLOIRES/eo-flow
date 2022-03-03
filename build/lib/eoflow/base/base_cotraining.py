import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from . import Configurable
from eoflow.base.base_callbacks import CustomReduceLRoP
from eoflow.models.data_augmentation import data_augmentation, timeshift, feature_noise
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining


class BaseModelCoTraining(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self,config_specs)

    def _clone_model(self, x_train):

        _ = self(tf.zeros(list(x_train.shape)))
        top_model = self.layers[0].layers[-2].output
        model = tf.keras.Model(inputs=self.layers[0].input,
                               outputs=Dense(units=1)(top_model))
        _ = model(tf.zeros(list(x_train.shape)))
        model.optimizer = tf.keras.optimizers.Adam(learning_rate = self.config['learning_rate'])
        model.loss_metric = tf.keras.metrics.Mean()
        return model

    def _init_model_pretrain(self, x, shift, noise):

        top_model = self.layers[0].layers[-2].output
        output_layer = Dense(units = x.shape[-1] * x.shape[-2])(top_model)
        output_layer_shift = Dense(units=1)(top_model)

        if noise and shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=[output_layer, output_layer_shift])
        elif noise :
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer)
        elif shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer_shift)

        print(model.summary())
        return model

    def pretraining(self,  x, pretraining_path,
                    batch_size=8, num_epochs=100,
                    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                    shift = 5, noise = 0.5, lambda_ = 1):

        _ = self(tf.zeros(list(x.shape)))
        n_layers = len(self.layers[0].layers)
        model = self._init_model_pretrain(x, shift, noise)

        for epoch in range(num_epochs):
            x = shuffle(x)
            ts_masking, mask = feature_noise(x, value=noise, proba=0.15)

            if shift:
                ts_masking, shift_arr, mask_sh = timeshift(ts_masking, shift, proba=0.15)
                shift_arr /= np.max(shift_arr)
            else:
                mask_sh, shift_arr = list(np.random.random(ts_masking.shape[0])), list(np.random.random(ts_masking.shape[0]).astype(int))

            train_ds = tf.data.Dataset.from_tensor_slices((x, ts_masking, mask, shift_arr, mask_sh))
            train_ds = train_ds.batch(batch_size)

            for x_batch, ts_masking_batch, mask_batch, shift_batch, mask_sh_batch in train_ds:  # tqdm
                n, t, d = x_batch.shape
                with tf.GradientTape() as tape:
                    if shift and noise:
                        x_preds, aux_preds = model.call(ts_masking_batch, training=True)
                        x_preds = tf.reshape(x_preds, (n, t, d))
                    elif shift:
                        aux_preds = model.call(x_batch, training=True)
                    elif noise:
                        x_preds = model.call(ts_masking_batch, training=True)
                        x_preds = tf.reshape(x_preds,(n, t, d))

                    if noise:
                        cost_noise = tf.reduce_mean(tf.multiply(loss(x_batch, x_preds), mask_batch))

                    if shift:
                        cost_shift = tf.reduce_mean(tf.multiply(loss(shift_batch, aux_preds), mask_sh_batch))

                    if shift and noise:
                        cost = cost_noise + lambda_ * cost_shift
                    elif shift:
                        cost = cost_shift
                    elif noise:
                        cost = cost_noise

                grads = tape.gradient(cost, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                self.loss_metric.update_state(cost)

            if epoch%5==0:
                print("Epoch {0}: Train loss {1}".format(str(epoch), str(self.loss_metric.result().numpy())))
            self.loss_metric.reset_states()

        for i in range(n_layers-1):
            self.layers[0].layers[i].set_weights(model.layers[i].get_weights())

        self.save_weights(os.path.join(pretraining_path, 'pretrained_model'))

    @staticmethod
    def _cost_cotraining(y_batch, y_preds, jcor, lambda_, forget_rate,
                         loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) ):
        cost = list(lambda_ * loss(y_batch, y_preds) + (1 - lambda_) * jcor)
        cost.sort()
        cost = tf.reduce_mean(cost[:(int(len(cost) * forget_rate))])
        return cost

    def _cotrain_step(self, model, x_batch, y_batch, y_preds_2, lambda_, forget_rate, div_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)):
        with tf.GradientTape() as tape:
            y_preds_1 = model.call(x_batch, training=True)
            jcor = (div_loss(y_preds_2, y_preds_1) + div_loss(y_preds_1, y_preds_2)) / 2
            cost_1 = self._cost_cotraining(y_batch, y_preds_1, jcor, lambda_, forget_rate)

        grads = tape.gradient(cost_1, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        model.loss_metric.update_state(cost_1)

        return y_preds_1, cost_1, jcor


    def cotraining(self, train_dataset, val_dataset,
                   num_epochs, pretraining_path,
                   batch_size=8, forget_rate = 0.9,
                   lambda_ = 0.5, patience =  50,
                   function=np.min):

        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = train_dataset
        y_train = y_train.astype('float32')
        x_val, y_val = val_dataset
        y_val = y_val.astype('float32')
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        if pretraining_path:
            self._init_weights_pretrained(pretraining_path)
        model = self._clone_model(x_train)

        forget_rate_ = 1
        lambd = 1

        for epoch in range(num_epochs + 1):

            x_train_, y_train_ = shuffle(x_train, y_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(batch_size)
            if epoch == patience:
                forget_rate_ = forget_rate
                lambd = lambda_
            for x_batch, y_batch in train_ds:  # tqdm
                with tf.GradientTape() as tape:
                    y_preds_2 = self.call(x_batch, training=True)
                    y_preds_1, cost_1, jcor = self._cotrain_step(model, x_batch, y_batch,
                                                                 y_preds_2,
                                                                 lambd, forget_rate_)

                    cost_2 = self._cost_cotraining(y_batch, y_preds_2, jcor,
                                                   lambd, forget_rate)
                    self.loss_metric.update_state(cost_2)

                grads = tape.gradient(cost_2, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            model.loss_metric.reset_states()

            if epoch%5==0:
                self.val_step(val_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()
                print(
                    "Epoch {0}: Train loss {1}, Val acc {2}".format(
                        str(epoch), str(loss_epoch), str(round(val_acc_result, 4)),
                    ))
