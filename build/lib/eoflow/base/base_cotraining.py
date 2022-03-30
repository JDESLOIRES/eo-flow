from sklearn.utils import shuffle
import numpy as np
import os

import tensorflow as tf

from eoflow.models.data_augmentation import timeshift, feature_noise
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

    @staticmethod
    def _get_lambda(factor, num_epochs, epoch):
        p = float(epoch) / (num_epochs)
        return tf.constant(factor * (2.0 / (1.0 + np.exp(-10.0 * p, dtype=np.float32)) - 1.0),
                           dtype='float32')

    def _init_model_pretrain(self, x, shift, noise):

        embedding = self.layers[0].layers[self.config.nb_conv_stacks * 4 + 1].output
        net_mean_emb = self._fcn_layer(embedding)
        for i in range(1, self.config.nb_fc_stacks):
            net_mean_emb = self._fcn_layer(net_mean_emb, i)
        output_layer = tf.keras.layers.Dense(units=x.shape[-2])(net_mean_emb)
        output_layer_shift = tf.keras.layers.Dense(units=1)(net_mean_emb)

        if noise and shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=[output_layer, output_layer_shift])
        elif noise:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer)
        elif shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer_shift)

        return model



    def pretraining(self,  x, pretraining_path,
                    batch_size=8, num_epochs=100,
                    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                    aux_vector_task=None,
                    shift = 5, noise = 0.5, lambda_ = 1, proba = 0.15):

        _ = self(tf.zeros(list(x.shape)))
        model = self._init_model_pretrain(x, shift, noise)

        for epoch in range(num_epochs):
            if aux_vector_task:
                x, aux_vector_sh = shuffle((x, aux_vector_task))
            else:
                x = shuffle(x)
                aux_vector_sh = None

            ts_masking, mask = feature_noise(x, value=noise, proba=0.15)
            noise_added = x[...,0] - ts_masking[...,0]

            if shift:
                ts_masking, shift_arr, mask_sh = timeshift(ts_masking, shift, proba=proba)
                shift_arr /= np.max(shift_arr)
            else:
                mask_sh, shift_arr = list(np.zeros(ts_masking.shape[0])), list(np.zeros(ts_masking.shape[0]).astype(int))

            train_ds = tf.data.Dataset.from_tensor_slices((x, ts_masking, noise_added, mask, shift_arr, mask_sh, aux_vector_sh))
            train_ds = train_ds.batch(batch_size)

            for x_batch, ts_masking_batch, noise_added_batch, mask_batch, shift_batch, mask_sh_batch, aux_vec in train_ds:  # tqdm
                n, t, d = x_batch.shape
                with tf.GradientTape() as tape:
                    if shift and noise:
                        x_preds, aux_preds = model.call(ts_masking_batch, training=True)
                        x_preds = tf.reshape(x_preds, (n, t))
                        cost = tf.reduce_mean(tf.multiply(tf.math.squared_difference(tf.cast(noise_added_batch, 'float32') , x_preds), mask_batch)) +\
                               lambda_ * tf.reduce_mean(tf.multiply(loss(shift_batch, aux_preds), mask_sh_batch))
                    elif shift:
                        aux_preds = model.call(x_batch, training=True)
                        cost = tf.reduce_mean(
                            tf.multiply(loss(shift_batch, aux_preds), mask_sh_batch))
                    elif noise:
                        x_preds = model.call(ts_masking_batch, training=True)
                        x_preds = tf.reshape(x_preds, noise_added_batch.shape)
                        cost = tf.reduce_mean(
                            tf.multiply(tf.math.squared_difference(tf.cast(noise_added_batch, 'float32'), x_preds),
                                        mask_batch))

                grads = tape.gradient(cost, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                self.loss_metric.update_state(cost)

            if epoch%5==0:
                print("Epoch {0}: Train loss {1}".format(str(epoch), str(self.loss_metric.result().numpy())))
            self.loss_metric.reset_states()

        model.save_weights(os.path.join(pretraining_path, 'pretrained_model'))


    def _init_model_pretrain(self, x, shift, noise):

        embedding = self.layers[0].layers[self.config.nb_conv_stacks * 4 + 1].output
        net_mean_emb = self._fcn_layer(embedding)
        for i in range(1, self.config.nb_fc_stacks):
            net_mean_emb = self._fcn_layer(net_mean_emb, i)
        output_layer = tf.keras.layers.Dense(units=x.shape[-2])(net_mean_emb)
        output_layer_shift = tf.keras.layers.Dense(units=1)(net_mean_emb)

        if noise and shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=[output_layer, output_layer_shift])
        elif noise:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer)
        elif shift:
            model = tf.keras.Model(inputs=self.layers[0].input, outputs=output_layer_shift)

        return model


    def _init_weights_pretrained(self, x,  shift, noise, model_directory = './'):

        self.build(inputs_shape= x.shape)
        _ = self(tf.zeros([k for k in x.shape]))
        model = self._init_model_pretrain(x, shift, noise)
        _ = model(tf.zeros([k for k in x.shape]))

        model.load_weights(os.path.join(model_directory, 'pretrained_model'))

        # Load weights encoder
        for i in range(self.config.nb_conv_stacks * 4 + 2):
            self.layers[0].layers[i].set_weights(model.layers[i].get_weights())


    def multitask_pretraining(self,
                              train_dataset,
                              val_dataset,
                              test_dataset,
                              pretraining_path = None,
                              additional_target = None,
                              batch_size=8, num_epochs=500,
                              patience = 100,
                              shift = 5, lambda_ = 1):

        x_train, y_train = train_dataset
        y_train = y_train.astype('float32')
        x_val, y_val = val_dataset
        y_val = y_val.astype('float32')
        x_test, y_test = test_dataset
        y_test = y_test.astype('float32')

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        _, _ = self(tf.zeros(list(x_train.shape)))
        wait = 0

        for epoch in range(num_epochs):
            x_train_, y_train_ = shuffle(x_train, y_train)
            if not additional_target:
                ts_masking, shift_arr, mask_sh = timeshift(x_train_, shift, proba=0.15)
            else:
                ts_masking,  shift_arr, mask_sh, = x_train_, additional_target, np.zeros((x_train_.shape[0]), dtype='float32') + 1
            shift_arr = (shift_arr - np.min(shift_arr))/(np.max(shift_arr) - np.min(shift_arr))

            train_ds = tf.data.Dataset.from_tensor_slices((ts_masking, y_train_, shift_arr, mask_sh))
            train_ds = train_ds.batch(batch_size)

            for ts_masking_batch, y_batch_train, shift_batch, mask_sh_batch in train_ds:  # tqdm
                with tf.GradientTape() as tape:
                    y_preds, aux_preds = self.call(ts_masking_batch, training=True)
                    cost_supervised = tf.reduce_mean(self.loss(y_batch_train, y_preds))
                    cost_unsupervised = tf.reduce_mean(tf.multiply(self.loss(shift_batch, aux_preds), mask_sh_batch))
                    cost = cost_supervised + lambda_ * cost_unsupervised

                grads = tape.gradient(cost, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.loss_metric.update_state(cost)

            loss_epoch = self.loss_metric.result().numpy()
            self.loss_metric.reset_states()

            if epoch%5==0:
                wait +=1
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()
                ####################################################
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
                reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

        self.save_weights(os.path.join(pretraining_path, 'multioutput_model'))

    @staticmethod
    def _cost_cotraining(y_batch, y_preds, jcor, lambda_, n_forget,
                         loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) ):
        cost = tf.add(loss(y_batch, y_preds), tf.multiply(jcor, lambda_))
        cost = tf.sort(cost, direction='DESCENDING')
        cost = cost[n_forget:]
        return tf.reduce_mean(cost)

    def _cotrain_step(self, model, x_batch, y_batch, y_preds_2, lambda_, n_forget,
                      div_loss =  tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)):

        with tf.GradientTape() as tape:
            y_preds_1 = model.call(x_batch, training=True)
            jcor = (div_loss(y_preds_2, y_preds_1) + div_loss(y_preds_1, y_preds_2)) / 2
            cost_1 = self._cost_cotraining(y_batch, y_preds_1, jcor, lambda_, n_forget)

        grads = tape.gradient(cost_1, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        model.loss_metric.update_state(cost_1)

        return y_preds_1, cost_1, jcor


    def cotraining(self,
                   train_dataset, val_dataset,
                   num_epochs, pretraining_path,
                   model_directory,
                   batch_size=8, forget = 1,
                   factor = 0.5, patience =  50,
                   reduce_lr = True,
                   function=np.min):

        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = train_dataset
        y_train = y_train.astype('float32')
        x_val, y_val = val_dataset
        y_val = y_val.astype('float32')
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience//4, factor=0.5)
        wait = 0

        if pretraining_path:
            self._init_weights_pretrained(pretraining_path)
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))

        model = self._clone_model(x_train)

        forget_ = 0

        for epoch in range(num_epochs + 1):

            x_train_, y_train_ = shuffle(x_train, y_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(batch_size)
            lambda_ = self._get_lambda(factor, num_epochs, epoch)

            if epoch == patience:
                forget_ = forget
            for x_batch, y_batch in train_ds:  # tqdm
                with tf.GradientTape() as tape:
                    y_preds_2, _ = self.call(x_batch, training=True)

                    y_preds_1, cost_1, jcor = self._cotrain_step(model, x_batch, y_batch,
                                                                 y_preds_2,
                                                                 lambda_, forget_)
                    cost_2 = self._cost_cotraining(y_batch, y_preds_2, jcor,
                                                   lambda_, forget)

                    self.loss_metric.update_state(cost_2)

                grads = tape.gradient(cost_2, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            model.loss_metric.reset_states()

            if epoch%5==0:
                wait += 1
                self.val_step(val_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch), str(loss_epoch),  str(test_loss_epoch),  str(round(val_acc_result, 4)),
                    ))
                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_pretrain_model'))
        model.save_weights(os.path.join(model_directory, 'last_model'))