from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation


class BaseModelAdapt(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def _init_models(self, x):

        _ = self(tf.zeros(list(x.shape)))

        inputs = self.layers[0].input
        if np.any(['conv' in x.split('_') for x in self.config.keys()]):
            encode = self.layers[0].layers[self.config.nb_conv_stacks * 4 + 1].output
            if self.config.loss in ['gaussian', 'laplacian']:
                dense_layers = self.layers[0].layers[-2 * 2 + 1].output
                output_discriminator = Dense(1, activation='sigmoid', name='Discriminator')(dense_layers)
                output_task = [self.layers[0].layers[-1].output, self.layers[0].layers[-2].output]
            else:
                dense_layers = self.layers[0].layers[-2].output
                output_discriminator = Dense(1, activation='sigmoid', name='Discriminator')(dense_layers)
                output_task = self.layers[0].layers[-1].output

        else:
            encode = self.layers[0].layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4 ].output
            dense_layers = self.layers[0].layers[-2].output
            output_discriminator = Dense(1, activation='sigmoid', name='Discriminator')(dense_layers)
            output_task = self.layers[0].layers[-1].output

        return inputs, encode, dense_layers, output_discriminator, output_task

    def _assign_properties(self, model):
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.loss_metric = tf.keras.metrics.Mean()
        return model

    def _get_encoder(self, x):
        inputs, encode, _, _, _ = self._init_models(x)
        encoder = tf.keras.Model(inputs=inputs, outputs=encode)
        encoder.summary()
        self.encoder = self._assign_properties(encoder)

    def _get_discriminator(self, x):
        _, encode, _, output_discriminator, _ = self._init_models(x)

        discriminator = tf.keras.Model(inputs=encode, outputs=output_discriminator)
        discriminator.summary()
        self.discriminator = self._assign_properties(discriminator)
        self.discriminator.loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def _get_task(self, x):
        _, encode, _, _, output_task = self._init_models(x)

        task = tf.keras.Model(inputs=encode, outputs=output_task)
        task.summary()
        self.task = self._assign_properties(task)

    @staticmethod
    def _init_dataset_training(x_s, y_s, x_t, y_t, batch_size):
        x_s = x_s.astype('float32')
        x_t = x_t.astype('float32')
        y_s = y_s.astype('float32')
        y_t = y_t.astype('float32')
        return tf.data.Dataset.from_tensor_slices((x_s, y_s, x_t, y_t)).batch(batch_size)

    @staticmethod
    def _assign_missing_obs(x_s, x_t, y_t):
        if x_t.shape[0] < x_s.shape[0]:
            x_t, y_t = np.repeat(x_t, x_s.shape[0] // x_t.shape[0], axis=0), \
                       np.repeat(y_t, x_s.shape[0] // x_t.shape[0])
            num_missing = x_s.shape[0] - x_t.shape[0]
            additional_obs = np.random.choice(x_t.shape[0], size=num_missing, replace=False)
            x_t, y_t = np.concatenate([x_t, x_t[additional_obs,]], axis=0), \
                       np.concatenate([y_t, y_t[additional_obs,]], axis=0)
        else:
            random_obs = np.random.choice(x_t.shape[0],
                                          size=x_s.shape[0],
                                          replace=False)
            x_t = x_t[random_obs,]

        return x_t, y_t

    @staticmethod
    def _get_lambda(factor, num_epochs, epoch):
        p = float(epoch) / (num_epochs)
        return tf.constant(factor * (2.0 / (1.0 + np.exp(-10.0 * p, dtype=np.float32)) - 1.0),
                           dtype='float32')

    @tf.function
    def trainstep_dann(self,
                       train_ds,
                       lambda_=1.0):

        for Xs, ys, Xt, yt in train_ds:
            with tf.GradientTape() as gradients_task, tf.GradientTape() as enc_tape, \
                    tf.GradientTape() as disc_tape:
                # Forward pass
                Xs_enc = self.encoder(Xs, training=True)
                if self.config.loss in ['gaussian', 'laplacian']:
                    ys_pred, sigma_s = self.task.call(Xs_enc, training=True)
                    ys_pred, sigma_s = tf.reshape(ys_pred, tf.shape(ys)), tf.reshape(sigma_s, tf.shape(ys))
                    cost = self.loss(ys_pred, sigma_s, ys)
                else:
                    ys_pred = self.task.call(Xs_enc, training=True)
                    ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                    cost = self.loss(ys, ys_pred)

                Xt_enc = self.encoder.call(Xt, training=True)

                ys_disc = self.discriminator.call(Xs_enc, training=True)
                yt_disc = self.discriminator.call(Xt_enc, training=True)

                # Compute the discriminator loss values
                disc_loss = 0.5 * (self.discriminator.loss(tf.ones_like(ys_disc), ys_disc) +
                                   self.discriminator.loss(tf.zeros_like(yt_disc), yt_disc))
                loss_dann = 0.5 * (self.discriminator.loss(tf.zeros_like(ys_disc), ys_disc) +
                                   self.discriminator.loss(tf.ones_like(yt_disc), yt_disc))

                # Compute the loss value
                enc_loss = cost + tf.multiply(loss_dann, lambda_)

                enc_loss = tf.reduce_mean(enc_loss)
                disc_loss = tf.reduce_mean(disc_loss)
                cost = tf.reduce_mean(cost)

                # https://stackoverflow.com/questions/56693863/why-does-model-losses-return-regularization-losses
                cost += sum(self.task.losses)
                disc_loss += sum(self.discriminator.losses)
                enc_loss += sum(self.encoder.losses)

            gradients_task = gradients_task.gradient(cost, self.task.trainable_variables)
            gradients_enc = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
            gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update weights
            opt_op_task = self.task.optimizer.apply_gradients(
                zip(gradients_task, self.task.trainable_variables))
            opt_op_enc = self.encoder.optimizer.apply_gradients(
                zip(gradients_enc, self.encoder.trainable_variables))
            opt_op_disc = self.discriminator.optimizer.apply_gradients(
                zip(gradients_disc, self.discriminator.trainable_variables))

            if self.config.ema:
                with tf.control_dependencies([opt_op_disc]):
                    self.ema.apply(self.discriminator.trainable_variables)
                with tf.control_dependencies([opt_op_enc]):
                    self.ema.apply(self.encoder.trainable_variables)
                with tf.control_dependencies([opt_op_task]):
                    self.ema.apply(self.task.trainable_variables)

            self.task.loss_metric.update_state(cost)
            self.encoder.loss_metric.update_state(enc_loss)
            self.discriminator.loss_metric.update_state(disc_loss)

    @tf.function
    def valstep_dann(self, val_ds):
        for x_batch, y_batch in val_ds:
            x_enc = self.encoder.call(x_batch, training=False)
            if self.config.loss in ['gaussian', 'laplacian']:
                y_pred, sigma_ = self.task.call(x_enc, training=False)
                y_pred, sigma_ = tf.reshape(y_pred, tf.shape(y_batch)), tf.reshape(sigma_, tf.shape(y_batch))
                cost = self.loss(y_pred, sigma_, y_batch)
            else:
                y_pred = self.task.call(x_enc, training=False)
                y_pred = tf.reshape(y_pred, tf.shape(y_batch))
                cost = self.loss(y_batch, y_pred)

            cost = tf.reduce_mean(cost)
            self.task.loss_metric.update_state(cost)
            self.metric.update_state(y_batch, y_pred)


    def fit_dann(self,
                 src_dataset,
                 val_dataset,
                 trgt_dataset,
                 batch_size,
                 num_epochs,
                 model_directory,
                 save_steps=10,
                 patience=30,
                 shift_step=0,
                 feat_noise=0,
                 sdev_label=0,
                 fillgaps = 0,
                 reduce_lr=False,
                 function=np.min):

        train_loss, val_loss, val_acc, test_loss, test_acc, \
        disc_loss, task_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        x_s, y_s = src_dataset
        x_t, y_t = trgt_dataset
        x_t_, y_t_ = self._assign_missing_obs(x_s, x_t, y_t)

        self._get_encoder(x_s)
        self._get_task(x_s)
        self._get_discriminator(x_s)

        x_v, y_v = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_v.astype('float32'), y_v.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_t.astype('float32'), y_t.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        _ = self(tf.zeros([k for k in x_s.shape]))

        for epoch in range(num_epochs + 1):

            x_s, y_s, x_t_, y_t_ = shuffle(x_s, y_s, x_t_, y_t_)
            '''
            if patience and epoch >= patience:
                x_s, y_s = data_augmentation(x_s, y_s, shift_step, feat_noise, sdev_label, fillgaps)
                x_t_, _ = data_augmentation(x_t_, y_t_, shift_step, feat_noise, sdev_label, fillgaps)
            '''

            train_ds = self._init_dataset_training(x_s, y_s, x_t_, y_t_, batch_size)

            if self.config.adaptative:
                lambda_ = self._get_lambda(self.config.factor, num_epochs, epoch)
            else:
                lambda_ = 1.0 * self.config.factor

            self.trainstep_dann(train_ds, lambda_= lambda_)
            task_loss_epoch = self.task.loss_metric.result().numpy()
            enc_loss_epoch = self.encoder.loss_metric.result().numpy()
            disc_loss_epoch = self.discriminator.loss_metric.result().numpy()
            train_loss.append(enc_loss_epoch)
            disc_loss.append(disc_loss_epoch)
            task_loss.append(task_loss_epoch)

            self.task.loss_metric.reset_states()
            self.encoder.loss_metric.reset_states()
            self.discriminator.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_dann(val_ds)
                val_loss_epoch = self.task.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.task.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_dann(test_ds)
                test_loss_epoch = self.task.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.task.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Task loss {1}, Enc loss {2}, Disc loss {3}, Val loss {4}, Val acc {5}, Test loss {6}, Test acc {7}".format(
                        str(epoch),
                        str(task_loss_epoch), str(enc_loss_epoch), str(disc_loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                        str(round(test_loss_epoch, 4)), str(round(test_acc_result, 4))
                    ))

                if (function is np.min and val_loss_epoch < function(val_loss)
                        or function is np.max and val_loss_epoch > function(val_loss)):
                    # wait = 0
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.encoder.save_weights(os.path.join(model_directory, 'best_encoder_model'))
                    self.task.save_weights(os.path.join(model_directory, 'best_task_model'))
                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.encoder.save_weights(os.path.join(model_directory, 'last_encoder_model'))
        self.task.save_weights(os.path.join(model_directory, 'last_task_model'))

        # History of the training
        # History of the training
        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)
