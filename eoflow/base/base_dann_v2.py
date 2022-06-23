from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from .base_dann import BaseModelAdapt
from eoflow.models.data_augmentation import data_augmentation


class BaseModelAdaptV2(BaseModelAdapt):
    def __init__(self, config_specs):
        BaseModelAdapt.__init__(self, config_specs)

    @tf.function
    def trainstep_dann_v2(self,
                       train_ds,
                       lambda_=1.0):

        for Xs, ys, Xt, yt in train_ds:
            with tf.GradientTape(persistent=True) as gradients_task:
                if self.config.loss in ['gaussian', 'laplacian']:
                    ys_pred, sigma_s,  Xs_enc = self.call(Xs, training=True)
                    ys_pred, sigma_s = tf.reshape(ys_pred, tf.shape(ys)), tf.reshape(sigma_s, tf.shape(ys))
                    cost = self.loss(ys_pred, sigma_s, ys)
                else:
                    ys_pred, Xs_enc = self.call(Xs, training=True)
                    ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                    cost = self.loss(ys, ys_pred)

                ys_disc = self.discriminator.call(Xs_enc, training=True)

                _, Xt_enc = self.call(Xt, training=True)
                yt_disc = self.discriminator.call(Xt_enc, training=True)
                yt_disc = tf.reshape(yt_disc, tf.shape(ys))

                # Compute the discriminator loss values
                disc_loss = 0.5 * (self.discriminator.loss(tf.ones_like(ys_disc), ys_disc) +
                                   self.discriminator.loss(tf.zeros_like(yt_disc), yt_disc))
                loss_dann = 0.5 * (self.discriminator.loss(tf.zeros_like(ys_disc), ys_disc) +
                                   self.discriminator.loss(tf.ones_like(yt_disc), yt_disc))

                enc_loss = cost + lambda_ * loss_dann
                enc_loss = tf.reduce_mean(enc_loss)

            grads_task = gradients_task.gradient(enc_loss, self.trainable_variables)
            grads_disc = gradients_task.gradient(disc_loss, self.discriminator.trainable_variables)

            cost += sum(self.losses)
            enc_loss += sum(self.discriminator.losses) + sum(self.losses)
            disc_loss += sum(self.discriminator.losses)

            # Update weights
            opt_op_task = self.optimizer.apply_gradients(zip(grads_task, self.trainable_variables))
            disc_op_task = self.discriminator.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

            if self.config.ema:
                with tf.control_dependencies([opt_op_task]):
                    self.ema.apply(self.trainable_variables)

            self.task.loss_metric.update_state(cost)
            self.loss_metric.update_state(enc_loss)
            self.discriminator.loss_metric.update_state(disc_loss)

    @tf.function
    def valstep_dann_v2(self, val_ds):
        for x_batch, y_batch in val_ds:
            if self.config.loss in ['gaussian', 'laplacian']:
                y_pred, sigma_, _ = self.call(x_batch, training=False)
                y_pred, sigma_ = tf.reshape(y_pred, tf.shape(y_batch)), tf.reshape(sigma_, tf.shape(y_batch))
                cost = self.loss(y_pred, sigma_, y_batch)
            else:
                y_pred, _ = self.call(x_batch, training=False)
                y_pred = tf.reshape(y_pred, tf.shape(y_batch))
                cost = self.loss(y_batch, y_pred)

            cost = tf.reduce_mean(cost)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y_batch, y_pred)

    def fit_dann_v2(self,
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
            if patience and epoch >= patience:
                if self.config.finetuning:
                    for i in range(len(self.layers[0].layers)):
                        self.layers[0].layers[i].trainable = True

                if sdev_label or fillgaps or shift_step:
                    x_s, y_s = data_augmentation(x_s, y_s, shift_step, feat_noise, sdev_label, fillgaps)
                    x_t_, _ = data_augmentation(x_t_, y_t_, shift_step, feat_noise, sdev_label, fillgaps)

            train_ds = self._init_dataset_training(x_s, y_s, x_t_, y_t_, batch_size)

            if self.config.adaptative:
                lambda_ = self._get_lambda(self.config.factor, num_epochs, epoch)
            else:
                lambda_ = self.config.factor

            self.trainstep_dann_v2(train_ds, lambda_)
            task_loss_epoch = self.task.loss_metric.result().numpy()
            enc_loss_epoch = self.loss_metric.result().numpy()
            disc_loss_epoch = self.discriminator.loss_metric.result().numpy()
            train_loss.append(task_loss_epoch)

            self.task.loss_metric.reset_states()
            self.loss_metric.reset_states()
            self.discriminator.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_dann_v2(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_dann_v2(test_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
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
                    self.save_weights(os.path.join(model_directory, 'best_model'))
                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))

        # History of the training
        # History of the training
        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

