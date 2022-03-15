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

    def trainstep_dann_v2(self,
                       train_ds,
                       lambda_=1.0,
                       eps=1e-6):

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

                # Compute the loss value
                loss = tf.reduce_mean(cost)
                disc_loss = tf.reduce_mean((-tf.math.log(ys_disc + eps) - tf.math.log(1 - yt_disc + eps)))
                task_loss = loss - lambda_ * disc_loss

            grads_task = gradients_task.gradient(task_loss, self.trainable_variables)
            grads_disc = gradients_task.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update weights
            opt_op_task = self.optimizer.apply_gradients(zip(grads_task, self.trainable_variables))
            with tf.control_dependencies([opt_op_task]):
                self.ema.apply(self.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

            self.loss_metric.update_state(task_loss)
            self.discriminator.loss_metric.update_state(disc_loss)

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
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    batch_size,
                    num_epochs,
                    model_directory,
                    save_steps=10,
                    patience=30,
                    factor=1.0,
                    shift_step=0,
                    feat_noise=0,
                    sdev_label=0,
                    fillgaps = 0,
                    reduce_lr=False,
                    function=np.min):

        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))
        x_s, y_s = train_dataset
        x_t, y_t = test_dataset
        x_t, y_t = self._assign_missing_obs(x_s, x_t, y_t)

        self._get_discriminator(x_s)

        x_v, y_v = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_v.astype('float32'), y_v.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        _ = self(tf.zeros([k for k in x_s.shape]))

        for epoch in range(num_epochs + 1):

            x_s, y_s, x_t, y_t = shuffle(x_s, y_s, x_t, y_t)
            if patience and epoch >= patience:
                x_s, y_s = data_augmentation(x_s, y_s, shift_step, feat_noise, sdev_label, fillgaps)
                x_t, _ = data_augmentation(x_t, y_t, shift_step, feat_noise, sdev_label, fillgaps)
            train_ds = self._init_dataset_training(x_s, y_s, x_t, y_t, batch_size)
            lambda_ = self._get_lambda(factor, num_epochs, epoch)

            self.trainstep_dann_v2(train_ds, lambda_)
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)

            self.loss_metric.reset_states()
            self.discriminator.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_dann_v2(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch), str(loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4))
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
        losses = dict(train_loss_results=train_loss,
                      val_loss_results=val_acc
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)