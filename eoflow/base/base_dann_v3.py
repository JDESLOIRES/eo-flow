from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from tensorflow.keras.layers import Dense
from .base_dann import BaseModelAdapt
from eoflow.models.data_augmentation import data_augmentation


class BaseModelAdaptV3(BaseModelAdapt):
    def __init__(self, config_specs):
        BaseModelAdapt.__init__(self, config_specs)

    @staticmethod
    def _loss_dom_func(input_logits, target_labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=target_labels))

    def trainstep_dann_v3(self,
                          train_ds,
                          lambda_=1.0):

        cost_disc = np.array([])
        cost_task = np.array([])

        for Xs, ys, Xt, yt in train_ds:
            with tf.GradientTape() as gradients_task:
                ys_pred, ys_disc = self.call(Xs, training=True)
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                cost = self.loss(ys, ys_pred)
                _, yt_disc = self.call(Xt, training=True)

                # Compute the discriminator loss values
                domain_labels = np.vstack([np.tile([1., 0.], [tf.shape(ys)[0], 1]),
                                           np.tile([0., 1.], [tf.shape(yt)[0], 1])]).astype('float32')
                y_disc = np.vstack([ys_disc, yt_disc])
                disc_loss = self._loss_dom_func(y_disc, domain_labels)
                loss = cost + lambda_ * disc_loss

            grads = gradients_task.gradient(loss, self.trainable_variables)
            # Update weights
            opt_op_task = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            if self.config.ema:
                with tf.control_dependencies([opt_op_task]):
                    self.ema.apply(self.trainable_variables)

            self.loss_metric.update_state(loss)
            cost_task = np.append(cost_task, tf.reduce_mean(cost).numpy())
            cost_disc = np.append(cost_disc, tf.reduce_mean(disc_loss).numpy())

        return np.round(np.mean(cost_task), 3),  np.round(np.mean(cost_disc), 3)



    @tf.function
    def valstep_dann_v3(self, val_ds):
        for x_batch, y_batch in val_ds:
            y_pred, _ = self.call(x_batch, training=False)
            y_pred = tf.reshape(y_pred, tf.shape(y_batch))
            cost = self.loss(y_batch, y_pred)

            cost = tf.reduce_mean(cost)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y_batch, y_pred)

    def fit_dann_v3(self,
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

        train_loss, val_loss, val_acc, test_loss, test_acc,\
        disc_loss, task_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        x_s, y_s = src_dataset
        x_t, y_t = trgt_dataset
        x_t_, y_t_ = self._assign_missing_obs(x_s, x_t, y_t)

        x_v, y_v = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_v.astype('float32'), y_v.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_t.astype('float32'), y_t.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        _, _ = self(tf.zeros([k for k in x_s.shape]))

        for epoch in range(num_epochs + 1):

            x_s, y_s, x_t_, y_t_ = shuffle(x_s, y_s, x_t_, y_t_)

            if (patience and epoch >= patience) and (shift_step or feat_noise or sdev_label or fillgaps):
                x_s, y_s = data_augmentation(x_s, y_s, shift_step, feat_noise, sdev_label, fillgaps)
                x_t_, _ = data_augmentation(x_t_, y_t_, shift_step, feat_noise, sdev_label, fillgaps)

            train_ds = self._init_dataset_training(x_s, y_s, x_t_, y_t_, batch_size)

            if self.config.adaptative:
                lambda_ = self._get_lambda(self.config.factor, num_epochs, epoch)
            else:
                lambda_ = 1.0 * self.config.factor

            task_loss_epoch, disc_loss_epoch = self.trainstep_dann_v3(train_ds, lambda_)
            enc_loss_epoch = self.loss_metric.result().numpy()

            train_loss.append(enc_loss_epoch)
            disc_loss.append(disc_loss_epoch)
            task_loss.append(task_loss_epoch)

            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_dann_v3(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_dann_v3(test_ds)
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
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.save_weights(os.path.join(model_directory, 'best_model'))
                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))

        # History of the training
        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )

        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)
