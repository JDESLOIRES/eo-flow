from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from .base_dann import BaseModelAdapt
from eoflow.models.data_augmentation import data_augmentation

class BaseModelAdaptCoral(BaseModelAdapt):
    def __init__(self, config_specs):
        BaseModelAdapt.__init__(self, config_specs)

    @tf.function
    def trainstep_coral(self,
                        train_ds,
                        _match_mean=1,
                        lambda_ = 1):

        for Xs, ys, Xt, yt in train_ds:
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:
                Xs_enc = self.encoder(Xs, training=True)
                ys_pred = self.task(Xs_enc, training=True)
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                Xt_enc = self.encoder(Xt, training=True)

                batch_size = tf.cast(tf.shape(Xs_enc)[0], Xs_enc.dtype)
                factor_1 = 1. / (batch_size - 1. + 1e-8)
                factor_2 = 1. / batch_size

                sum_src = tf.reduce_sum(Xs_enc, axis=0)
                sum_src_row = tf.reshape(sum_src, (1, -1))
                sum_src_col = tf.reshape(sum_src, (-1, 1))

                cov_src = factor_1 * (
                        tf.matmul(tf.transpose(Xs_enc), Xs_enc) -
                        factor_2 * tf.matmul(sum_src_col, sum_src_row)
                )

                sum_tgt = tf.reduce_sum(Xt_enc, axis=0)
                sum_tgt_row = tf.reshape(sum_tgt, (1, -1))
                sum_tgt_col = tf.reshape(sum_tgt, (-1, 1))

                cov_tgt = factor_1 * (
                        tf.matmul(tf.transpose(Xt_enc), Xt_enc) -
                        factor_2 * tf.matmul(sum_tgt_col, sum_tgt_row)
                )

                mean_src = tf.reduce_mean(Xs_enc, 0)
                mean_tgt = tf.reduce_mean(Xt_enc, 0)

                task_loss = self.loss(ys, ys_pred)
                disc_loss_cov = 0.25 * tf.square(cov_src - cov_tgt)
                disc_loss_mean = tf.square(mean_src - mean_tgt)

                task_loss = tf.reduce_mean(task_loss)
                disc_loss_cov = tf.reduce_mean(disc_loss_cov)
                disc_loss_mean = tf.reduce_mean(disc_loss_mean)
                disc_loss = lambda_ * (disc_loss_cov + _match_mean * disc_loss_mean)

            grads_task = task_tape.gradient(task_loss, self.task.trainable_variables)
            grads_disc = enc_tape.gradient(disc_loss, self.encoder.trainable_variables)

            self.loss_metric.update_state(task_loss)
            self.encoder.loss_metric.update_state(disc_loss)

    @tf.function
    def valstep_coral(self, val_ds):
        for x_batch, y_batch in val_ds:
            x_enc = self.encoder.call(x_batch, training=False)
            y_pred = self.task.call(x_enc, training = False)
            cost = self.loss(y_batch, y_pred)

            cost = tf.reduce_mean(cost)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y_batch, y_pred)

    def fit_coral(self,
                src_dataset,
                val_dataset,
                trgt_dataset,
                batch_size,
                num_epochs,
                model_directory,
                save_steps=10,
                patience=30,
                _match_mean = 1,
                lambda_ = 1,
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

            self.trainstep_coral(train_ds, _match_mean, lambda_)
            disc_loss_epoch = self.encoder.loss_metric.result().numpy()
            task_loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(task_loss_epoch)

            self.encoder.loss_metric.reset_states()
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_coral(test_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: Task loss {1}, Disc loss {2}, Val loss {3}, Val acc {4}, Test loss {5}, Test acc {6}".format(
                        str(epoch),
                        str(task_loss_epoch), str(disc_loss_epoch),
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
        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )

        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)
