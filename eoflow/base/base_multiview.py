from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation


class BaseModelMultiview(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def _assign_properties(self, model):
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.loss_metric = tf.keras.metrics.Mean()
        model.loss = self.loss
        return model


    def _get_multiview_model(self, x,  model_view_2, x_2):

        _ = self(tf.zeros(list(x.shape)))
        _ = model_view_2(tf.zeros(list(x_2.shape)))
        embedding_1, embedding_2 = self.layers[0].get_layer('embedding').output, \
                                   model_view_2.layers[0].get_layer('embedding').output

        concatenate = tf.keras.layers.Concatenate(axis=1)([embedding_1, embedding_2])
        net = self._fcn_layer(concatenate, 64)
        net = self._fcn_layer(net, 32)
        output =  tf.keras.layers.Dense(1, activation='linear', name='Discriminator')(net)

        model = tf.keras.Model(inputs = concatenate, outputs = output)
        self.model = self._assign_properties(model)
        self.model_view_2 = model_view_2

    @tf.function
    def trainstep_multiview(self,
                       train_ds):

        for Xs, Xt, y in train_ds:
            with tf.GradientTape() as enc_tape_1, tf.GradientTape() as enc_tape_2, tf.GradientTape() as task_tape:
                y_1, Xs_enc = self.call(Xs, training=True)
                y_2, Xt_enc = self.model_view_2(Xt, training=True)
                y_view = self.model.call([Xs_enc, Xt_enc], training=True)

                view1_loss = tf.reduce_mean(self.loss(y, y_1))
                view2_loss = tf.reduce_mean(self.loss(y, y_2))
                concatenate = tf.reduce_mean(self.loss(y, y_view))

                # https://stackoverflow.com/questions/56693863/why-does-model-losses-return-regularization-losses
                view1_loss += sum(self.losses)
                view2_loss += sum(self.model_view_2.losses)
                concatenate += sum(self.model.losses)

            gradients_enc_1 = enc_tape_1.gradient(view1_loss, self.trainable_variables)
            gradients_enc_2 = enc_tape_2.gradient(view2_loss, self.model_view_2.trainable_variables)
            gradients_final = task_tape.gradient(concatenate, self.model.trainable_variables)

            # Update weights
            opt_op_enc_1 = self.optimizer.apply_gradients(
                zip(gradients_enc_1, self.trainable_variables))
            opt_op_enc_2 = self.model_view_2.optimizer.apply_gradients(
                zip(gradients_enc_2, self.model_view_2.trainable_variables))
            opt_op_model = self.model.optimizer.apply_gradients(
                zip(gradients_final, self.model.trainable_variables))

            if self.config.ema:
                with tf.control_dependencies([opt_op_enc_1]):
                    self.ema.apply(self.trainable_variables)
                with tf.control_dependencies([opt_op_enc_2]):
                    self.ema.apply(self.model_view_2.trainable_variables)
                with tf.control_dependencies([opt_op_model]):
                    self.ema.apply(self.model.trainable_variables)

            self.loss_metric.update_state(view1_loss)
            self.model_view_2.loss_metric.update_state(view2_loss)
            self.model.loss_metric.update_state(concatenate)

    @tf.function
    def valstep_multiview(self, val_ds):
        for Xs, Xt, y in val_ds:
            y_1, Xs_enc = self.call(Xs, training=False)
            y_2, Xt_enc = self.model_view_2(Xt, training=False)
            y_view = self.model.call([Xs_enc, Xt_enc], training=False)

            view1_loss = tf.reduce_mean(self.loss(y, y_1))
            view2_loss = tf.reduce_mean(self.loss(y, y_2))
            concatenate = tf.reduce_mean(self.loss(y, y_view))

            self.loss_metric.update_state(view1_loss)
            self.model_view_2.loss_metric.update_state(view2_loss)
            self.model.loss_metric.update_state(concatenate)

            self.metric.update_state(y, y_view)


    def fit_multiview(self,
                 src_train_dataset,
                 aux_train_dataset,
                 src_test_dataset,
                 aux_test_dataset,
                 batch_size,
                 num_epochs,
                 model_directory,
                 save_steps=10,
                 patience=30,
                 reduce_lr=True,
                 function=np.min):

        train_loss, val_loss, val_acc, test_loss, test_acc, \
        disc_loss, task_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        x_s, y_s = src_train_dataset
        x_t = src_train_dataset

        self._get_multiview_model(x_s)

        x_v, y_v = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_v.astype('float32'), y_v.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_t.astype('float32'), y_t.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        _ = self(tf.zeros([k for k in x_s.shape]))

        for epoch in range(num_epochs + 1):

            x_s, y_s, x_t_, y_t_ = shuffle(x_s, y_s, x_t_, y_t_)
            if patience and epoch >= patience:
                x_s, y_s = data_augmentation(x_s, y_s, shift_step, feat_noise, sdev_label, fillgaps)
                x_t_, _ = data_augmentation(x_t_, y_t_, shift_step, feat_noise, sdev_label, fillgaps)

            train_ds = self._init_dataset_training(x_s, y_s, x_t_, y_t_, batch_size)

            if self.config.adaptative:
                lambda_ = self._get_lambda(self.config.factor, num_epochs, epoch)
            else:
                lambda_ = 1.0 * self.config.factor

            self.trainstep_multiview(train_ds, lambda_= lambda_)
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
                self.valstep_multiview(val_ds)
                val_loss_epoch = self.task.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.task.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_multiview(test_ds)
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