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
        return model

    def _add_fc_layers(self, net, nb_neurons, dropout_rate):
        layer_fcn = Dense(units=nb_neurons)(net)
        layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)
        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        return tf.keras.layers.Activation('relu')(layer_fcn)

    def _get_multiview_model(self, x,  model_view_2, x_2):

        _ = self(tf.zeros(list(x.shape)))
        _ = model_view_2(tf.zeros(list(x_2.shape)))
        embedding_1, embedding_2 = self.layers[0].get_layer('embedding').output, \
                                   model_view_2.layers[0].get_layer('embedding').output

        concatenate = tf.keras.layers.Concatenate(axis=1)([embedding_1, embedding_2])
        net = self._add_fc_layers(concatenate, 64, 0.5)
        net = self._add_fc_layers(net, 32, 0.5)
        output =  tf.keras.layers.Dense(1, activation='linear', name='task')(net)
        model_multiview = tf.keras.Model(inputs = concatenate, outputs = output)

        model_multiview = self._assign_properties(model_multiview)
        model_view_2 = self._assign_properties(model_view_2)

        return model_multiview


    @tf.function
    def trainstep_multiview(self,
                            train_ds,
                            model_multiview,
                            model_view_2,
                            lambda_ = 0.5,
                            gamma_ = 0.5,
                            loss = tf.keras.losses.MeanSquaredError()):

        for Xs, Xt, y in train_ds:
            with tf.GradientTape() as enc_tape_1, \
                    tf.GradientTape() as enc_tape_2, \
                    tf.GradientTape() as task_tape:

                y_1, Xs_enc = self.call(Xs, training=True)
                y_2, Xt_enc = model_view_2(Xt, training=True)

                data_fusion = tf.experimental.numpy.concatenate([Xs_enc, Xt_enc], axis = 1)
                y_view = model_multiview(data_fusion, training=True)

                concatenate = self.loss(y, y_view)
                view1_loss = self.loss(y, y_1) + lambda_ * concatenate + gamma_ * loss(y_2, y_1)
                view2_loss = self.loss(y, y_2) + lambda_ * concatenate + gamma_ * loss(y_1, y_2)

                view1_loss += sum(self.losses)
                view2_loss += sum(model_view_2.losses)
                concatenate += sum(model_multiview.losses)

            gradients_enc_1 = enc_tape_1.gradient(view1_loss, self.trainable_variables)
            gradients_enc_2 = enc_tape_2.gradient(view2_loss, model_view_2.trainable_variables)
            gradients_final = task_tape.gradient(concatenate, model_multiview.trainable_variables)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients_enc_1, self.trainable_variables))
            model_view_2.optimizer.apply_gradients(zip(gradients_enc_2, model_view_2.trainable_variables))
            model_multiview.optimizer.apply_gradients(zip(gradients_final, model_multiview.trainable_variables))

            self.loss_metric.update_state(tf.reduce_mean(view1_loss))
            model_view_2.loss_metric.update_state(tf.reduce_mean(view2_loss))
            model_multiview.loss_metric.update_state(tf.reduce_mean(concatenate))


    @tf.function
    def valstep_multiview(self, val_ds,
                          model_multiview,
                          model_view_2):
        for Xs, Xt, y in val_ds:
            _, Xs_enc = self.call(Xs, training=False)
            _, Xt_enc = model_view_2(Xt, training=False)
            data_fusion = tf.experimental.numpy.concatenate([Xs_enc, Xt_enc], axis=1)
            y_view = model_multiview.call(data_fusion, training=False)

            concatenate = tf.reduce_mean(self.loss(y, y_view))
            self.loss_metric.update_state(concatenate)
            self.metric.update_state(y, y_view)


    def fit_multiview(self,
                      model_view_2,
                      src_train_dataset,
                      src_val_dataset,
                      src_test_dataset,
                      batch_size,
                      num_epochs,
                      model_directory,
                      save_steps=10,
                      patience=30,
                      reduce_lr=True,
                      function=np.min):

        view1_loss, val_loss, val_acc, test_loss, test_acc, \
        view2_loss, multiview_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        xs, xt, y_s = src_train_dataset
        x_vs, xvt, y_vs = src_val_dataset
        x_ts, xtt, y_ts = src_test_dataset

        model_multiview = self._get_multiview_model(xs, model_view_2, xt)

        val_ds = tf.data.Dataset.from_tensor_slices((x_vs.astype('float32'), xvt.astype('float32'), y_vs.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_ts.astype('float32'), xtt.astype('float32'), y_ts.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        for epoch in range(num_epochs + 1):

            xs_, xt_, y_s_ = shuffle(xs, xt, y_s)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (xs_.astype('float32'), xt_.astype('float32'), y_s_.astype('float32'))).batch(batch_size)

            self.trainstep_multiview(train_ds, model_multiview, model_view_2)
            view1_loss_epoch = self.loss_metric.result().numpy()
            view2_loss_epoch = model_view_2.loss_metric.result().numpy()
            multiview_loss_epoch = model_multiview.loss_metric.result().numpy()

            view1_loss.append(view1_loss_epoch)
            view2_loss.append(view2_loss_epoch)
            multiview_loss.append(multiview_loss_epoch)

            self.loss_metric.reset_states()
            model_view_2.loss_metric.reset_states()
            model_multiview.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_multiview(val_ds, model_multiview, model_view_2)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_multiview(test_ds, model_multiview, model_view_2)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: View 1 loss {1}, View 2 loss {2}, Multiview loss {3}, Val loss {4}, Val acc {5}, Test loss {6}, Test acc {7}".format(
                        str(epoch),
                        str(view1_loss_epoch), str(view2_loss_epoch), str(multiview_loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                        str(round(test_loss_epoch, 4)), str(round(test_acc_result, 4))
                    ))

                if (function is np.min and val_loss_epoch < function(val_loss)
                        or function is np.max and val_loss_epoch > function(val_loss)):
                    # wait = 0
                    print('Best score seen so far ' + str(val_loss_epoch))

                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))
        model_view_2.save_weights(os.path.join(model_directory, 'last_model_view_2'))
        model_multiview.save_weights(os.path.join(model_directory, 'last_model_multiview'))
        # History of the training

        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)