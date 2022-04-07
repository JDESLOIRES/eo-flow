from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation


class BaseModelfusion(BaseModelCustomTraining):
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


    def _get_fusion_model(self, x_train, model_view_2, x_train_2):

        _ = self(tf.zeros(list(x_train.shape)))
        inputs = self.layers[0].layers[0].input
        _ = model_view_2(tf.zeros(list(x_train_2.shape)))
        inputs_2 = model_view_2.layers[0].layers[0].input


        model_view_2.layers[0].get_layer('embedding')._name = 'embedding_2'
        embedding_1, embedding_2 = self.layers[0].get_layer('embedding').output,\
                                   model_view_2.layers[0].get_layer('embedding_2').output

        concatenate = tf.keras.layers.Concatenate(axis=1)([embedding_1, embedding_2])
        net = self._add_fc_layers(concatenate, 32, 0.5)
        net = self._add_fc_layers(net, 16, 0.5)
        output =  tf.keras.layers.Dense(1, activation='linear', name='task')(net)
        model_fusion = tf.keras.Model(inputs = [inputs, inputs_2], outputs = output)
        self.model_fusion = self._assign_properties(model_fusion)
        print(self.model_fusion.summary())

    @tf.function
    def trainstep_fusion(self,
                        train_ds):

        for Xs, Xt, y in train_ds:
            with tf.GradientTape() as tape:
                y_preds = self.model_fusion.call([Xs, Xt], training=True)
                cost = self.loss(y, y_preds)
                cost += sum(self.model_fusion.losses)

            gradients_enc_1 = tape.gradient(cost, self.model_fusion.trainable_variables)
            # Update weights
            self.model_fusion.optimizer.apply_gradients(zip(gradients_enc_1, self.model_fusion.trainable_variables))
            self.loss_metric.update_state(tf.reduce_mean(cost))

    @tf.function
    def valstep_fusion(self, val_ds):

        for Xs, Xt, y in val_ds:
            y_preds = self.model_fusion.call([Xs, Xt], training=False)
            cost = self.loss(y, y_preds)
            self.metric.update_state(y, y_preds)
            self.loss_metric.update_state(tf.reduce_mean(cost))

    def fit_fusion(self,
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
        view2_loss, fusion_loss = ([np.inf] if function == np.min else [-np.inf] for i in range(7))

        xs, xt, y_s = src_train_dataset
        x_vs, xvt, y_vs = src_val_dataset
        x_ts, xtt, y_ts = src_test_dataset

        self._get_fusion_model(xs, model_view_2, xt)

        val_ds = tf.data.Dataset.from_tensor_slices((x_vs.astype('float32'), xvt.astype('float32'), y_vs.astype('float32'))).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_ts.astype('float32'), xtt.astype('float32'), y_ts.astype('float32'))).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience // 4, factor=0.5)
        wait = 0

        for epoch in range(num_epochs + 1):

            xs_, xt_, y_s_ = shuffle(xs, xt, y_s)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (xs_.astype('float32'), xt_.astype('float32'), y_s_.astype('float32'))).batch(batch_size)

            self.trainstep_fusion(train_ds)
            fusion_loss_epoch = self.loss_metric.result().numpy()
            fusion_loss.append(fusion_loss_epoch)

            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                self.valstep_fusion(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                self.valstep_fusion(test_ds)
                test_loss_epoch = self.loss_metric.result().numpy()
                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print(
                    "Epoch {0}: fusion loss {1}, Val loss {2}, Val acc {3}, Test loss {4}, Test acc {5}".format(
                        str(epoch),
                        str(fusion_loss_epoch),
                        str(round(val_loss_epoch, 4)), str(round(val_acc_result, 4)),
                        str(round(test_loss_epoch, 4)), str(round(test_acc_result, 4))
                    ))

                if (function is np.min and val_loss_epoch < function(val_loss)
                        or function is np.max and val_loss_epoch > function(val_loss)):
                    print('Best score seen so far ' + str(val_loss_epoch))

                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)

        self.save_weights(os.path.join(model_directory, 'last_model'))
        model_view_2.save_weights(os.path.join(model_directory, 'last_model_view_2'))
        self.model_fusion.save_weights(os.path.join(model_directory, 'last_model_fusion'))
        # History of the training

        losses = dict(train_loss_results=train_loss[1:],
                      val_loss_results=val_acc[1:],
                      disc_loss_results=disc_loss[1:],
                      task_loss_results=task_loss[1:]
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)