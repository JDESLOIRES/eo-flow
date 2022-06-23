from sklearn.utils import shuffle
import numpy as np
import os

import tensorflow as tf

from eoflow.models.data_augmentation import timeshift, feature_noise
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation
import pickle


class BaseModelSLLTraining(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)


    @tf.function
    def train_step_ssl(self,
                       train_ds,
                       f_map = True,
                       lambda_ = 0.5):

        for x_batch_train, x_batch_u, x_batch_u_shift, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:

                y_preds, _ = self.call(x_batch_train, training=True)
                if f_map:
                    _, orig = self.call(x_batch_u, training=False)
                    _, shift = self.call(x_batch_u, training=False)
                else:
                    orig, _ = self.call(x_batch_u, training=False)
                    shift, _ = self.call(x_batch_u, training=False)

                cost = self.loss(y_batch_train, y_preds) + lambda_ * self.loss(orig, shift)
                cost += sum(self.losses)

                cost = tf.reduce_mean(cost)

            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_metric.update_state(cost)

            if self.config.ema:
                with tf.control_dependencies([opt_op]):
                    self.ema.apply(self.trainable_variables)

    @tf.function
    def valstep_ssl(self,
                    train_ds):

        for x_batch_train, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:

                y_preds, _ = self.call(x_batch_train, training=False)


                cost = self.loss(y_batch_train, y_preds)
                cost += sum(self.losses)

                cost = tf.reduce_mean(cost)
            self.loss_metric.update_state(cost)
            self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_preds)), y_preds)


    def fit_ssl(self,
                train_dataset,
                val_dataset,
                unl_dataset,
                test_dataset,
                batch_size,
                num_epochs,
                model_directory,
                save_steps=10,
                shift_step=0,
                feat_noise=0,
                sdev_label=0,
                fillgaps=0,
                patience = 30,
                lambda_ = 1,
                reduce_lr = False,
                f_map = True,
                function=np.min):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = train_dataset
        x_val, y_val = val_dataset
        x_test, y_test = test_dataset

        _ = self(tf.zeros(list(x_train.shape)))

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience//4, factor=0.5)
        wait = 0
        n_forget = 0

        for epoch in range(num_epochs + 1):
            if unl_dataset.shape[0]>x_train.shape[0]:
                random_obs = np.random.choice(unl_dataset.shape[0],
                                              size=x_train.shape[0], replace=False)
                x_unl = unl_dataset[random_obs,]
            else:
                unl_dataset  = np.repeat(unl_dataset, x_train.shape[0] // unl_dataset.shape[0], axis=0)
                num_missing = x_train.shape[0] - unl_dataset.shape[0]
                additional_obs = np.random.choice(unl_dataset.shape[0], size=num_missing, replace=False)
                x_unl = np.concatenate([unl_dataset, unl_dataset[additional_obs,]], axis=0)

            x_train_, y_train_ = shuffle(x_train, y_train)

            if patience and epoch >= patience and self.config.finetuning:
                for i in range(len(self.layers[0].layers)):
                    self.layers[0].layers[i].trainable = True

            if shift_step:
                x_unl_shift, _ = data_augmentation(x_unl, np.zeros(x_unl.shape[0]),
                                                   shift_step, feat_noise, sdev_label, fillgaps)
            else:
                x_unl_shift = x_unl * np.random.normal(1, 0.1, (x_unl.shape))


            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, x_unl, x_unl_shift, y_train_)).batch(batch_size)

            self.train_step_ssl(train_ds, f_map, lambda_)
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                wait +=1
                self.valstep_ssl(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()
                self.valstep_ssl(test_ds)
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
        losses = dict(train_loss_results=train_loss,
                      val_loss_results=val_acc
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

    def train_and_evaluate(self,
                           train_dataset,
                           val_dataset,
                           num_epochs,
                           iterations_per_epoch,
                           model_directory,
                           **kwargs):

        return self.fit(train_dataset,
                        val_dataset,
                        num_epochs=num_epochs,
                        model_directory=model_directory,
                        save_steps=iterations_per_epoch,
                        **kwargs)
