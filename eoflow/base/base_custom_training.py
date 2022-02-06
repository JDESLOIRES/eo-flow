import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle

import os

import tensorflow as tf

from . import Configurable
from eoflow.models.data_augmentation import add_random_shift, add_random_noise, add_random_target

class BaseModelCustomTraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self.init_model()

    def init_model(self):
        """ Called on __init__. Keras self initialization. Create self here if does not require the inputs shape """
        pass

    def build(self, inputs_shape):
        """ Keras method. Called once to build the self. Build the self here if the input shape is required. """
        pass

    def call(self, inputs, training=False):
        pass

    def prepare(self, optimizer=None, loss=None, metrics=None,
                epoch_loss_metric = None,
                epoch_val_metric = None,
                **kwargs):
        """ Prepares the self for training and evaluation. This method should create the
        optimizer, loss and metric functions and call the compile method of the self. The self
        should provide the defaults for the optimizer, loss and metrics, which can be overriden
        with custom arguments. """

        raise NotImplementedError

    @tf.function
    def train_step(self,
                   train_ds,
                   noisy = False):
        # pb_i = Progbar(len(list(train_ds)), stateful_metrics='acc')
        for x_batch_train, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:
                y_preds = self.call(x_batch_train,
                                    training=True)
                cost = self.loss(y_batch_train, y_preds)

            grads = tape.gradient(cost, self.trainable_variables)
            opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_metric.update_state(cost)
            with tf.control_dependencies([opt_op]):
                self.ema.apply(self.trainable_variables)

    # Function to run the validation step.
    @tf.function
    def val_step(self, val_ds):
        for x, y in val_ds:
            y_preds = self.call(x, training=False)
            cost = self.loss(y, y_preds)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y +1, y_preds+1)

    @staticmethod
    def _data_augmentation(x_train_, y_train_, timeshift,random_noise, noisy_label):
        if timeshift:
            x_train_ = add_random_shift(x_train_, timeshift)
        if random_noise:
            x_train_ = add_random_noise(x_train_, random_noise)
        if noisy_label:
            y_train_ = add_random_target(y_train_, noisy_label)

        return x_train_, y_train_

    def fit(self,
            dataset,
            val_dataset,
            batch_size,
            num_epochs,
            model_directory,
            save_steps=10,
            timeshift = 0,
            random_noise = 0,
            noisy_label = 0,
            function=np.max):

        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = dataset


        x_val, y_val = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        _ = self(tf.zeros([k for k in x_train.shape]))

        for epoch in range(num_epochs + 1):

            x_train_, y_train_ = shuffle(x_train, y_train)
            x_train_, y_train_ = self._data_augmentation(x_train_, y_train_, timeshift,random_noise, noisy_label)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(batch_size)

            self.train_step(train_ds)

            # End epoch
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % save_steps == 0:
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch),
                        str(loss_epoch),
                        str(round(val_loss_epoch, 4)),
                        str(round(val_acc_result, 4)),
                    ))

                if (
                    function is np.min
                    and val_loss_epoch < function(val_loss)
                    or function is np.max
                    and val_loss_epoch > function(val_loss)
                ):
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.save_weights(os.path.join(model_directory, 'best_model'))

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)
                self.loss_metric.reset_states()
                self.metric.reset_states()


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