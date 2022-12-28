from sklearn.utils import shuffle
import numpy as np
import os

from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation
import pickle
import itertools
import helpers

import tensorflow as tf
from sklearn.metrics import r2_score


class BaseModelForecast(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def _layer_decoding(self, net, nb_neurons, activation="linear"):

        layer_fcn = Dense(
            units=nb_neurons,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer),
        )(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(1 - self.config.keep_prob)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(activation)(layer_fcn)

        return layer_fcn

    def _init_ssl_models(self, input_shape, output_shape, add_layer=False):

        _ = self(tf.zeros(list((1, input_shape))))

        inputs = self.layers[0].input
        latent = (
            self.layers[0]
            .layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4]
            .output
        )
        decode = self._layer_decoding(
            latent, nb_neurons=output_shape, activation="linear"
        )

        if add_layer:
            output_task = self.layers[0].layers[-2].output
            output_task = self._layer_decoding(
                output_task,
                nb_neurons=self.config.nb_fc_neurons // 4,
                activation="relu",
            )
            output_task = Dense(1, activation="linear", name="prediction")(output_task)
        else:
            output_task = self.layers[0].layers[-1].output

        encoder = tf.keras.Model(inputs=inputs, outputs=latent)
        encoder.summary()
        self.encoder = self._assign_properties(encoder)

        decoder = tf.keras.Model(inputs=latent, outputs=decode)
        decoder.summary()
        self.decoder = self._assign_properties(decoder)

        task = tf.keras.Model(inputs=latent, outputs=output_task)
        task.summary()
        self.task = self._assign_properties(task)

    def forward_step(self, x, training=True):
        h = self.encoder(x, training)
        x_reco = self.decoder(h, training)
        task = self.task(h, training)
        return h, x_reco, task

    def forecast_step(self, xis, x_orig, kd, lambda_, p_m, noise_level):

        x_bar_noisy = self.generate_noisy_xbar(xis, noise_level=noise_level)
        # Generate binary mask
        mask = np.random.binomial(1, p_m, xis.shape)
        # Replace selected x_bar features with the noisy ones
        x_bar = xis * (1 - mask) + x_bar_noisy * mask
        loss_task = 0

        with tf.GradientTape(persistent=True) as tape:
            his, x_reco_is, task_is = self.forward_step(x=x_bar, training=True)
            loss_reconstruction = self.loss(x_orig, x_reco_is)

            if len(set(list(kd.flatten()))) > 1:
                loss_task = tf.reduce_mean(lambda_ * self.loss(kd, task_is))

            total_loss = loss_reconstruction + loss_task

        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_variables)
        )

        gradients = tape.gradient(loss_reconstruction, self.decoder.trainable_variables)
        self.decoder.optimizer.apply_gradients(
            zip(gradients, self.decoder.trainable_variables)
        )

        gradients = tape.gradient(loss_task, self.task.trainable_variables)
        self.task.optimizer.apply_gradients(
            zip(gradients, self.task.trainable_variables)
        )

        self.encoder.loss_metric.update_state(tf.reduce_mean(total_loss))
        self.decoder.loss_metric.update_state(tf.reduce_mean(loss_reconstruction))
        self.task.loss_metric.update_state(tf.reduce_mean(loss_task))

    def trainstep_is(self, xis, y_batch_train, kd, lambda_=1):

        with tf.GradientTape(persistent=True) as tape:
            h, x_reco, y_pred = self.forward_step(xis, training=True)
            loss_task = tf.reduce_mean(self.loss(y_batch_train, y_pred))
            if 1 < len(set(list(kd.flatten()))):
                loss_task += tf.reduce_mean(lambda_ * self.loss(kd, y_pred))

        gradients = tape.gradient(loss_task, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_variables)
        )

        gradients = tape.gradient(loss_task, self.task.trainable_variables)
        self.task.optimizer.apply_gradients(
            zip(gradients, self.task.trainable_variables)
        )

        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(
            tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred
        )

    def valstep_is(self, xis, y_batch_train):

        h, x_reco, task = self.forward_step(xis, training=False)

        loss_task = self.loss(y_batch_train, task)
        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(task)), task)

    def fit_forecast(
        self,
        x_is,
        x_orig,
        model_directory,
        batch_size=8,
        num_epochs=500,
        model_kd=None,
        p_m=0,
        noise_level=0,
        lambda_=1,
    ):

        train_loss = []
        self._init_ssl_models(input_shape=x_is.shape[-1], output_shape=x_orig.shape[-1])

        if model_kd is None:
            y_preds = np.zeros((x_is.shape[0], 1))
        elif model_kd.config.adaptative:
            y_preds, h_latent = model_kd.predict(x_orig)
        else:
            y_preds = model_kd.predict(x_orig)

        for ep in range(num_epochs):
            x_is_, x_orig_, y_preds_ = shuffle(x_is, x_orig, y_preds)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_is_, x_orig_, y_preds_)
            ).batch(batch_size)

            for x_is_batch_train, x_orig_batch_train, y_batch_train in train_ds:
                self.forecast_step(
                    xis=x_is_batch_train,
                    x_orig=x_orig_batch_train,
                    kd=y_batch_train.numpy(),
                    lambda_=lambda_,
                    p_m=p_m,
                    noise_level=noise_level,
                )

            loss_epoch = self.encoder.loss_metric.result().numpy()
            loss_reco = self.decoder.loss_metric.result().numpy()
            loss_task = self.task.loss_metric.result().numpy()

            print(
                "Epoch "
                + str(ep)
                + " : "
                + str(loss_epoch)
                + "; reconstruction "
                + str(loss_reco)
                + "; task "
                + str(loss_task)
            )
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.encoder.loss_metric.reset_states()
            self.task.loss_metric.reset_states()

        self.encoder.save_weights(os.path.join(model_directory, "encoder_model"))
        self.decoder.save_weights(os.path.join(model_directory, "decoder_model"))
        self.task.save_weights(os.path.join(model_directory, "task_model"))

        # History of the training
        losses = dict(train_loss_results=train_loss)
        with open(os.path.join(model_directory, "history.pickle"), "wb") as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

    def fit_is(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_epochs,
        model_directory,
        save_steps=10,
        reduce_lr=True,
        patience=50,
        finetuning=False,
        add_layer=False,
        unfreeze=False,
        model_kd=None,
        lambda_=1,
        function=np.min,
    ):

        global val_acc_result
        train_loss, val_loss, val_acc = (
            [np.inf] if function == np.min else [-np.inf] for i in range(3)
        )

        x_is_train, x_orig_train, y_train = train_dataset
        x_is_val, y_val = val_dataset
        x_is_test, y_test = test_dataset

        if model_kd is None:
            y_preds = np.zeros((x_is_train.shape[0], 1))
        elif model_kd.config.adaptative:
            y_preds, h_latent = model_kd.predict(x_orig_train)
        else:
            y_preds = model_kd.predict(x_orig_train)

        self._init_ssl_models(
            input_shape=x_is_train.shape[-1],
            output_shape=x_orig_train.shape[-1],
            add_layer=add_layer,
        )

        _ = self.encoder(tf.zeros([0, x_is_train.shape[-1]]))
        _ = self.task(_)
        self.encoder.load_weights(os.path.join(model_directory, "encoder_model"))

        try:
            self.task.load_weights(os.path.join(model_directory, "task_model"))
        except:
            pass

        if finetuning:
            for i in range(len(self.encoder.layers)):
                self.encoder.layers[i].trainable = False

        val_ds = tf.data.Dataset.from_tensor_slices((x_is_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_is_test, y_test)).batch(
            batch_size
        )

        reduce_rl_plateau = self._reduce_lr_on_plateau(
            patience=patience // 4, factor=0.5
        )
        wait = 0

        for epoch in range(num_epochs):
            if patience and epoch >= patience and self.config.finetuning and unfreeze:
                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].trainable = True

            x_is_train_, x_orig_train_, y_train_, y_preds_ = shuffle(
                x_is_train, x_orig_train, y_train, y_preds
            )
            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_is_train_, x_orig_train_, y_train_, y_preds_)
            ).batch(batch_size)

            for (
                x_is_batch_train,
                x_orig_batch_train,
                y_batch_train,
                y_preds_batch,
            ) in train_ds:
                self.trainstep_is(
                    xis=x_is_batch_train,
                    y_batch_train=y_batch_train,
                    kd=y_preds_batch.numpy(),
                    lambda_=lambda_,
                )

            loss_epoch = self.loss_metric.result().numpy()
            train_acc_result = self.metric.result().numpy()
            print("Epoch " + str(epoch) + " : " + str(loss_epoch))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.metric.reset_states()

            if epoch % save_steps == 0:
                wait += 1
                for x_is_batch_train, y_batch_train in val_ds:
                    self.valstep_is(x_is_batch_train, y_batch_train)

                val_loss_epoch = self.loss_metric.result().numpy()
                val_loss.append(val_loss_epoch)

                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                for x_is_batch_train, y_batch_train in test_ds:
                    self.valstep_is(x_is_batch_train, y_batch_train)

                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print("Val loss " + str(epoch) + " : " + str(val_loss_epoch))
                print("Train acc " + str(epoch) + " : " + str(val_acc_result))
                print("Val acc " + str(epoch) + " : " + str(train_acc_result))
                print("Test acc " + str(epoch) + " : " + str(test_acc_result))

                if (
                    function is np.min
                    and val_loss_epoch <= function(val_loss)
                    or function is np.max
                    and val_loss_epoch >= function(val_loss)
                ):
                    print("Best score seen so far " + str(val_loss_epoch))
                    self.encoder.save_weights(
                        os.path.join(model_directory, "encoder_best_model")
                    )
                    self.task.save_weights(
                        os.path.join(model_directory, "task_best_model")
                    )

                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

        self.encoder.save_weights(os.path.join(model_directory, "encoder_last_model"))
        self.task.save_weights(os.path.join(model_directory, "task_last_model"))
