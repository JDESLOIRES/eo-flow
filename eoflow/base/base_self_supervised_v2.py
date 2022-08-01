import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os

from tensorflow.keras.layers import Dense

from .base_self_supervised import BaseModelCustomTraining

import pickle


import tensorflow as tf


cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):
    return cosine_sim_1d(x, y)


def _cosine_simililarity_dim2(x, y):
    # sourcery skip: inline-immediately-returned-variable
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


#https://keras.io/examples/vision/semisupervised_simclr/

class BaseModelSelfTrainingV2(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def _init_models_cl(self, input_shape, output_shape, add_layer = False):

        _ = self(tf.zeros(list((1, input_shape))))
        inputs = self.layers[0].input

        latent = self.layers[0].layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4].output

        z = Dense(self.layers[0].layers[0].input.shape[-1], activation='relu', name='head_1')(latent)
        z = Dense(self.layers[0].layers[0].input.shape[-1], activation=None, name='linear_2')(z)

        decode = self._layer_decoding(latent, nb_neurons = output_shape, activation='linear')

        if add_layer:
            output_task = self.layers[0].layers[-2].output
            output_task = self._layer_decoding(output_task, nb_neurons=self.config.nb_fc_neurons//4, activation='relu')
            output_task = Dense(1, activation='linear', name='prediction')(output_task)
        else:
            output_task = self.layers[0].layers[-1].output

        self._layer_decoding(latent, nb_neurons=output_shape, activation='linear')

        projection = tf.keras.Model(inputs=latent, outputs=z)
        projection.summary()
        self.projection = self._assign_properties(projection)

        encoder = tf.keras.Model(inputs=inputs, outputs=latent)
        encoder.summary()
        self.encoder = self._assign_properties(encoder)

        decoder = tf.keras.Model(inputs=latent, outputs=decode)
        decoder.summary()
        self.decoder = self._assign_properties(decoder)

        task = tf.keras.Model(inputs=latent, outputs=output_task)
        task.summary()
        self.task = self._assign_properties(task)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.math.log(rho) - rho * tf.math.log(rho_hat) + (1 - rho) \
               * tf.math.log(1 - rho) - (1 - rho) * tf.math.log(1 - rho_hat)

    def unsupervised_step(self, xis, xjs, x_orig, temperature, rho = 0.05):

        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        negative_mask = get_negative_mask(tf.shape(xis)[0])

        with tf.GradientTape() as enc_tape,  tf.GradientTape() as dec_tape,  tf.GradientTape() as dist_tape:
            his, zis, x_reco_is, task_is = self.forward(x = xis, training =True)
            hjs, zjs, x_reco_js, task_js = self.forward(x = xjs, training=True)

            # normalize projection feature vectors
            zis = tf.math.l2_normalize(zis, axis=1)
            zjs = tf.math.l2_normalize(zjs, axis=1)

            l_pos = _cosine_simililarity_dim1(zis, zjs)
            l_pos = tf.reshape(l_pos, (tf.shape(l_pos)[0], 1))
            l_pos /= temperature

            negatives = tf.concat([zjs, zis], axis=0)

            loss_contrastive_ = 0
            for positives in [zis, zjs]:
                l_neg = _cosine_simililarity_dim2(positives, negatives)

                labels = tf.zeros(tf.shape(positives)[0], dtype=tf.int32)

                l_neg = tf.boolean_mask(l_neg, negative_mask)
                l_neg = tf.reshape(l_neg, (tf.shape(xis)[0], -1))
                l_neg /= temperature

                logits = tf.concat([l_pos, l_neg], axis=1)
                loss_contrastive_ += criterion(y_pred=logits, y_true=labels)

            loss_contrastive_ /= 2 * float(tf.shape(xis)[0])
            loss_contrastive = tf.reduce_mean(loss_contrastive_)
            loss_reconstruction = tf.reduce_mean((self.loss(x_orig, x_reco_is) + self.loss(x_orig, x_reco_js))/2)

            if rho:
                rho_is = tf.reduce_mean(his, axis=0)
                rho_js = tf.reduce_mean(hjs, axis=0)

                kl = (self.kl_divergence(rho, rho_is + 1e-10) + self.kl_divergence(rho, rho_js + 1e-10)) / 2
                kl = tf.reduce_sum(kl)
                loss_reconstruction += 0.1 * kl

            loss_distance = tf.reduce_mean(self.loss(zjs, zis))
            total_loss = loss_contrastive + loss_reconstruction + loss_distance

        gradients = enc_tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        gradients = dec_tape.gradient(loss_reconstruction, self.decoder.trainable_variables)
        self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        gradients = dist_tape.gradient(loss_contrastive, self.projection.trainable_variables)
        self.projection.optimizer.apply_gradients(zip(gradients, self.projection.trainable_variables))

        self.loss_metric.update_state(total_loss)
        self.encoder.loss_metric.update_state(loss_reconstruction)
        self.projection.loss_metric.update_state(loss_contrastive)
        self.task.loss_metric.update_state(loss_distance)

    def forward(self, x, training = True):
        h = self.encoder(x, training)
        z_ = self.projection(h, training)
        x_reco = self.decoder(h, training)
        task = self.task(h, training)
        return h, z_, x_reco, task

    def ssl_train_step(self, x, y_batch_train):

        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:

            h = self.encoder(x, training=True)
            y_pred = self.task(h, training = True)
            loss_task = tf.reduce_mean(self.loss(y_batch_train, y_pred))
            loss_task += (self.task.losses + self.encoder.losses)

        gradients = task_tape.gradient(loss_task, self.task.trainable_variables)
        self.task.optimizer.apply_gradients(zip(gradients, self.task.trainable_variables))

        gradients_enc = enc_tape.gradient(loss_task, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(zip(gradients_enc, self.encoder.trainable_variables))

        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred)

    def ssl_val_step(self, x, y_batch_train):

        h = self.encoder(x, training=False)
        y_pred = self.task(h, training = False)

        loss_task = self.loss(y_batch_train, y_pred)
        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred)

    def fit_unsupervised(self, x_dynamic, x_static, x_orig, model_directory,
                         batch_size=8, num_epochs=500, permut=True,
                         p_m=0.3, noise_level=0.1, temperature=1, rho = 0.05, **kwargs):

        train_loss = []

        self._init_models_cl(
            input_shape=np.concatenate([x_dynamic, x_static], axis = 1).shape[-1],
            output_shape=x_orig.shape[-1])


        for ep in range(num_epochs):
            x_dynamic_, x_static_ = shuffle(x_dynamic, x_static)
            train_ds = tf.data.Dataset.from_tensor_slices((x_dynamic_, x_static_, x_orig)).batch(batch_size)
            for x_dyn_batch_train, x_stat_batch_train, x_orig_batch in train_ds:
                xis = self.noise_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                           x_static=x_stat_batch_train.numpy(),
                                           p_m=p_m, noise_level=noise_level, permut=permut)

                xjs = self.noise_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                            x_static=x_stat_batch_train.numpy(),
                                            p_m=p_m, noise_level=noise_level, permut=permut)

                self.unsupervised_step(xis = xis, xjs = xjs,
                                       x_orig = x_orig_batch,
                                       temperature=temperature,
                                       rho= rho)

            loss_epoch = self.loss_metric.result().numpy()
            loss_reco = self.encoder.loss_metric.result().numpy()
            loss_task = self.task.loss_metric.result().numpy()
            loss_dist = self.projection.loss_metric.result().numpy()

            print('Epoch ' + str(ep) + ' : ' + str(loss_epoch) + '; reconstruction ' + str(loss_reco) + '; task ' + str(loss_task)+ '; distance ' + str(loss_dist))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.encoder.loss_metric.reset_states()
            self.projection.loss_metric.reset_states()
            self.task.loss_metric.reset_states()

        self.encoder.save_weights(os.path.join(model_directory, 'encoder_model'))
        self.decoder.save_weights(os.path.join(model_directory, 'decoder_model'))
        self.task.save_weights(os.path.join(model_directory, 'task_model'))
        for i in range(len(self.encoder.layers)):
            self.layers[0].layers[i].set_weights(self.encoder.layers[i].get_weights())
        self.save_weights(os.path.join(model_directory, 'model'))

        # History of the training
        losses = dict(train_loss_results=train_loss)
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

    def noise_generator(self, x_dynamic, x_static, p_m=0, noise_level=0, permut = False, **kwargs):
        # Get subset of features to create list of cropped data
        x_bar = np.concatenate([x_dynamic, x_static], axis = 1)
        # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
        x_bar_noisy = self.generate_noisy_xbar(x_bar, noise_level = noise_level)
        # Generate binary mask
        mask = np.random.binomial(1, p_m, x_bar.shape)

        # Replace selected x_bar features with the noisy ones
        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
        #Swap
        indices = np.random.RandomState(seed=0).permutation(x_bar.shape[1])
        x_bar_shifted = x_bar[:,indices]
        mask = np.random.binomial(1, p_m, x_bar.shape)
        if permut:
            x_bar = x_bar * (1 - mask) + x_bar_shifted * mask
        return x_bar


    def fit_task(self,
                 train_dataset, val_dataset, test_dataset, batch_size, num_epochs, model_directory,
                 save_steps=10, reduce_lr=True, patience=50,
                 finetuning=False, add_layer=False, unfreeze=False,
                 function=np.min,
                 **kwargs):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_dyn_train, x_stat_train, y_train = train_dataset
        x_dyn_val, x_stat_val, y_val = val_dataset
        x_dyn_test, x_stat_test, y_test = test_dataset

        dims = np.concatenate([x_dyn_train, x_stat_train], axis = 1).shape[-1]

        self._init_models_cl(input_shape=dims,
                              output_shape=dims,
                              add_layer = add_layer)

        _ = self.encoder(tf.zeros([0, dims]))
        _ = self.task(_)
        self.encoder.load_weights(os.path.join(model_directory, 'encoder_model'))

        try:
            self.task.load_weights(os.path.join(model_directory, 'task_model'))
        except:
            pass

        if finetuning:
            for i in range(len(self.encoder.layers)//2):
                self.encoder.layers[i].trainable = False

        val_ds = tf.data.Dataset.from_tensor_slices((x_dyn_val, x_stat_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_dyn_test, x_stat_test, y_test)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience//4, factor=0.5)
        wait = 0

        for epoch in range(num_epochs):
            if (
                patience
                and epoch >= patience
                and self.config.finetuning
                and unfreeze
            ):
                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].trainable = True

            x_dyn_train_, x_stat_train_, y_train_ = shuffle(x_dyn_train, x_stat_train, y_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_dyn_train_, x_stat_train_, y_train_)).batch(batch_size)

            for x_dyn_batch_train, x_stat_batch_train, y_batch_train in train_ds:
                xis = self.noise_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                            x_static=x_stat_batch_train.numpy(),
                                            p_m=0, noise_level=0, permut=False)

                self.ssl_train_step(x = xis,
                                    y_batch_train=y_batch_train)

            loss_epoch = self.loss_metric.result().numpy()
            train_acc_result = self.metric.result().numpy()
            print('Epoch ' + str(epoch) + ' : ' + str(loss_epoch))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.metric.reset_states()

            if epoch%save_steps ==0:
                wait +=1
                for x_dyn_batch_train, x_stat_batch_train, y_batch_train in val_ds:
                    x_tilde_val = self.noise_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                        x_static=x_stat_batch_train.numpy(),
                                                        p_m=0, noise_level=0, permut=False
                                                        )
                    self.ssl_val_step(x_tilde_val, y_batch_train)

                val_loss_epoch = self.loss_metric.result().numpy()
                val_loss.append(val_loss_epoch)

                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                for x_dyn_batch_train, x_stat_batch_train, y_batch_train in test_ds:
                    x_tilde_test = self.noise_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                         x_static=x_stat_batch_train.numpy(),
                                                         p_m=0, noise_level=0, permut=False)
                    self.ssl_val_step(x_tilde_test, y_batch_train)

                test_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                print('Val loss ' + str(epoch) + ' : ' + str(val_loss_epoch))
                print('Train acc ' + str(epoch) + ' : ' + str(val_acc_result))
                print('Val acc ' + str(epoch) + ' : ' + str(train_acc_result))
                print('Test acc ' + str(epoch) + ' : ' + str(test_acc_result))

                if (function is np.min and val_loss_epoch <= function(val_loss)
                        or function is np.max and val_loss_epoch >= function(val_loss)):
                    print('Best score seen so far ' + str(val_loss_epoch))
                    self.encoder.save_weights(os.path.join(model_directory, 'encoder_best_model'))
                    self.task.save_weights(os.path.join(model_directory, 'task_best_model'))

                if reduce_lr:
                    reduce_rl_plateau.on_epoch_end(wait, val_acc_result)

        self.encoder.save_weights(os.path.join(model_directory, 'encoder_last_model'))
        self.task.save_weights(os.path.join(model_directory, 'task_last_model'))











