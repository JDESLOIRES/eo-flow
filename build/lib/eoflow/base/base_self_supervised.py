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

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
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


class BaseModelSelfTraining(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)

    def _layer_decoding(self, net, nb_neurons, activation = 'linear'):

        layer_fcn = Dense(units=nb_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          activation = activation,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        #if self.config.batch_norm:layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        #layer_fcn = tf.keras.layers.Dropout(1 - self.config.keep_prob)(layer_fcn)
        #layer_fcn = tf.keras.layers.Activation(activation)(layer_fcn)

        return layer_fcn

    def _init_ssl_models(self, input_shape, output_shape, add_layer = False):

        _ = self(tf.zeros(list((1, input_shape))))
        inputs = self.layers[0].input

        latent = self.layers[0].layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4].output

        linear_layer1 = Dense(self.layers[0].layers[0].input.shape[-1], activation=None, name='linear_1')(latent)
        z = tf.keras.layers.LeakyReLU()(linear_layer1)
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


    @staticmethod
    def generate_noisy_xbar( x, noise_level = 0.1):
        no, dim = x.shape
        x_bar = x + np.random.normal(0, noise_level, x.shape)
        for i in range(dim):
            idx = np.random.permutation(no)
            x_bar[:, i] = x[idx, i]
        return x_bar

    @staticmethod
    def process_batch(xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""

        # Convert the batch to tensor and move it to where the model is
        #Xbatch = self._tensor(Xbatch)
        # Return batches
        return np.concatenate((xi, xj), axis=0)

    def subset_generator(self,
                         x_dynamic,
                         x_static,
                         n_subsets = 3, overlap = 0.75, p_m =0,
                         noise_level = 0, swap = False, mode="train"):

        n_column = x_dynamic.shape[-1]

        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)

        # Get the range over the number of features
        column_idx = list(range(n_column))
        # Permute the order of subsets to avoid any bias during training. The order is unchanged at the test time.
        permuted_order = np.random.permutation(n_subsets) if mode == "train" else range(n_subsets)
        # Pick subset of columns (equivalent of cropping)
        subset_column_idx_list = []

        # Generate subsets.
        for i in permuted_order:
            # If subset is in skip, don't include it in training. Otherwise, continue.
            if i == 0:
                start_idx = 0
                stop_idx = n_column_subset + n_overlap
            else:
                start_idx = i * n_column_subset - n_overlap
                stop_idx = (i + 1) * n_column_subset
            # Get the subset
            subset_column_idx_list.append(column_idx[start_idx:stop_idx])

        # Add a dummy copy if there is a single subset
        if len(subset_column_idx_list) == 1:
            subset_column_idx_list.append(subset_column_idx_list[0])

        # Get subset of features to create list of cropped data
        x_tilde_list = []
        for subset_column_idx in subset_column_idx_list:
            x_bar = np.concatenate([x_dynamic[:, subset_column_idx], x_static], axis = 1)

            # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
            if noise_level:
                x_bar_noisy = self.generate_noisy_xbar(x_bar, noise_level = noise_level)
                # Generate binary mask
                mask = np.random.binomial(1, p_m, x_bar.shape)
                # Replace selected x_bar features with the noisy ones
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
            if swap:
                indices = np.random.RandomState(seed=0).permutation(x_bar.shape[1])
                x_bar_shifted = x_bar[:, indices]
                mask = np.random.binomial(1, p_m, x_bar.shape)
                x_bar = x_bar * (1 - mask) + x_bar_shifted * mask

            # Add the subset to the list
            x_tilde_list.append(x_bar)

        return x_tilde_list


    def get_combinations_of_subsets(self, x_tilde_list):

        # Compute combinations of subsets [(x1, x2), (x1, x3)...]
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        # List to store the concatenated subsets
        concatenated_subsets_list = []

        # Go through combinations
        for (xi, xj) in subset_combinations:
            # Concatenate xi, and xj, and turn it into a tensor
            Xbatch = self.process_batch(xi, xj)
            # Add it to the list
            concatenated_subsets_list.append(Xbatch)

        # Return the list of combination of subsets
        return concatenated_subsets_list


    def forward(self, x, training = True):
        h = self.encoder(x, training)
        z_ = self.projection(h, training)
        x_reco = self.decoder(h, training)
        task = self.task(h, training)
        return h, z_, x_reco, task


    def pretrain_step(self, concatenated_subsets_list, x_orig, temperature, rho = 0 ):

        loss_contrastive = 0
        loss_reconstruction = 0
        loss_distance = 0
        loss_task = 0

        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for el in concatenated_subsets_list:
            xis, xjs = el[:el.shape[0]//2], el[el.shape[0]//2:]
            negative_mask = get_negative_mask(tf.shape(xis)[0])

            with tf.GradientTape(persistent=True) as tape:
                his, zis, x_reco_is, task_is = self.forward(x = xis, training = True)
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
                loss_contrastive += tf.reduce_mean(loss_contrastive_)
                loss_reconstruction += tf.reduce_mean((self.loss(x_orig, x_reco_is) + self.loss(x_orig, x_reco_js))/2)

                if rho:
                    rho_is = tf.reduce_mean(his, axis=0)
                    rho_js = tf.reduce_mean(hjs, axis=0)

                    kl = (self.kl_divergence(rho, rho_is + 1e-10) + self.kl_divergence(rho, rho_js + 1e-10)) / 2
                    kl = tf.reduce_sum(kl)
                    loss_reconstruction += 0.1 * kl

                loss_distance += tf.reduce_mean(self.loss(zjs, zis))

                total_loss = loss_contrastive + loss_reconstruction + loss_distance

        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        gradients = tape.gradient(loss_reconstruction, self.decoder.trainable_variables)
        self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        gradients = tape.gradient(loss_contrastive, self.projection.trainable_variables)
        self.projection.optimizer.apply_gradients(zip(gradients, self.projection.trainable_variables))

        if loss_task>0:
            gradients = tape.gradient(loss_task, self.task.trainable_variables)
            self.task.optimizer.apply_gradients(zip(gradients, self.task.trainable_variables))
            self.task.loss_metric.update_state(tf.reduce_mean(loss_task))

        self.loss_metric.update_state(total_loss)
        self.encoder.loss_metric.update_state(loss_reconstruction)
        self.projection.loss_metric.update_state(loss_contrastive)
        self.task.loss_metric.update_state(loss_distance)


    def _get_weights(self, x_dyn_train, x_stat_train, y_train, n_subsets, overlap, swap = False):

        x_tilde = self.subset_generator(x_dyn_train,
                                        x_stat_train,
                                        n_subsets=n_subsets, overlap=overlap,
                                        swap = swap,
                                        mode="test")

        h_tilde = [self.encoder(x, training=False) for x in x_tilde]
        w_loss = [tf.reduce_mean(self.loss(y_train, self.task(h))).numpy() for h in h_tilde]
        w_loss /= np.sum(w_loss)
        w_loss = 1 / w_loss
        w_loss /= np.sum(w_loss)

        return w_loss


    def subtab_train_step(self, x_tilde, y_batch_train, kd, lambda_ = 1):

        with tf.GradientTape(persistent=True) as tape:
            h_tilde = [self.encoder(x, training = True) for x in x_tilde]

            w_loss = [tf.reduce_mean(self.loss(y_batch_train, self.task(h))).numpy() for h in h_tilde]
            w_loss /= np.sum(w_loss)
            w_loss = 1/w_loss
            w_loss /= np.sum(w_loss)
            ###########
            h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]
            #h = h_tilde[np.argmax(w_loss)]
            h = tf.math.reduce_sum(h_tilde, 0)

            y_pred = self.task(h, training = True)
            loss_task = tf.reduce_mean(self.loss(y_batch_train, y_pred))
            if len(set(list(kd.flatten()))) > 1:
                loss_task += tf.reduce_mean(lambda_ * self.loss(kd, y_pred))

        gradients = tape.gradient(loss_task, self.task.trainable_variables)
        self.task.optimizer.apply_gradients(zip(gradients, self.task.trainable_variables))

        gradients = tape.gradient(loss_task, self.encoder.trainable_variables)

        self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred)

    def subtab_val_step(self, x_tilde, y_batch_train, w_loss = None):

        h_tilde = [self.encoder(x, training=False) for x in x_tilde]

        if w_loss is None:
            w_loss = [1 for _ in h_tilde]

        h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]
        #h = h_tilde[np.argmax(w_loss)]

        h = tf.math.reduce_sum(h_tilde, 0)

        y_pred = self.task(h, training = False)

        loss_task = self.loss(y_batch_train, y_pred)
        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred)

    def subtab_pred_step(self,
                         x_test_,
                         model_directory,
                         n_subsets=3, overlap=0.75,
                         w_loss = None
                         ):

        import copy
        x_test = copy.deepcopy(x_test_)

        n_column = x_test.shape[-1]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        stop_idx = n_column_subset + n_overlap
        self._init_ssl_models(input_shape=stop_idx, output_shape=x_test.shape[-1])
        self.encoder.load_weights(os.path.join(model_directory, 'encoder_best_model'))
        self.task.load_weights(os.path.join(model_directory, 'task_best_model'))

        x_tilde = self.subset_generator(x_test,
                                        n_subsets=n_subsets, overlap=overlap,
                                        mode="test", swap = False)

        h_tilde = [self.encoder(x, training=False) for x in x_tilde]
        if w_loss is None:
            w_loss = [1 for _ in h_tilde]
        #h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]

        h = tf.math.reduce_max(h_tilde, 0)

        return self.task(h, training=False)


    def fit_pretrain(self,
                     x_dynamic,
                     x_static,
                     model_directory,
                     batch_size =8,
                     num_epochs = 500,
                     n_subsets=3, overlap=0.75,
                     p_m=0.3, noise_level=0.15,
                     temperature = 1,
                     model_kd = None,
                     swap = False,
                     rho = 0):

        train_loss = []

        n_column = x_dynamic.shape[-1]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        if x_static is not None:
            stop_idx = n_column_subset + n_overlap + x_static.shape[1]
            self._init_ssl_models(input_shape=stop_idx,
                                  output_shape=np.concatenate([x_dynamic, x_static], axis = 1).shape[-1])
        else:
            stop_idx = n_column_subset + n_overlap
            self._init_ssl_models(input_shape=stop_idx,
                                  output_shape=x_dynamic.shape[-1])

        if model_kd is None:
            y_preds = np.zeros((x_dynamic.shape[0], 1))
        elif model_kd.config.adaptative:
            y_preds, h_latent = model_kd.predict(np.concatenate([x_dynamic, x_static], axis = 1))
        else:
            y_preds = model_kd.predict(np.concatenate([x_dynamic, x_static], axis = 1))

        for ep in range(num_epochs):
            x_dynamic_, x_static_, y_preds_ = shuffle(x_dynamic, x_static, y_preds)
            train_ds = tf.data.Dataset.from_tensor_slices((x_dynamic_, x_static_, y_preds_)).batch(batch_size)
            for x_dyn_batch_train, x_stat_batch_train, y_batch_train in train_ds:
                x_tilde_list = self.subset_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                     x_static=x_stat_batch_train.numpy(),
                                                     n_subsets = n_subsets, overlap = overlap,
                                                     swap = swap,
                                                     p_m =p_m, noise_level = noise_level, mode="train")

                concatenated_subsets_list = self.get_combinations_of_subsets(x_tilde_list)
                self.pretrain_step(concatenated_subsets_list,
                                   x_orig = np.concatenate([x_dyn_batch_train.numpy(), x_stat_batch_train.numpy()], axis = 1),
                                   temperature=temperature,
                                   rho = rho)

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

        # History of the training
        losses = dict(train_loss_results=train_loss)
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)

    def fit_supervised(self,
                       train_dataset,
                       val_dataset,
                       test_dataset,
                       batch_size,
                       num_epochs,
                       model_directory,
                       save_steps = 10,
                       n_subsets=3, overlap=0.75,
                       p_m =0.3, noise_level=0.15,
                       reduce_lr = True,
                       patience = 50,
                       finetuning = False,
                       add_layer = False,
                       unfreeze = False,
                       model_kd = None,
                       lambda_ = 1,
                       function = np.min):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_dyn_train, x_stat_train, y_train = train_dataset
        x_dyn_val, x_stat_val, y_val = val_dataset
        x_dyn_test, x_stat_test, y_test = test_dataset

        if model_kd is None:
            y_preds = np.zeros((x_dyn_train.shape[0], 1))
        elif model_kd.config.adaptative:
            y_preds, h_latent = model_kd.predict(np.concatenate([x_dyn_train, x_stat_train], axis = 1))
        else:
            y_preds = model_kd.predict(np.concatenate([x_dyn_train, x_stat_train], axis = 1))

        train_loss = []

        n_column = x_dyn_train.shape[-1]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        stop_idx = n_column_subset + n_overlap + x_stat_train.shape[-1]

        self._init_ssl_models(input_shape=stop_idx,
                              output_shape=np.concatenate([x_dyn_train, x_stat_train], axis = 1).shape[-1],
                              add_layer = add_layer)

        _ = self.encoder(tf.zeros([0, stop_idx]))
        _ = self.task(_)
        self.encoder.load_weights(os.path.join(model_directory, 'encoder_model'))

        try:
            self.task.load_weights(os.path.join(model_directory, 'task_model'))
        except:
            pass

        if finetuning:
            for i in range(len(self.encoder.layers)):
                self.encoder.layers[i].trainable = False

        val_ds = tf.data.Dataset.from_tensor_slices((x_dyn_val, x_stat_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_dyn_test, x_stat_test, y_test)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience, factor=0.5)
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

            x_dyn_train_, x_stat_train_, y_train_, y_preds_ = shuffle(x_dyn_train, x_stat_train, y_train, y_preds)
            train_ds = tf.data.Dataset.from_tensor_slices((x_dyn_train_, x_stat_train_, y_train_, y_preds_)).batch(batch_size)

            for x_dyn_batch_train, x_stat_batch_train, y_batch_train, y_preds_batch in train_ds:
                x_tilde_train = self.subset_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                      x_static=x_stat_batch_train.numpy(),
                                                      n_subsets=n_subsets, overlap=overlap,
                                                      p_m=p_m, noise_level=noise_level, mode="test")

                #kd = model_kd(x_batch_train) if model_kd else None
                self.subtab_train_step(x_tilde=x_tilde_train,
                                       y_batch_train=y_batch_train,
                                       kd = y_preds_batch.numpy(),
                                       lambda_=lambda_)

            loss_epoch = self.loss_metric.result().numpy()
            train_acc_result = self.metric.result().numpy()
            print('Epoch ' + str(epoch) + ' : ' + str(loss_epoch))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.metric.reset_states()

            if epoch%save_steps ==0:
                wait +=1
                w_h = self._get_weights(x_dyn_train_, x_stat_train_, y_train_, n_subsets, overlap)
                for x_dyn_batch_train, x_stat_batch_train, y_batch_train in val_ds:
                    x_tilde_val = self.subset_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                        x_static=x_stat_batch_train.numpy(),
                                                        n_subsets=n_subsets, overlap=overlap,
                                                        mode="test")
                    self.subtab_val_step(x_tilde_val, y_batch_train, w_loss=w_h)

                val_loss_epoch = self.loss_metric.result().numpy()
                val_loss.append(val_loss_epoch)

                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                for x_dyn_batch_train, x_stat_batch_train, y_batch_train in test_ds:
                    x_tilde_test = self.subset_generator(x_dynamic=x_dyn_batch_train.numpy(),
                                                         x_static=x_stat_batch_train.numpy(),
                                                         n_subsets=n_subsets, overlap=overlap,
                                                         mode="test")
                    self.subtab_val_step(x_tilde_test, y_batch_train, w_loss=w_h)

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
