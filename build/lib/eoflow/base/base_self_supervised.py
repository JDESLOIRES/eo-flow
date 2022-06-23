from sklearn.utils import shuffle
import numpy as np
import os

import tensorflow as tf

from eoflow.models.data_augmentation import timeshift, feature_noise
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from eoflow.models.data_augmentation import data_augmentation
import pickle
import itertools
import helpers

import tensorflow as tf

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

    def _layer_decoding(self, net, nb_neurons, dropout_rate = 0.1, activation = 'linear'):

        layer_fcn = Dense(units=nb_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(activation)(layer_fcn)

        return layer_fcn

    def _init_ssl_models(self, input_shape = 262, output_shape = 450):
        _ = self(tf.zeros(list((1, input_shape))))

        inputs = self.layers[0].input

        latent = self.layers[0].layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4].output
        linear_layer1 = Dense(self.layers[0].layers[0].input.shape[-1], activation='linear', name='linear_1')(latent)
        z = Dense(output_shape, activation='relu', name='z')(linear_layer1)

        decode = self._layer_decoding(latent, nb_neurons = output_shape)

        output_task = self.layers[0].layers[-1].output

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
        return x * np.random.normal(1, noise_level, x.shape)

    @staticmethod
    def process_batch(xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""

        # Convert the batch to tensor and move it to where the model is
        #Xbatch = self._tensor(Xbatch)
        # Return batches
        return np.concatenate((xi, xj), axis=0)

    def subset_generator(self, x, n_subsets = 3, overlap = 0.75, p_m =0.3, noise_level = 0.15, mode="train"):

        n_column = x.shape[-1]

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
            x_bar = x[:, subset_column_idx]
            # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
            if noise_level:
                x_bar_noisy = self.generate_noisy_xbar(x_bar, noise_level = 0.1)
                # Generate binary mask
                mask = np.random.binomial(1, p_m, x_bar.shape)

                # Replace selected x_bar features with the noisy ones
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask

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

    def pretrain_step(self, concatenated_subsets_list, x_orig, temperature =1):

        loss_contrastive = 0
        loss_reconstruction = 0
        loss_distance = 0
        total_loss = 0

        for el in concatenated_subsets_list:
            xis, xjs = el[:el.shape[0]//2], el[el.shape[0]//2:]
            negative_mask = get_negative_mask(tf.shape(xis)[0])

            with tf.GradientTape(persistent=True) as tape:
                his, zis, x_reco_is, task_is = self.forward(x = xis, training = True)
                hjs, zjs, x_reco_js, task_js = self.forward(x=xjs, training=True)

                # normalize projection feature vectors
                zis = tf.math.l2_normalize(zis, axis=1)
                zjs = tf.math.l2_normalize(zjs, axis=1)

                l_pos = _cosine_simililarity_dim1(zis, zjs)
                l_pos = tf.reshape(l_pos, (tf.shape(l_pos)[0], 1))
                l_pos /= temperature

                negatives = tf.concat([zjs, zis], axis=0)

                criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                          reduction=tf.keras.losses.Reduction.SUM)

                for positives in [zis, zjs]:
                    l_neg = _cosine_simililarity_dim2(positives, negatives)

                    labels = tf.zeros(tf.shape(positives)[0], dtype=tf.int32)

                    l_neg = tf.boolean_mask(l_neg, negative_mask)
                    l_neg = tf.reshape(l_neg, (tf.shape(xis)[0], -1))
                    l_neg /= temperature

                    logits = tf.concat([l_pos, l_neg], axis=1)
                    loss_contrastive += criterion(y_pred=logits, y_true=labels)

                loss_contrastive /= 2 * float(tf.shape(xis)[0])
                loss_reconstruction += (self.loss(x_orig, x_reco_is) + self.loss(x_orig, x_reco_js))/2
                loss_distance += self.loss(zjs, zis)
                total_loss += (loss_contrastive + loss_reconstruction + loss_distance)

        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        gradients = tape.gradient(loss_reconstruction, self.decoder.trainable_variables)
        self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        gradients = tape.gradient(loss_distance, self.projection.trainable_variables)
        self.projection.optimizer.apply_gradients(zip(gradients, self.projection.trainable_variables))
        self.loss_metric.update_state(tf.reduce_mean(total_loss))


    def fit_pretrain(self,
                     x_train,
                     model_directory,
                     batch_size =8,
                     num_epochs = 500,
                     n_subsets=3, overlap=0.75,
                     p_m=0.3, noise_level=0.15):

        train_loss = []
        n_column = x_train.shape[-1]

        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        stop_idx = n_column_subset + n_overlap
        self._init_ssl_models(input_shape=stop_idx, output_shape=450)

        for ep in range(num_epochs):
            x_train_ = shuffle(x_train)
            train_ds = tf.data.Dataset.from_tensor_slices(x_train_).batch(batch_size)
            for x_batch_train in train_ds:
                x_tilde_list = self.subset_generator(x_batch_train.numpy(),
                                                     n_subsets = n_subsets, overlap = overlap,
                                                     p_m =p_m, noise_level = noise_level, mode="train")

                concatenated_subsets_list = self.get_combinations_of_subsets(x_tilde_list)
                self.pretrain_step( concatenated_subsets_list, x_orig = x_batch_train, temperature=1)

            loss_epoch = self.loss_metric.result().numpy()
            print('Epoch ' + str(ep) + ' : ' + str(loss_epoch))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

        self.encoder.save_weights(os.path.join(model_directory, 'encoder_model'))
        self.decoder.save_weights(os.path.join(model_directory, 'decoder_model'))

        # History of the training
        losses = dict(train_loss_results=train_loss)
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)
















