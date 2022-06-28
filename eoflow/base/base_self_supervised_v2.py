from sklearn.utils import shuffle
import numpy as np
import os

from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from .base_self_supervised import BaseModelSelfTraining
from eoflow.models.data_augmentation import data_augmentation
import pickle
import itertools
import helpers

import tensorflow as tf
from sklearn.metrics import r2_score

class BaseModelSelfTrainingV2(BaseModelSelfTraining):
    def __init__(self, config_specs):
        BaseModelSelfTrainingV2.__init__(self, config_specs)

    def _init_ssl_models(self, input_shape, output_shape, add_layer = False):

        _ = self(tf.zeros(list((1, input_shape))))
        inputs = self.layers[0].input

        latent = self.layers[0].layers[(self.config.nb_fc_stacks - self.config.layer_before) * 4].output
        linear_layer1 = Dense(self.layers[0].layers[0].input.shape[-1], activation=None, name='linear_1')(latent)
        z = tf.keras.layers.Activation('relu')(linear_layer1)
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
        return x * np.random.normal(1, noise_level, x.shape)

    @staticmethod
    def process_batch(xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""

        # Convert the batch to tensor and move it to where the model is
        #Xbatch = self._tensor(Xbatch)
        # Return batches
        return np.concatenate((xi, xj), axis=0)

    def subset_generator(self, x, n_subsets = 3, overlap = 0.75, p_m =0, noise_level = 0, mode="train"):

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
                x_bar_noisy = self.generate_noisy_xbar(x_bar, noise_level = noise_level)
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


    def pretrain_step(self, concatenated_subsets_list, x_orig, kd, temperature , lambda_):

        loss_contrastive = 0
        loss_reconstruction = 0
        loss_distance = 0
        loss_task = 0
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

                criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

                if len(set(list(kd.flatten())))>1:
                    loss_task += lambda_ * (self.loss(kd, task_js) + self.loss(kd, task_is)) / 2
                    total_loss += loss_task

        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        gradients = tape.gradient(loss_reconstruction, self.decoder.trainable_variables)
        self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        gradients = tape.gradient(loss_distance, self.projection.trainable_variables)
        self.projection.optimizer.apply_gradients(zip(gradients, self.projection.trainable_variables))

        gradients = tape.gradient(loss_task, self.task.trainable_variables)
        self.task.optimizer.apply_gradients(zip(gradients, self.task.trainable_variables))

        self.loss_metric.update_state(tf.reduce_mean(total_loss))
        self.encoder.loss_metric.update_state(tf.reduce_mean(loss_reconstruction))
        self.projection.loss_metric.update_state(tf.reduce_mean(loss_contrastive))
        self.task.loss_metric.update_state(tf.reduce_mean(loss_task))

    def _get_weights(self, x_train, y_train, n_subsets, overlap):

        x_tilde = self.subset_generator(x_train,
                                        n_subsets=n_subsets, overlap=overlap,
                                        mode="test")

        h_tilde = [self.encoder(x, training=False) for x in x_tilde]
        w_loss = [tf.reduce_mean(self.loss(y_train, self.task(h))).numpy() for h in h_tilde]
        w_loss /= np.sum(w_loss)
        w_loss = 1 / w_loss
        w_loss /= np.sum(w_loss)

        return w_loss


    def subtab_train_step(self, x_tilde, y_batch_train):

        with tf.GradientTape(persistent=True) as tape:
            h_tilde = [self.encoder(x, training = True) for x in x_tilde]
            w_loss = [tf.reduce_mean(self.loss(y_batch_train, self.task(h))).numpy() for h in h_tilde]
            w_loss /= np.sum(w_loss)
            w_loss = 1/w_loss
            w_loss /= np.sum(w_loss)

            ###########
            #h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]
            #h = tf.math.reduce_sum(h_tilde, 0)
            h = h_tilde[np.argmax(w_loss)]

            y_pred = self.task(h, training = True)
            loss_task = tf.reduce_mean(self.loss(y_batch_train, y_pred))

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

        #h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]
        #h = tf.math.reduce_sum(h_tilde, 0)
        h = h_tilde[np.argmax(w_loss)]
        y_pred = self.task(h, training = False)

        loss_task = self.loss(y_batch_train, y_pred)
        self.loss_metric.update_state(tf.reduce_mean(loss_task))
        self.metric.update_state(tf.reshape(y_batch_train[:, 0], tf.shape(y_pred)), y_pred)

    def subtab_pred_step(self,
                         x_test_,
                         model_directory,
                         permut = True,
                         n_subsets=3, overlap=0.75,
                         w_loss = None
                         ):

        import copy
        x_test = copy.deepcopy(x_test_)
        indices = np.random.RandomState(seed=0).permutation(x_test.shape[1])
        if permut:
            x_test = x_test[:,indices]

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
                                        mode="test")

        h_tilde = [self.encoder(x, training=False) for x in x_tilde]
        if w_loss is None:
            w_loss = [1 for _ in h_tilde]

        h_tilde = [tf.multiply(h, w) for h, w in zip(h_tilde, w_loss)]
        h = tf.math.reduce_sum(h_tilde, 0)


        return self.task(h, training=False)


    def fit_pretrain(self,
                     x_train,
                     model_directory,
                     batch_size =8,
                     num_epochs = 500,
                     permut = True,
                     n_subsets=3, overlap=0.75,
                     p_m=0.3, noise_level=0.15,
                     temperature = 1,
                     model_kd = None,
                     lambda_ = 1):

        train_loss = []

        n_column = x_train.shape[-1]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        stop_idx = n_column_subset + n_overlap
        self._init_ssl_models(input_shape=stop_idx, output_shape=x_train.shape[-1])

        if model_kd is None:
            y_preds = np.zeros((x_train.shape[0], 1))

        elif model_kd.config.adaptative:
            y_preds, _ = model_kd.predict(x_train)
        else:
            y_preds = model_kd.predict(x_train)

        indices = np.random.RandomState(seed=0).permutation(x_train.shape[1])

        if permut:
            x_train = x_train[:,indices]

        for ep in range(num_epochs):
            x_train_ = shuffle(x_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_preds)).batch(batch_size)
            for x_batch_train, y_batch_train in train_ds:
                x_tilde_list = self.subset_generator(x_batch_train.numpy(),
                                                     n_subsets = n_subsets, overlap = overlap,
                                                     p_m =p_m, noise_level = noise_level, mode="train")

                concatenated_subsets_list = self.get_combinations_of_subsets(x_tilde_list)
                self.pretrain_step(concatenated_subsets_list,
                                   x_orig = x_batch_train.numpy(),
                                   temperature=temperature,
                                   kd = y_batch_train.numpy(),
                                   lambda_ = lambda_)

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
                       permut = True,
                       finetuning = False,
                       add_layer = False,
                       function = np.min):

        global val_acc_result
        train_loss, val_loss, val_acc = ([np.inf] if function == np.min else [-np.inf] for i in range(3))

        x_train, y_train = train_dataset
        x_val, y_val = val_dataset
        x_test, y_test = test_dataset

        indices = np.random.RandomState(seed=0).permutation(x_train.shape[1])
        if permut:
            x_train = x_train[:,indices]
            x_val = x_val[:, indices]
            x_test = x_test[:, indices]

        train_loss = []

        n_column = x_train.shape[-1]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        stop_idx = n_column_subset + n_overlap

        self._init_ssl_models(input_shape=stop_idx, output_shape=x_train.shape[-1], add_layer = add_layer)
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

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        reduce_rl_plateau = self._reduce_lr_on_plateau(patience=patience//4, factor=0.5)
        wait = 0

        for epoch in range(num_epochs):
            if patience and epoch >= patience:
                if self.config.finetuning:
                    for i in range(len(self.encoder.layers)):
                        self.encoder.layers[i].trainable = True

            x_train_, y_train_ = shuffle(x_train, y_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train_, y_train_)).batch(batch_size)
            for x_batch_train, y_batch_train in train_ds:
                x_tilde_train = self.subset_generator(x_batch_train.numpy(),
                                                      n_subsets=n_subsets, overlap=overlap,
                                                      p_m=p_m, noise_level=noise_level, mode="test")
                #kd = model_kd(x_batch_train) if model_kd else None
                self.subtab_train_step(x_tilde=x_tilde_train, y_batch_train=y_batch_train)

            loss_epoch = self.loss_metric.result().numpy()
            train_acc_result = self.metric.result().numpy()
            print('Epoch ' + str(epoch) + ' : ' + str(loss_epoch))
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()
            self.metric.reset_states()

            if epoch%save_steps ==0:
                wait +=1
                w_h = self._get_weights(x_train_, y_train_, n_subsets, overlap)
                for x_batch_train, y_batch_train in val_ds:
                    x_tilde_val = self.subset_generator(x_batch_train.numpy(),
                                                        n_subsets=n_subsets, overlap=overlap,
                                                        mode="test")
                    self.subtab_val_step(x_tilde_val, y_batch_train, w_loss=w_h)

                val_loss_epoch = self.loss_metric.result().numpy()
                val_loss.append(val_loss_epoch)

                val_acc_result = self.metric.result().numpy()
                self.loss_metric.reset_states()
                self.metric.reset_states()

                for x_batch_train, y_batch_train in test_ds:
                    x_tilde_test = self.subset_generator(x_batch_train.numpy(),
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












