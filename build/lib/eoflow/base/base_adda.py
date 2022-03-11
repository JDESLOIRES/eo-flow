import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pickle
import os

import tensorflow as tf

from . import Configurable
from eoflow.base.base_callbacks import CustomReduceLRoP
from eoflow.models.data_augmentation import data_augmentation, timeshift, feature_noise
from tensorflow.keras.layers import Dense
from .base_custom_training import BaseModelCustomTraining
from keras.models import clone_model

class BaseModelAdapt(BaseModelCustomTraining):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)
        self.eps = 1e-8

    def _init_models(self, x, path_pretrain):

        _ = self(tf.zeros(list(x.shape)))

        if path_pretrain:
            self.load_weights(path_pretrain)

        inputs = self.layers[0].input
        encode = self.layers[0].layers[self.config.nb_conv_stacks * 4 + 1].output
        dense_layers = self.layers[0].layers[-2].input
        output_discriminator =  Dense(2, activation='sigmoid', name='Discriminator')(dense_layers)
        output_task =  self.layers[0].layers[-1].output

        return inputs, encode, dense_layers, output_discriminator, output_task

    def _assign_properties(self, model):
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.loss_metric = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        return model

    def _get_encoder(self, x, path_pretrain = None):
        inputs, encode, _, _, _ = self._init_models(x, path_pretrain)
        encoder = tf.keras.Model(inputs=inputs, outputs=encode)

        if path_pretrain:
            encoder.load_weights(path_pretrain, by_name=True)

        encoder = self._assign_properties(encoder)
        return encoder


    def _get_discriminator(self, x, path_pretrain = None):
        _, encode, _, output_discriminator, _ = self._init_models(x, path_pretrain)
        discriminator = tf.keras.Model(inputs=encode, outputs=output_discriminator)
        if path_pretrain:
            discriminator.load_weights(path_pretrain, by_name=True)
        discriminator = self._assign_properties(discriminator)
        discriminator.loss = tf.keras.losses.BinaryCrossentropy()

        return discriminator

    def pretrainstep(self,
                     source_dataset,
                     encoder_src,
                     num_epochs,
                     model_directory):

        for Xs, ys in source_dataset:
            Xs, ys = shuffle(Xs, ys)
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:
                # Forward pass
                Xs_enc = encoder_src(Xs, training=True)
                ys_pred = self(Xs_enc, training=True)
                # Reshape
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                # Compute the loss value
                loss = self.loss(ys, ys_pred)
                cost = tf.reduce_mean(loss)

            # Compute gradients
            trainable_vars_task = self.trainable_variables
            trainable_vars_enc = encoder_src.trainable_variables
    
            gradients_task = task_tape.gradient(cost, trainable_vars_task)
            gradients_enc = enc_tape.gradient(cost, trainable_vars_enc)
    
            # Update weights
            self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
            encoder_src.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
    
            # Update metrics
            self.compiled_metrics.update_state(ys, ys_pred)
            self.compiled_loss(ys, ys_pred)


    def train_target_discriminator(self, x_source, x_target,
                                   model_directory,num_epochs =  500):
        #Initialize models with inputs
        source_encoder = self._get_encoder(x_source, model_directory)
        for layer in self.source_encoder.layers:
            layer.trainable = False
        target_encoder = self._get_encoder(x_target)
        discriminator = self._get_discriminator(x_source)
        #Then,during training, discriminator will see encoded source and target and must classify it

        y_source = np.zeros_like(x_source.shape[0])
        y_target = np.ones_like(x_target.shape[0])








