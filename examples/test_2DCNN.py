import eoflow.models.tempnets_task.cnn_tempnets as cnn_tempnets
import tensorflow as tf

# Model configuration CNNLSTM
import numpy as np
import os
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from eoflow.models.data_augmentation import feature_noise, timeshift, noisy_label
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

########################################################################################################################
########################################################################################################################

def npy_concatenate(path, prefix = 'training_x'):
    path_npy = os.path.join(path, prefix)
    x_bands = np.load(path_npy + '_bands.npy')
    x_vis = np.load(path_npy  + '_vis.npy')
    return np.concatenate([x_bands, x_vis], axis = -1)

path = '/home/johann/Documents/Syngenta/Histograms/2020'
x_train = npy_concatenate(path, 'training_x')
x_train[np.isnan(x_train)] = 0

y_train = np.load(os.path.join(path, 'training_y.npy'))
x_val = npy_concatenate(path, 'val_x')
x_val[np.isnan(x_val)] = 0

y_val = np.load(os.path.join(path, 'val_y.npy'))

x_test = npy_concatenate(path, 'test_x')
x_test[np.isnan(x_test)] = 0
y_test = np.load(os.path.join(path, 'test_y.npy'))

x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)

np.mean(x_train.flatten())

# Model configuration CNN

# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-4,
    "keep_prob" : 0.5,
    "nb_conv_filters": 64,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 1024,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [3,3],
    "nb_conv_strides" : [1,1],
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "SAME",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'Flatten',
    "loss": "mse",
    "enumerate" : True,
    "metrics": 'r_square'
}



model_cnn = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
# Prepare the model (must be run before training)
model_cnn.prepare()
#model_cnn.build((None, 32, 30, 15))
#model_cnn(x_train)




model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 16,
    function = np.min,
    patience = 30,
    reduce_lr = False,
    pretraining = False,
    model_directory='/home/johann/Documents/model_hist',
)

