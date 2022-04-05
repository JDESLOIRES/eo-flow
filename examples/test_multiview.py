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
from importlib import reload
cnn_tempnets = reload(cnn_tempnets)

########################################################################################################################
########################################################################################################################

def npy_concatenate(path, prefix = 'training_x'):
    path_npy = os.path.join(path, prefix)
    x_bands = np.load(path_npy + '_bands.npy')
    x_vis = np.load(path_npy  + '_vis.npy')
    return np.concatenate([x_bands, x_vis], axis = -1)


def ts_concatenate(path, prefix='training_x', T=30):
    def reshape_array(x, T=30):
        x = x.reshape(x.shape[0], x.shape[1] // T, T)
        x = np.moveaxis(x, 2, 1)
        return x
    path_npy = os.path.join(path, prefix)
    x = np.load(path_npy + '_S2.npy')
    x = reshape_array(x, T)
    return x

path = '/home/johann/Documents/Syngenta/Histograms/2020'

x_train = npy_concatenate(path, 'training_x')
x_1 = x_train
x_val = npy_concatenate(path, 'val_x')
x_test = npy_concatenate(path, 'test_x')

x_train[np.isnan(x_train)] = 0
x_val[np.isnan(x_val)] = 0
x_test[np.isnan(x_test)] = 0


# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-4,
    "keep_prob" : 0.5,
    "nb_conv_filters": 16,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 128,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [2,2],
    "kernel_initializer" : 'he_normal',
    "fc_activation" : 'relu',
    "batch_norm": True,
    'emb_layer' : 'GlobalAveragePooling2D',
    "padding": "SAME",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "loss": "mse",
    "enumerate" : True,
    "metrics": 'r_square'
}


model_view_1 = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
model_view_1.prepare()
self = model_view_1

model_cfg_cnn_stride = {
    "learning_rate": 10e-3,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 32, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 32,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 7,
    "n_strides" :1,
    "padding": "CAUSAL",
    "emb_layer" : 'GlobalAveragePooling1D',
    "enumerate" : True,
    'str_inc' : True,
    'batch_norm' : True,
    "metrics": "r_square",
    'ker_dec' : True,
    'fc_dec' : True,
    #"activity_regularizer" : 1e-4,
    "loss": "mse",
    'multioutput' : False
}

model_view_2 = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# pare the model (must be run before training)
model_view_2.prepare()

path = '/home/johann/Documents/Syngenta/cleaned_V2/2021'
x_train_2 = ts_concatenate(path, 'training_x')
x_val = ts_concatenate(path, 'val_x')
x_test = ts_concatenate(path, 'test_x')



