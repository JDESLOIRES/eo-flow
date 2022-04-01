import pandas as pd

import eoflow.models.tempnets_task.cnn_tempnets as cnn_tempnets
import eoflow.models.tempnets_task.cnn_tempnets_functional  as cnn_tempnets_functional
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


########################################################################################################################
########################################################################################################################

def reshape_array(x, T=30):
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x


def npy_concatenate(path, prefix='training_x', T=30):
    path_npy = os.path.join(path, prefix)
    '''

    x_bands = np.load(path_npy + '_bands.npy')
    x_bands = reshape_array(x_bands, T)
    x_vis = np.load(path_npy  + '_vis.npy')
    x_vis = reshape_array(x_vis, T)
    np.concatenate([x_bands, x_vis], axis = -1)
    '''
    x = np.load(path_npy + '_S2.npy')
    x = reshape_array(x, T)
    return x


#path = '/home/johann/Documents/Syngenta/cleaned_V2/2021'
path = '/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V3/cleaner_V2_training_10_folds/2021/fold_1'

x_train = npy_concatenate(path, 'training_x')
y_train = np.load(os.path.join(path, 'training_y.npy'))

x_val = npy_concatenate(path, 'val_x')
y_val = np.load(os.path.join(path, 'val_y.npy'))

x_test = npy_concatenate(path, 'test_x')
y_test = np.load(os.path.join(path, 'test_y.npy'))

# x_train = np.concatenate([x_train, x_val], axis = 0)
# y_train = np.concatenate([y_train, y_val], axis = 0)


'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=8)
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
model.fit(x_train, y_train)
preds = model.predict(x_test)
r2_score(y_test, preds)

'''

model_cfg_cnn_stride = {
    "learning_rate": 10e-4,
    "keep_prob": 0.65,  # should keep 0.8
    "nb_conv_filters": 32,  # wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons": 32,
    "nb_fc_stacks": 2,  # Nb FCN layers
    "fc_activation": 'relu',
    "kernel_size": 7,
    "padding": "CAUSAL",
    "emb_layer": 'GlobalAveragePooling1D',
    "enumerate": True,
    'str_inc': True,
    'fc_dec' : True,
    'ker_dec' : True,
    "metrics": "r_square",
    'factor' : 0.5,
    'adaptative' : False,
    "loss": "mse"  # huber was working great for 2020 and 2021
}

# console 1 et 3 : activation in the layer + flipout
# console 4 et 5 : activation outsie
# MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()
self = model_cnn
x = x_train


model_cnn.fit_dann(
    src_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    trgt_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size=12,
    patience=50,
    fillgaps=0,
    shift_step=0,
    sdev_label=0,
    feat_noise=0,
    reduce_lr=True,
    model_directory='/home/johann/Documents/model_16',
)



model_cnn.summary()

########################################################################################################################

