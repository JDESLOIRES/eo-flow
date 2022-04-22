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

def npy_concatenate(path, prefix = 'training_x', suffix = None, T= 30):
    path_npy = os.path.join(path, prefix)
    def reshape_array(x, T=30):
        x = x.reshape(x.shape[0], x.shape[1] // T, T)
        x = np.moveaxis(x, 2, 1)
        return x
    if not suffix:
        x_bands = np.load(path_npy + '_bands.npy')
        x_vis = np.load(path_npy + '_' + 'vis' + '.npy')
        return  reshape_array(np.concatenate([x_bands, x_vis], axis = -1), T)
    else:
        return reshape_array(np.load(path_npy + '_' + suffix + '.npy'), T)



year = 2021
path = '/home/johann/Documents/Syngenta/test_multiview/3D_' + str(year) + '_W/fold_1/'

x_train = npy_concatenate(path, 'training_x')
x_val = npy_concatenate(path, 'val_x')
x_test = npy_concatenate(path, 'test_x')


model_cfg_cnn_stride = {
    "learning_rate": 10e-3,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 32, #wiorks great with 32
    "nb_fc_neurons" : 64,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 7,
    "padding": "CAUSAL",
    "emb_layer" : 'Flatten',
    "enumerate" : True,
    'str_inc' : True,
    'batch_norm' : True,
    "metrics": "r_square",
    'ker_dec' : True,
    'fc_dec' : True,
    "kernel_regularizer" : 1e-6,
    "loss": "rmse",
}


model_view_1 = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
model_view_1.prepare()
self = model_view_1

model_cfg_cnn_stride = {
    "learning_rate": 10e-3,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 16, #wiorks great with 32
    "nb_fc_neurons" : 64,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 5,
    "padding": "VALID",
    "emb_layer" : 'Flatten',
    "enumerate" : True,
    'str_inc' : True,
    'batch_norm' : True,
    "metrics": "r_square",
    'ker_dec' : True,
    'fc_dec' : True,
    "kernel_regularizer" : 1e-6,
    "loss": "rmse",
}


model_view_2 = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# pare the model (must be run before training)
model_view_2.prepare()

#ind = [1, 2,4,5,10]
x_train_2 = npy_concatenate(path, 'training_x', suffix = 'weather', T = 27)#[..., ind]
x_val_2 = npy_concatenate(path, 'val_x', suffix = 'weather', T = 27)#[..., ind]
x_test_2 = npy_concatenate(path, 'test_x', suffix = 'weather', T = 27)#[..., ind]

y_train =  np.load(os.path.join(path, 'training_y.npy'))
y_val =  np.load(os.path.join(path, 'val_y.npy'))
y_test =  np.load(os.path.join(path, 'test_y.npy'))
lst_features = np.load(os.path.join(path, 'list_features_weather.npy'))

model_view_1.fit_multiview(
    model_view_2 = model_view_2,
    src_train_dataset = (x_train, x_train_2, y_train),
    src_val_dataset = (x_val, x_val_2, y_val),
    src_test_dataset= (x_test, x_test_2, y_test),
    batch_size = 8,
    num_epochs = 500,
    model_directory = './test',
    save_steps=10,
    patience=50,
    reduce_lr=True
)

model_view_1.summary()