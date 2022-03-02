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


path = '/home/johann/Documents/Syngenta/cleaned_V2/2021'
x_train = npy_concatenate(path, 'training_x')
y_train = np.load(os.path.join(path, 'training_y.npy'))

x_val = npy_concatenate(path, 'val_x')
y_val = np.load(os.path.join(path, 'val_y.npy'))

x_test = npy_concatenate(path, 'test_x')
y_test = np.load(os.path.join(path, 'test_y.npy'))

# x_train = np.concatenate([x_train, x_val], axis = 0)
# y_train = np.concatenate([y_train, y_val], axis = 0)



model_cfg_cnn_stride = {
    "learning_rate": 10e-3,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_conv_filters": 32,  # wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "kernel_size": 3,
    "batch_norm": True,
    'use_residual' : True,
    "kernel_regularizer": 1e-6,
    "loss": "mse",  # huber was working great for 2020 and 2021
    "metrics": "r_square",
}

#MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.InceptionCNN(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()


ts=3
self = model_cnn
x = x_train
batch_size = 8

print(path)

model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    test_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 32,
    function = np.min,
    shift_step = 1, #3
    sdev_label =0.05, #0.1
    feat_noise = 0, #0.2
    patience = 100,
    forget = 1,
    reduce_lr = True,
    #finetuning = True,
    #pretraining_path ='/home/johann/Documents/model_64_Causal_Stride_shift_0',
    model_directory='/home/johann/Documents/model_16',
)

t = model_cnn.predict(x_test)
plt.scatter(t, y_test)
plt.xlim((0.2,1))
plt.show()