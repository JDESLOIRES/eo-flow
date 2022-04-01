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

    x = np.load(path_npy + '_S2.npy')
    x = reshape_array(x, T)
    return x


year = '2019'
#path = '/home/johann/Documents/Syngenta/cleaned_V2/' + year
path = '/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V3/cleaner_V2_training_10_folds/' + year + '/fold_1'

x_train = npy_concatenate(path, 'training_x')
y_train = np.load(os.path.join(path, 'training_y.npy'))

x_val = npy_concatenate(path, 'val_x')
y_val = np.load(os.path.join(path, 'val_y.npy'))

x_test = npy_concatenate(path, 'test_x')
y_test = np.load(os.path.join(path, 'test_y.npy'))

x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)


model_cfg_cnn_stride = {
    "learning_rate": 10e-4,
    "keep_prob": 0.65,  # should keep 0.8
    "nb_conv_filters": 32,  # wiorks great with 32
    "nb_fc_neurons": 32,
    "metrics": "r_square",
    "loss": "mse",
    'factor' : 10e-4,
    'adaptative' : True,
    'ema': True,
    'loss' : 'rmse'
}


model_cnn = cnn_tempnets_functional.TempDANN(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()



model_cnn.fit_dann_v3(
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
    model_directory='/home/johann/Documents/DANN_v3/' + year,
)


import pickle
history = pickle.load(open(os.path.join('/home/johann/Documents/DANN_v3/' + '2019', 'history.pickle'), 'rb'))
history_train = pd.DataFrame(history['train_loss_results'])
history_disc = pd.DataFrame(history['disc_loss_results'])
history_task = pd.DataFrame(history['task_loss_results'])

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(history_train.iloc[10:], color = 'red',  label='Encoder')
ax.plot(history_task.iloc[10:], color = 'green', label = 'Task')
ax.legend()
ax2 = ax.twinx()
ax2.plot(history_disc[10:], color = 'blue',  label='Disc')
ax2.legend(loc = 'upper left')
plt.show()


model_cnn.summary()
