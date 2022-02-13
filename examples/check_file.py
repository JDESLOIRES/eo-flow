import pandas as pd

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
def reshape_array(x, T=30) :
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x

def npy_concatenate(path, prefix = 'training_x',T = 30):
    path_npy = os.path.join(path, prefix)
    x_bands = np.load(path_npy + '_bands.npy')
    x_bands = reshape_array(x_bands, T)
    x_vis = np.load(path_npy  + '_vis.npy')
    x_vis = reshape_array(x_vis, T)
    return np.concatenate([x_bands, x_vis], axis = -1)

x_vals = []
preds_vals_list = []
preds_test_list = []

for i in range(1,11):
    path = '/home/johann/Documents/Syngenta/cleaned_training_5_folds/2020/fold_' + str(i)
    x_train = npy_concatenate(path, 'training_x')
    y_train = np.load(os.path.join(path, 'training_y.npy'))

    x_val = npy_concatenate(path, 'val_x')
    y_val = np.load(os.path.join(path, 'val_y.npy'))

    x_test = npy_concatenate(path, 'test_x')
    y_test = np.load(os.path.join(path, 'test_y.npy'))


    meta = np.load(os.path.join(path, 'training_x_meta.npy'), allow_pickle=True)

    model = RandomForestRegressor()
    x_train = x_train.reshape((x_train.shape[0], 450))
    model.fit(x_train, y_train)
    preds_val = model.predict(x_val.reshape((x_val.shape[0], 450)))
    error_val = y_val.flatten() - preds_val.flatten()
    preds_vals_list.append(error_val)
    preds_test = model.predict(x_test.reshape((x_test.shape[0], 450)))
    preds_test_list.append(preds_test.flatten())
    x_vals.append(x_val.reshape((x_val.shape[0], 450)))


x_new_val = np.concatenate(x_vals, axis = 0)
new_y_val = np.concatenate(preds_vals_list, axis = 0)

avg_preds_test = (preds_test_list[0][0] + preds_test_list[1][0] +
                  preds_test_list[2][0] + preds_test_list[3][0] + preds_test_list[4][0])/5

plt.scatter(y_test, avg_preds_test)
plt.show()

model = RandomForestRegressor(max_depth=3)
model.fit(x_new_val,new_y_val)
npre = model.predict(x_test.reshape((x_test.shape[0], 450)))
npre = npre.flatten() + avg_preds_test.flatten()
plt.scatter(y_test, npre)
plt.show()

r2_score(y_test, npre)
mean_absolute_error(y_test, npre)

df = pd.DataFrame('/home/johann/Documents/Syngenta/')