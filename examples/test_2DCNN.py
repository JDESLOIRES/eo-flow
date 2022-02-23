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
x_train[np.isnan(x_train)] = 0
plt.imshow(x_train[10,:,:,12], origin = 'lower')

y_train = np.load(os.path.join(path, 'training_y.npy'))
mins = np.where(y_train == np.min(y_train))[0]
maxs = np.where(y_train == np.max(y_train))[0]
plt.imshow(x_train[mins[2],:,:,13], origin = 'lower')
plt.show()
plt.imshow(x_train[maxs[2],:,:,13], origin = 'lower')
plt.show()
x_train_ts = ts_concatenate('/home/johann/Documents/Syngenta/cleaned_training_5_folds/2020/fold_1', 'training_x')
y_ts = np.load('/home/johann/Documents/Syngenta/cleaned_training_5_folds/2020/fold_1/training_y.npy')

n,h,w = x_train_ts.shape
x_train_ts_un = x_train_ts.reshape(n,h*w)
x_train_ts_un.reshape(n,h,w) == x_train_ts
plt.plot(x_train_ts[mins[2],:,13])
plt.plot(x_train_ts[maxs[2],:,13])
plt.show()


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
    "nb_conv_filters": 16,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 128,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [2,2],
    "nb_conv_strides" : [1,1],
    "kernel_initializer" : 'he_normal',
    "fc_activation" : 'relu',
    "batch_norm": True,
    'emb_layer' : 'Flatten',
    "padding": "SAME",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "loss": "mse",
    "enumerate" : True,
    "metrics": 'r_square'
}



model_cnn = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
# Prepare the model (must be run before training)
model_cnn.prepare()


model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=1000,
    save_steps=5,
    batch_size = 8,
    function = np.min,
    patience = 30,
    reduce_lr = False,
    sdev_label = 0,
    model_directory='/home/johann/Documents/model_hist',
)

t = model_cnn.predict(x_train)
plt.scatter(y_train.flatten(), t.flatten())
plt.show()