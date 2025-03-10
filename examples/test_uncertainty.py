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


def npy_concatenate(path, prefix="training_x", T=30):
    path_npy = os.path.join(path, prefix)
    """

    x_bands = np.load(path_npy + '_bands.npy')
    x_bands = reshape_array(x_bands, T)
    x_vis = np.load(path_npy  + '_vis.npy')
    x_vis = reshape_array(x_vis, T)
    np.concatenate([x_bands, x_vis], axis = -1)
    """
    x = np.load(path_npy + "_S2.npy")
    x = reshape_array(x, T)
    return x


path = "/home/johann/Documents/Syngenta/cleaned_V2/2020"
x_train = npy_concatenate(path, "training_x")
y_train = np.load(os.path.join(path, "training_y.npy"))

x_val = npy_concatenate(path, "val_x")
y_val = np.load(os.path.join(path, "val_y.npy"))

x_test = npy_concatenate(path, "test_x")
y_test = np.load(os.path.join(path, "test_y.npy"))

# x_train = np.concatenate([x_train, x_val], axis = 0)
# y_train = np.concatenate([y_train, y_val], axis = 0)


"""
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=8)
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
model.fit(x_train, y_train)
preds = model.predict(x_test)
r2_score(y_test, preds)

"""


model_cfg_cnn_stride = {
    "learning_rate": 10e-4,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_conv_filters": 64,  # wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons": 128,
    "nb_fc_stacks": 1,  # Nb FCN layers
    "fc_activation": "relu",
    "kernel_size": 3,
    "n_strides": 1,
    "kernel_initializer": "he_normal",
    "batch_norm": True,
    "padding": "SAME",
    "kernel_regularizer": 1e-6,
    "emb_layer": "GlobalAveragePooling1D",
    "enumerate": True,
    "str_inc": True,
    "metrics": "r_square",
    "loss": "laplacian",  # huber was working great for 2020 and 2021
}


# MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    test_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size=8,
    function=np.min,
    shift_step=0,  # 3
    sdev_label=0,  # 0.1
    feat_noise=0,  # 0.2
    patience=100,
    reduce_lr=True,
    forget=0,
    model_directory="/home/johann/Documents/model_16",
)
