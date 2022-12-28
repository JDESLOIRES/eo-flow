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


path = "/home/johann/Documents/Syngenta/gdd_training_30/2021"
x_train = npy_concatenate(path, "training_x")
# x_train = x_train[..., [10, -1]]
x_train.shape
y_train = np.load(os.path.join(path, "training_y.npy"))

x_val = npy_concatenate(path, "val_x")
# x_val = x_val[..., [10, -1]]
y_val = np.load(os.path.join(path, "val_y.npy"))

x_test = npy_concatenate(path, "test_x")
# x_test = x_test[..., [10, -1]]
y_test = np.load(os.path.join(path, "test_y.npy"))

ls_features = np.load(os.path.join(path, "list_features_S2.npy"))
"""
model = RandomForestRegressor()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
model.fit(x_train, y_train)
preds = model.predict(x_test)
r2_score(y_test, preds)
"""
# x_train = np.concatenate([x_train, x_val], axis = 0)
# y_train = np.concatenate([y_train, y_val], axis = 0)

nb_split = x_train.shape[1] // 4

# switch = np.random.randint(low = 0, high = 4, size = 4, re = False)
############################################################################
###########################################################################


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
    "keep_prob_conv": 0.8,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_conv_filters": 10,  # wiorks great with 32
    "nb_conv_stacks": 2,
    "nb_fc_neurons": 64,
    "kernel_size": 3,
    "nb_fc_stacks": 2,  # Nb FCN layers
    "fc_activation": "relu",
    "static_fc_neurons": 64,
    "padding": "SAME",
    "metrics": "r_square",
    "kernel_regularizer": 1e-7,
    "loss": "mse",
    "reduce": False,
    "str_inc": True,
    "ema": False,
}

# console 1 et 3 : activation in the layer + flipout
# console 4 et 5 : activation outsie
# MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.MultiBranchCNN(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()


y_train_ = np.concatenate([y_train, y_train], axis=1)
y_train_ = np.concatenate([y_train, y_train], axis=1)
y_test_ = np.concatenate([y_test, y_test], axis=1)


model_cnn.fit(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    test_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size=32,
    function=np.min,
    patience=50,
    forget=0,
    reduce_lr=True,
    model_directory="/home/johann/Documents/model_16_",
)


#############
########################################################################################################################
########################################################################################################################

# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-5,
    "keep_prob": 0.8,
    "nb_conv_filters": 128,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons": 2048,
    "nb_fc_stacks": 1,  # Nb FCN layers
    "kernel_size": [1, 1],
    "n_strides": [1, 1],
    "kernel_initializer": "he_normal",
    "batch_norm": True,
    "padding": "VALID",  # "VALID", CAUSAL works great?!
    "kernel_regularizer": 1e-6,
    "final_layer": "Flatten",
    "loss": "huber",
    "enumerate": True,
    "metrics": ["mse", "mae"],
}


model_cnn = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
# Prepare the model (must be run before training)
model_cnn.prepare()
model_cnn.build((None, 30, 32, 9))
model_cnn.summary()
output_file_name_cnnlstm, checkpoint = utils.define_callbacks(
    path_DL, model_cfg_cnnlstm, prefix="cnnlstm_"
)


# Train the model
model_cnn.train_and_evaluate(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=500,
    iterations_per_epoch=iterations_per_epoch,
    model_directory=os.path.join(
        path_DL, os.path.join(output_file_name_cnnlstm, "model")
    ),
    save_steps=10,
    summary_steps="epoch",
    callbacks=[checkpoint],
)
