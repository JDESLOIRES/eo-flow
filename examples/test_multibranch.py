import os
import numpy as np
from eoflow.models.tempnets_task import cnn_tempnets, mlp_tempnets
import tensorflow


path_experiments = "/home/johann/Documents/Syngenta/FINAL_TRAINING/2021/fold_1/"

training_s1_x = np.load(
    os.path.join(path_experiments, "training_x_satellite_features.npy")
)
val_s1_x = np.load(os.path.join(path_experiments, "val_x_satellite_features.npy"))
test_s1_x = np.load(os.path.join(path_experiments, "test_x_satellite_features.npy"))


def reshape_array(x, T=13):
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x


training_s1_x = reshape_array(training_s1_x)
val_s1_x = reshape_array(val_s1_x)
test_s1_x = reshape_array(test_s1_x)

training_s2_x = np.load(
    os.path.join(path_experiments, "training_x_weather_features.npy")
)
val_s2_x = np.load(os.path.join(path_experiments, "val_x_weather_features.npy"))
test_s2_x = np.load(os.path.join(path_experiments, "test_x_weather_features.npy"))

training_s3_x = np.load(
    os.path.join(path_experiments, "training_x_static_features.npy")
)
val_s3_x = np.load(os.path.join(path_experiments, "val_x_static_features.npy"))
test_s3_x = np.load(os.path.join(path_experiments, "test_x_static_features.npy"))

training_s3_x = np.concatenate([training_s3_x, training_s2_x], axis=1)
val_s3_x = np.concatenate([val_s3_x, val_s2_x], axis=1)
test_s3_x = np.concatenate([test_s3_x, test_s2_x], axis=1)


training_y = np.load(os.path.join(path_experiments, "training_y.npy"))
val_y = np.load(os.path.join(path_experiments, "val_y.npy"))
test_y = np.load(os.path.join(path_experiments, "test_y.npy"))

######################################################################################################


def reshape_array(x, T=13):
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x


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

model_cfg_rnn_stride = {
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
model_cnn.prepare()

model_rnn = cnn_tempnets.MultiBranchCNN(model_cfg_rnn_stride)
model_rnn.prepare()


shapes = [
    list((training_s1_x.shape[0], training_s1_x.shape[1], 1))
    for i in range(training_s1_x.shape[-1])
]
static_shape = list(training_s2_x.shape)
inputs_shape = [shapes, static_shape]


model_cnn.fit_mb(
    train_dataset=(training_s1_x, training_s3_x, training_y),
    val_dataset=(val_s1_x, val_s3_x, val_y),
    test_dataset=(test_s1_x, test_s3_x, test_y),
    num_epochs=500,
    save_steps=5,
    batch_size=16,
    function=np.min,
    patience=50,
    reduce_lr=True,
    model_directory="/home/johann/Documents/model_16_",
)


x_dyn_batch_train = [test_s1_x[..., i] for i in range(test_s1_x.shape[-1])]
preds, _ = model_cnn.call([x_dyn_batch_train, test_s3_x], training=False)

import matplotlib.pyplot as plt

plt.scatter(test_y, preds)
plt.show()

from sklearn.metrics import r2_score

r2_score(test_y, preds)

#############################################################################
