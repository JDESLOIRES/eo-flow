import pandas as pd


import eoflow.models.tempnets_task.mlp_tempnets as mlp_tempnets
import numpy as np
import os

import matplotlib.pyplot as plt

########################################################################################################################
########################################################################################################################


def load_gdd_data(
    path_training="/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V4/training_final",
    group="2021/fold_1",
):
    def load_data(path, rsuffix):
        training_x = np.load(
            os.path.join(path, "training_x_" + rsuffix + ".npy"), allow_pickle=True
        )
        val_x = np.load(
            os.path.join(path, "val_x_" + rsuffix + ".npy"), allow_pickle=True
        )
        test_x = np.load(
            os.path.join(path, "test_x_" + rsuffix + ".npy"), allow_pickle=True
        )

        training_y = np.load(os.path.join(path, "training_y.npy"), allow_pickle=True)
        val_y = np.load(os.path.join(path, "val_y.npy"), allow_pickle=True)
        test_y = np.load(os.path.join(path, "test_y.npy"), allow_pickle=True)

        return training_x, val_x, test_x, training_y, val_y, test_y

    training_x_s1, val_x_s1, test_x_s1, training_y, val_y, test_y = load_data(
        rsuffix="satellite_features", path=os.path.join(path_training, group)
    )

    training_x_s2, val_x_s2, test_x_s2, _, _, _ = load_data(
        rsuffix="weather_features", path=os.path.join(path_training, group)
    )

    training_x_s3, val_x_s3, test_x_s3, _, _, _ = load_data(
        rsuffix="static_features", path=os.path.join(path_training, group)
    )

    return (
        training_x_s1,
        np.concatenate([training_x_s3, training_x_s2], axis=1),
        val_x_s1,
        np.concatenate([val_x_s3, val_x_s2], axis=1),
        test_x_s1,
        np.concatenate([test_x_s3, test_x_s2], axis=1),
        training_y,
        val_y,
        test_y,
    )


path_experiments = "/home/johann/Documents/Experiments SUBTAB/training_final_scaled_V2"
year = 2021

(
    training_dyn,
    training_x_static,
    val_x_dyn,
    val_x_static,
    test_x_dyn,
    test_x_static,
    training_y,
    val_y,
    test_y,
) = load_gdd_data(
    path_training=path_experiments, group=os.path.join(str(year), "fold_1")
)
training_x, val_x, test_x = (
    np.concatenate([training_dyn, training_x_static], axis=1),
    np.concatenate([val_x_dyn, val_x_static], axis=1),
    np.concatenate([test_x_dyn, test_x_static], axis=1),
)

training_x.shape

model_cfg_mlp = {
    "learning_rate": 5e-4,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_fc_neurons": 64,
    "nb_fc_stacks": 2,  # Nb FCN layers
    "kernel_initializer": "he_normal",
    "increase": True,
    "reduce": False,
    "batch_norm": True,
    "metrics": "r_square",
    "kernel_regularizer": 1e-7,
    "loss": "mse",
    "multibranch": False,
    "multioutput": False,
    "adaptative": True,
    "layer_before": 1,
    "factor": 0.0001,
    "ema": False,
}


# console 1 et 3 : activation in the layer + flipout
# console 4 et 5 : activation outsie
# MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = mlp_tempnets.MLPDANN(model_cfg_mlp)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_cnn.fit_dann(
    src_dataset=(training_x, training_y),
    val_dataset=(val_x, val_y),
    trgt_dataset=(test_x, test_y),
    num_epochs=500,
    save_steps=5,
    batch_size=12,
    patience=50,
    reduce_lr=True,
    init_models=False,
    model_directory="/home/johann/Documents/DANN_/",
)


import pickle

history = pickle.load(
    open(os.path.join("/home/johann/Documents/DANN/" + "2019", "history.pickle"), "rb")
)
history_train = pd.DataFrame(history["train_loss_results"])
history_disc = pd.DataFrame(history["disc_loss_results"])
history_task = pd.DataFrame(history["task_loss_results"])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history_train, color="red", label="Encoder")
ax.plot(history_task, color="green", label="Task")
ax.legend()
ax2 = ax.twinx()
ax2.plot(history_disc, color="blue", label="Disc")
ax2.legend(loc="upper left")
plt.show()


model_cnn.summary()

########################################################################################################################
