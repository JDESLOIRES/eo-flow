import matplotlib.pyplot as plt

import eoflow.models.tempnets_task.mlp_tempnets as mlp_tempnets
import numpy as np
import os


def npy_concatenate(path, prefix='training_x', T=30):
    path_npy = os.path.join(path, prefix)
    return np.load(path_npy + '_S2.npy')

model_cfg_mlp = {
    "learning_rate": 10e-4,
    "keep_prob": 0.65,  # should keep 0.8
    "nb_fc_neurons": 256,
    "nb_fc_stacks": 3,  # Nb FCN layers
    "kernel_initializer": 'he_normal',
    "increase": False,
    "reduce": True,
    "batch_norm": True,
    "metrics": "r_square",
    "kernel_regularizer": 1e-7,
    "loss": "mse",
    "multibranch": False,
    "multioutput": False,
    "adaptative" : True,
    "layer_before" : 1,
    "ema": False
}


def load_gdd_data(path_training =  '/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V4/training_final',
                  group = '2021/fold_1'):

    def load_data(path, rsuffix):
        training_x = np.load(os.path.join(path, 'training_x_' + rsuffix + '.npy'), allow_pickle=True)
        val_x = np.load(os.path.join(path, 'val_x_' + rsuffix + '.npy'), allow_pickle=True)
        test_x = np.load(os.path.join(path, 'test_x_' + rsuffix + '.npy'), allow_pickle=True)

        training_y = np.load(os.path.join(path, 'training_y.npy'), allow_pickle=True)
        val_y = np.load(os.path.join(path, 'val_y.npy'), allow_pickle=True)
        test_y = np.load(os.path.join(path, 'test_y.npy'), allow_pickle=True)

        return training_x, val_x, test_x, training_y, val_y, test_y

    training_x_s1, val_x_s1, test_x_s1, training_y, val_y, test_y = load_data(rsuffix='satellite_features',
                                                                              path=os.path.join(path_training, group))

    training_x_s2, val_x_s2, test_x_s2, _, _, _ = load_data(rsuffix='weather_features',
                                                            path=os.path.join(path_training, group))

    training_x_s3, val_x_s3, test_x_s3, _, _, _ = load_data(rsuffix='static_features',
                                                            path=os.path.join(path_training, group))

    training_x = np.concatenate([training_x_s1, training_x_s2, training_x_s3], axis = 1)
    val_x = np.concatenate([val_x_s1, val_x_s2, val_x_s3], axis = 1)
    test_x = np.concatenate([test_x_s1, test_x_s2, test_x_s3], axis = 1)

    return training_x, val_x, test_x, training_y, val_y, test_y


path_experiments = '/home/johann/Documents/SSL'
year = 2020
training_x, val_x, test_x, training_y, val_y, test_y = load_gdd_data(path_training=path_experiments,
                                                                     group=str(year) )


model_compiled = mlp_tempnets.MLP(model_cfg_mlp)
model_compiled.prepare()

self = model_compiled


model_compiled.fit_pretrain(
    x_train=np.concatenate([training_x, val_x, test_x], axis = 0),
    num_epochs=100,
    batch_size =8,
    n_subsets=3, overlap=0.75,
    p_m=0.2, noise_level=0.15,
    model_directory='/home/johann/Documents/SSL_3_RELU/' + str(year)
)


model_compiled.fit_supervised(
    train_dataset=(training_x, training_y),
    val_dataset=(val_x, val_y),
    test_dataset=(test_x, test_y),
    batch_size=8,
    num_epochs=200,
    model_directory='/home/johann/Documents/SSL/' + str(year),
    save_steps = 10,
    n_subsets=3, overlap=0.75,
    p_m =0, noise_level=0,
)


preds =model_compiled.subtab_pred_step(test_x)
import matplotlib.pyplot as plt
plt.scatter(preds, test_y)
plt.show()