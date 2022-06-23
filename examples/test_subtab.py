import eoflow.models.tempnets_task.mlp_tempnets as mlp_tempnets
import numpy as np
import os


def npy_concatenate(path, prefix='training_x', T=30):
    path_npy = os.path.join(path, prefix)
    return np.load(path_npy + '_S2.npy')

model_cfg_mlp = {
    "learning_rate": 1e-4,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_fc_neurons": 64,
    "nb_fc_stacks": 3,  # Nb FCN layers
    "kernel_initializer": 'he_normal',
    "increase": True,
    "reduce": False,
    "batch_norm": True,
    "metrics": "r_square",
    "kernel_regularizer": 1e-7,
    "loss": "mse",
    "multibranch": False,
    "multioutput": False,
    "adaptative" : True,
    "layer_before" : 0,
    "ema": False
}



year = '2021'
path = '/home/johann/Documents/Syngenta/cleaned_V2/' + year
#path = '/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V3/cleaner_V2_training_10_folds/' + year + '/fold_1'

x_train = npy_concatenate(path, 'training_x')
y_train = np.load(os.path.join(path, 'training_y.npy'))

x_val = npy_concatenate(path, 'val_x')
y_val = np.load(os.path.join(path, 'val_y.npy'))


x_test = npy_concatenate(path, 'test_x')
y_test = np.load(os.path.join(path, 'test_y.npy'))


model_compiled = mlp_tempnets.MLP(model_cfg_mlp)
model_compiled.prepare()

self = model_compiled

model_compiled.fit_pretrain(
    x_train=x_train,
    num_epochs=100,
    batch_size =8,
    n_subsets=3, overlap=0.75,
    p_m=0.3, noise_level=0.15,
    model_directory='/home/johann/Documents/SSL/' + year
)


model_compiled.fit_supervised(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    test_dataset=(x_test, y_test),
    batch_size=8,
    num_epochs=10,
    model_directory='/home/johann/Documents/SSL/' + year,
    save_steps = 10,
    n_subsets=3, overlap=0.75,
    p_m =0.3, noise_level=0.15,
)

