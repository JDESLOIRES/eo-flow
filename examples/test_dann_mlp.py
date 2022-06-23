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
self.loss
x = x_train


self(x_test)
model_compiled.fit_dann_v3(
    src_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    trgt_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size=12,
    patience=50,
    reduce_lr=True,
    model_directory='/home/johann/Documents/DANN/' + year
)


self._get_task(x)