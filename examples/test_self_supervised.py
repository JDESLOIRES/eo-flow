import matplotlib.pyplot as plt
import eoflow
from importlib import reload
eoflow = reload(eoflow)

import eoflow.models.tempnets_task.mlp_tempnets as mlp_tempnets

import numpy as np
import os


def npy_concatenate(path, prefix='training_x', T=30):
    path_npy = os.path.join(path, prefix)
    return np.load(path_npy + '_S2.npy')



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

    return np.concatenate([training_x_s1, training_x_s2], axis = 1), training_x_s3, \
           np.concatenate([val_x_s1, val_x_s2], axis = 1), val_x_s3, \
           np.concatenate([test_x_s1, test_x_s2], axis = 1), test_x_s3, \
           training_y, val_y, test_y


path_experiments = '/home/johann/Documents/Experiments SUBTAB/training_final_scaled_V2'
year = 2017

training_dyn, training_x_static, \
val_x_dyn, val_x_static,\
test_x_dyn, test_x_static, \
training_y, val_y, test_y = load_gdd_data(path_training=path_experiments,
                                          group=os.path.join(str(year), 'fold_1'))

training_x, val_x, test_x = np.concatenate([training_dyn, training_x_static], axis = 1),\
                            np.concatenate([val_x_dyn, val_x_static], axis = 1),\
                            np.concatenate([test_x_dyn, test_x_static], axis = 1)

a,b,c = training_dyn[:, :13], training_dyn[:, 13:26], training_dyn[:, 26:39]


y = np.concatenate([training_y, val_y, test_y])
min_y, max_y = np.quantile(y, 0.01), np.quantile(y, 0.99)

training_y_ = (training_y - min_y) / (max_y - min_y)
training_y_[training_y_<0] = 0
training_y_[training_y_>1] = 1

val_y_ = (val_y - min_y) / (max_y - min_y)
val_y_[val_y_<0] = 0
val_y_[val_y_>1] = 1

test_y_ = (test_y - min_y) / (max_y - min_y)
test_y_[test_y_<0] = 0
test_y_[test_y_>1] = 1


x_dyn = np.concatenate([training_dyn, val_x_dyn, test_x_dyn], axis = 0)
x_st = np.concatenate([training_x_static, val_x_static, test_x_static], axis = 0)


model_cfg_mlp = {
    "learning_rate": 10e-3,
    "keep_prob": 0.5,  # should keep 0.8
    "nb_fc_neurons": 128,
    "nb_fc_stacks": 3,  # Nb FCN layers
    "kernel_initializer": 'he_normal',
    "increase": False,
    "reduce": False,
    "batch_norm": True,
    "metrics": "r_square",
    "kernel_regularizer": 1e-7,
    "loss": "mse",
    "multibranch": False,
    "multioutput": False,
    "adaptative" : False,
    "layer_before" : 1,
    'variational' : False,
    "ema": False
}


model_compiled = mlp_tempnets.MLP(model_cfg_mlp)
model_compiled.prepare()
self = model_compiled


model_compiled.fit_unsupervised(
    x_dynamic=x_dyn,
    x_static=x_st,
    x_orig=np.concatenate([x_dyn, x_st], axis = 1),
    num_epochs=100,
    batch_size=12,
    p_m=0.25, noise_level=0.25,
    temperature= 0.01, #0.1,
    permut=True,
    rho= 0.05,
    mask_loss=False,
    model_directory='/home/johann/Documents/Experiments SUBTAB/SSL_ENC_128_NOMASK/'
)



'''
works??§ no..
model_directory = '/home/johann/Documents/Experiments SUBTAB/SSL_ENC_128_MASK/' + str(year),
save_steps = 5,
finetuning = True,
unfreeze = True,
'''

model_cfg_mlp['loss'] = 'lcc'
model_cfg_mlp['keep_prob'] = 0.5
model_compiled = mlp_tempnets.MLP(model_cfg_mlp)
model_compiled.prepare()
#nomask better
model_compiled.fit_task(
    train_dataset=(training_dyn, training_x_static, training_y),
    val_dataset=(val_x_dyn, val_x_static,val_y),
    test_dataset=(test_x_dyn, test_x_static, test_y),
    batch_size=12,
    num_epochs=500,
    model_directory='/home/johann/Documents/Experiments SUBTAB/SSL_ENC_128_NOMASK/' + str(year),
    save_steps = 5,
    finetuning = True,
    load_weights=True,
    unfreeze=True,
    patience = 50,
    p_m =0.1, noise_level=0.1,
    permut=True
)


preds = model_compiled.predict(np.concatenate([test_x_dyn, test_x_static], axis=1))
from sklearn.metrics import r2_score, mean_absolute_error
plt.scatter(test_y, preds)
plt.show()
r2_score(test_y, preds)

model_compiled.encoder.load_weights(os.path.join('/home/johann/Documents/Experiments SUBTAB/SSL_ENC_128_NOMASK/' + str(year), 'encoder_last_model'))
model_compiled.task.load_weights(os.path.join('/home/johann/Documents/Experiments SUBTAB/SSL_ENC_128_NOMASK/' + str(year), 'task_last_model'))
year
h, z_, x_reco, task = model_compiled.forward(np.concatenate([test_x_dyn, test_x_static], axis=1), training=False)
plt.scatter(test_y, task)
plt.show()
r2_score(test_y, task)
mean_absolute_error(test_y, task)

model_compiled.fit(
    train_dataset=(np.concatenate([training_dyn, training_x_static], axis = 1), training_y),
    val_dataset=(np.concatenate([val_x_dyn, val_x_static], axis = 1), val_y),
    test_dataset=(np.concatenate([test_x_dyn, test_x_static], axis = 1), test_y),
    batch_size=12,
    num_epochs=200,
    model_directory='/home/johann/Documents/Experiments SUBTAB/SUBTAB_ENC_NODP_SWAP_SPARSE/' + str(year),
    save_steps = 5,
    patience = 0,
)

