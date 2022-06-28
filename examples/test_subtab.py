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

model_cfg_mlp = {
    "learning_rate": 10e-4,
    "keep_prob": 0.65,  # should keep 0.8
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
    "layer_before" : 1,
    "ema": False
}

model_cfg_mlp_pre = {
    "learning_rate": 10e-4,
    "keep_prob": 0.8,  # should keep 0.8
    "nb_fc_neurons": 64,
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

model_compiled = mlp_tempnets.MLP(model_cfg_mlp)
model_compiled.prepare()

model_kd = mlp_tempnets.MLP(model_cfg_mlp_pre)
model_kd.prepare()

model_kd.load_weights(os.path.join('/home/johann/Documents/SSL/' + str(year),'model/last_model'))
model_kd(training_x)

self = model_compiled


model_compiled.fit_pretrain(
    x_train=np.concatenate([training_x, val_x, test_x], axis = 0),
    num_epochs=30,
    batch_size =8,
    n_subsets=4, overlap=0.9,
    p_m=0.2, noise_level=0.1,
    temperature=1,
    model_kd=model_kd,
    lambda_=0.0001,
    model_directory='/home/johann/Documents/SSL_64-128-64_4L_3_0.75_RELU_KD/' + str(year)
)



model_compiled.fit_supervised(
    train_dataset=(training_x, training_y),
    val_dataset=(val_x, val_y),
    test_dataset=(test_x, test_y),
    batch_size=8,
    num_epochs=200,
    add_layer=False,
    #model_directory='/home/johann/Documents/SSL_64_3_0.75_RELU_KD/' + str(2021),
    model_directory='/home/johann/Documents/SSL_64-128-64_4L_3_0.75_RELU_KD/' + str(year),
    save_steps = 5,
    finetuning = True,
    n_subsets=4, overlap=0.9,
    p_m =0, noise_level=0,
)



# indices = np.random.RandomState(seed=0).permutation(test_x.shape[1])
# if permut: test_x = test_x[:,indices]

preds = model_compiled.subtab_pred_step(x_test_=test_x,
                                        model_directory='/home/johann/Documents/SSL_256_2L_3_0.75_RELU_KD/' + str(2021),
                                        #model_directory='/home/johann/Documents/SSL_3_V2_RELU/' + str(2017),
                                        permut = True,
                                        n_subsets=2, overlap=0.75)

import matplotlib.pyplot as plt
plt.scatter( test_y, preds)
plt.show()


from sklearn.metrics import r2_score, mean_absolute_error
r2_score(test_y, preds)
mean_absolute_error(test_y, preds)
year