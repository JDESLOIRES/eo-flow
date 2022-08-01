import pandas as pd


import eoflow.models.tempnets_task.mlp_tempnets as mlp_tempnets
import numpy as np
import os

import matplotlib.pyplot as plt
""
########################################################################################################################
########################################################################################################################
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

    return training_x_s1, np.concatenate([training_x_s3, training_x_s2], axis = 1), \
           val_x_s1, np.concatenate([val_x_s3, val_x_s2], axis = 1), \
           test_x_s1, np.concatenate([test_x_s3, test_x_s2], axis = 1), \
           training_y, val_y, test_y


path_experiments = '/home/johann/Documents/Experiments SUBTAB/training_final_scaled_paper'
year = 2019

training_dyn, training_x_static, \
val_x_dyn, val_x_static,\
test_x_dyn, test_x_static, \
training_y, val_y, test_y = load_gdd_data(path_training=path_experiments,
                                          group=os.path.join(str(year), 'fold_1'))
training_x, val_x, test_x = np.concatenate([training_dyn, training_x_static], axis = 1),\
                            np.concatenate([val_x_dyn, val_x_static], axis = 1),\
                            np.concatenate([test_x_dyn, test_x_static], axis = 1)

path_experiments = '/home/johann/Documents/Experiments SUBTAB/training_final_scaled_is_paper'
training_x_is_dyn, training_x_is_static, \
val_x_is_dyn, val_x_is_static,\
test_x_is_dyn, test_x_is_static, \
training_y, val_y, test_y = load_gdd_data(path_training=path_experiments,
                                          group=os.path.join(str(year), 'fold_1'))

training_x_is, val_x_is, test_x_is = np.concatenate([training_x_is_dyn, training_x_is_static], axis = 1),\
                                     np.concatenate([val_x_is_dyn, val_x_is_static], axis = 1),\
                                     np.concatenate([test_x_is_dyn, test_x_is_static], axis = 1)



model_cfg_mlp = {
    "learning_rate": 10e-4,
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
    "ema": False,
    "layer_before" : 1
}


model_cnn = mlp_tempnets.MLP(model_cfg_mlp)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_student = mlp_tempnets.MLP(model_cfg_mlp)
# Prepare the model (must be run before training)
model_student.prepare()



'''
model_student.fit_unsupervised(
    x_dynamic=training_x_is_dyn,
    x_static=training_x_is_static,
    x_orig=np.concatenate([training_dyn, training_x_static], axis = 1),
    num_epochs=50,
    batch_size=8,
    p_m=0.075, noise_level=0.15,
    temperature= 0.01, #0.1,
    permut=True,
    rho= 0.05,
    model_directory='/home/johann/Documents/STUDENT_PRETRAIN/' + str(year)
)



model_cnn.fit_unsupervised(
    x_dynamic=training_dyn,
    x_static=training_x_static,
    x_orig=np.concatenate([training_dyn, training_x_static], axis = 1),
    num_epochs=50,
    batch_size=8,
    p_m=0.075, noise_level=0.15,
    temperature= 0.01, #0.1,
    permut=True,
    rho= 0.05,
    model_directory='/home/johann/Documents/TEACHER_PRETRAIN/' + str(year)
)
'''



#model_student.load_weights()

##TODO : Save final model with encoder weights in

self = model_cnn

model_cnn.fit_kd(
    src_dataset=(training_x, training_y),
    trgt_dataset=(training_x_is, training_y),
    #train_dataset=(training_x, training_y),
    val_dataset=(val_x_is, val_y),
    test_dataset=(test_x_is, test_y),
    teacher_model=model_student,
    num_epochs=500,
    save_steps=5,
    batch_size=12,
    patience=50,
    temperature=1,
    gamma_=0.2,
    lamda_=0.5,#fmap
    reduce_lr=True,
    model_directory='/home/johann/Documents/KD/',
    pretrain_student_path='/home/johann/Documents/STUDENT_PRETRAIN/' + str(2021),
    pretrain_teacher_path='/home/johann/Documents/TEACHER_PRETRAIN/' + str(2021),
)


model_cnn.load_weights('/home/johann/Documents/STUDENT_PRETRAIN/' + str(year) +'/model')
model_student.load_weights('/home/johann/Documents/TEACHER_PRETRAIN/' + str(year) +'/model')



preds, _ = model_student.predict(test_x_is)
plt.scatter(test_y, preds)
plt.show()