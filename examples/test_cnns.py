import eoflow.models.tempnets_task.cnn_tempnets as cnn_tempnets
import tensorflow as tf

# Model configuration CNNLSTM
import numpy as np
import os
import tensorflow_addons as tfa
import matplotlib.pyplot as plt



########################################################################################################################
########################################################################################################################
def reshape_array(x, T=27) :
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x

path = '/home/johann/Documents/Syngenta/2020/fold_5/'
x_train = np.load(os.path.join(path, 'training_x_S2.npy'))
y_train = np.load(os.path.join(path, 'training_y.npy'))
x_train = reshape_array(x_train)
x_val = np.load(os.path.join(path, 'val_x_S2.npy'))
x_val = reshape_array(x_val)
y_val = np.load(os.path.join(path, 'val_y.npy'))
x_test = np.load(os.path.join(path, 'test_x_S2.npy'))
x_test = reshape_array(x_test)
y_test = np.load(os.path.join(path, 'test_y.npy'))

x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)

# Model configuration CNN
model_cfg_cnn = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5,
    "nb_conv_filters": 64,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 256,
    "nb_fc_stacks": 1, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "nb_conv_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "CAUSAL",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "final_layer" : 'GlobalAveragePooling1D',
    "loss": "huber",
    "enumerate" : True,
    "metrics": "mae"
}


model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn)
# Prepare the model (must be run before training)
model_cnn.prepare()

#EMA 0.99
# Train the model
timeshift = 4
model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    test_dataset = (x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 8,
    function = np.min,
    shift_step = timeshift,
    sdev_label =0.15,
    feat_noise = 0.2,
    reduce_lr = False,
    pretraining = True,
    model_directory='/home/johann/Documents/model_KR_MSE_' + str(timeshift),
)

#200 epochs enough? or dropout 0.5?
#enumerate start with 64, globalaveragepooling, one dense and zou

'''
console 0 : timeshift = 2
consiole 2 : timeshift = 0
console 3 :timeshift = 4
console 4 :timeshift = 1
'''
#NOTE : modèle KR avec sans timeshit : 0.11 last model good mais -0,12 sinon ..
#NOTE : modèle KR avec timeshit 3 : 0.15 last model good mais -0,22 sinon .. monté jusqu'a 0.25 :0
#NOTE : modèle KR avec timeshit 3 + NL : -0.09 last model good mais -0,32 sinon ..
#NOTE : modèle KR avec timeshit 3 + NL + RN : 0.08 last model good mais -0,32 sinon ..
#Just timeshift 4 pas bon, timeshift 2 0.065
timeshift =1
model_cnn.load_weights('/home/johann/Documents/model_KR_MSE_' + str(timeshift) + '/best_model')
t = model_cnn.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
mean_squared_error(y_test, t)
r2_score(y_test, t)
mean_absolute_error(y_test, t)

import matplotlib.pyplot as plt
plt.scatter(y_test,t, vmin = 0, vmax = 1)
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))
plt.show()
########################################################################################################################
########################################################################################################################

# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5,
    "nb_conv_filters": 128,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 2048,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [1,1],
    "nb_conv_strides" : [1,1],
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "VALID",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "final_layer" : 'Flatten',
    "loss": "huber",
    "enumerate" : True,
    "metrics": ["mse", "mae"]
}



model_cnn = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
# Prepare the model (must be run before training)
model_cnn.prepare()
model_cnn.build((None, 30, 32, 9))
model_cnn.summary()
output_file_name_cnnlstm, checkpoint = utils.define_callbacks(path_DL, model_cfg_cnnlstm, prefix = 'cnnlstm_')


# Train the model
model_cnn.train_and_evaluate(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=500,
    iterations_per_epoch=iterations_per_epoch,
    model_directory=os.path.join(path_DL, os.path.join(output_file_name_cnnlstm, "model")),
    save_steps=10,
    summary_steps='epoch',
    callbacks=[checkpoint]
)
