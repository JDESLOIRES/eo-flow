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

def reshape_array(x, T=30) :
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x

def npy_concatenate(path, prefix = 'training_x',T = 30):
    path_npy = os.path.join(path, prefix)
    '''
    
    x_bands = np.load(path_npy + '_bands.npy')
    x_bands = reshape_array(x_bands, T)
    x_vis = np.load(path_npy  + '_vis.npy')
    x_vis = reshape_array(x_vis, T)
    np.concatenate([x_bands, x_vis], axis = -1)
    '''
    x = np.load(path_npy + '_S2.npy')
    x = reshape_array(x, T)
    return  x

path = '/home/johann/Documents/Syngenta/cleaned_V2/2017'
x_train = npy_concatenate(path, 'training_x')
y_train = np.load(os.path.join(path, 'training_y.npy'))

x_val = npy_concatenate(path, 'val_x')
y_val = np.load(os.path.join(path, 'val_y.npy'))

x_test = npy_concatenate(path, 'test_x')
y_test = np.load(os.path.join(path, 'test_y.npy'))

x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)

nb_split = x_train.shape[1]//4

#switch = np.random.randint(low = 0, high = 4, size = 4, re = False)
############################################################################
###########################################################################


'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=8)
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
model.fit(x_train, y_train)
preds = model.predict(x_test)
r2_score(y_test, preds)
'''


model_cfg_cnn_stride = {
    "learning_rate": 10e-3,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 32, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 32,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 7,
    "n_strides" : 1,
    "padding": "CAUSAL",
    "emb_layer" : 'GlobalAveragePooling1D',
    "enumerate" : True,
    'str_inc' : True,
    'batch_norm' : True,
    "metrics": "r_square",
    'ker_dec' : True,
    'fc_dec' : True,
    #"activity_regularizer" : 1e-4,
    "loss": "mse",
    'multioutput' : False
}

#console 1 et 3 : activation in the layer + flipout
#console 4 et 5 : activation outsie
#MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_cnn.summary
self = model_cnn
x = x_train
y = y_train

pretraining = False
cotraining = False


if pretraining:
    x_pretrain = np.concatenate([x_train, x_test], axis = 0)
    model_cnn.pretraining(x_pretrain,
                          pretraining_path='/home/johann/Documents/model_64_Causal_Stride_shift_4',
                          num_epochs=10, shift=0, proba=0.5, noise = 0.5)
    model_directory = '/home/johann/Documents/model_64_Causal_Stride_shift_4'



if cotraining:
    model_cnn.cotraining(train_dataset=(x_train, y_train),
                         val_dataset=(x_test, y_test),
                         num_epochs=500,
                         batch_size = 8,
                         lambda_=0.8,
                         patience = 50,
                         pretraining_path ='/home/johann/Documents/model_64_Causal_Stride_shift_4')


ts=3
self = model_cnn
x = x_train
batch_size = 8
print('clip to 0.5')


y_train_ = np.concatenate([y_train, y_train] , axis = 1)
y_train_ = np.concatenate([y_train, y_train] , axis = 1)
y_test_ = np.concatenate([y_test, y_test] , axis = 1)


model_cnn.fit(
    train_dataset=(x_train, y_train_),
    val_dataset=(x_train, y_train_),
    test_dataset=(x_test, y_test_),
    num_epochs=500,
    save_steps=5,
    batch_size = 12,
    function = np.min,
    shift_step = 0, #3
    sdev_label =0, #0.11
    fillgaps=0,
    feat_noise = 0, #0.2
    patience = 50,
    forget = 0,
    reduce_lr = True,
    model_directory='/home/johann/Documents/model_16_',
)



#CONSOLE 15 : clip 0.15 ; console 16 : clip to 0.001-1.0
model_cnn.load_weights('/home/johann/Documents/model_v5_' + str(ts) + '/best_model')
t, _ = model_cnn.predict(x_test)
plt.scatter(y_test, t)
plt.show()
mean_absolute_error(y_test, t)
plt.show()
import pandas as pd
df = pd.DataFrame([t.flatten(), y_test.flatten(),np.sqrt(np.exp(_sig.flatten()))]).T
df.columns = ['preds', 'true', 'uncertainty']
df['scaled'] = (df['uncertainty']/(df['preds']+1e-2))

plt.hist(df['uncertainty'], bins=16)
plt.show()
threshold = np.quantile(df['uncertainty'],0.95)
df = df[df.uncertainty < threshold]
plt.scatter(df['true'], df['preds'])
plt.show()
r2_score(df['true'], df['preds'])
mean_absolute_error(df['true'], df['preds'])


plt.scatter(y_test,np.sqrt(np.exp(_sig)))
plt.show()
plt.scatter(y_test,t)
plt.show()
mean_squared_error(y_test,t)
r2_score(y_test,t)

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1)
plt.scatter(y_test.flatten(),t.flatten(), color = 'red')
plt.errorbar(y_test.flatten(),t.flatten(), yerr=1.5*np.sqrt(np.exp(_sig).flatten()), fmt="o")
plt.show()

np.corrcoef(y_test.flatten(),t.flatten())



'''
model_cfg_cnn_stride = { #was working great with timeshift = p :.5, t=2, noise 0.1 feat_noise 0.3
    "learning_rate": 5e-4,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 64, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 64,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "n_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "SAME",
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'GlobalAveragePooling1D',
    "loss": "mse", #huber was working great for 2020 and 2021
    "enumerate" : True,
    "metrics": "r_square"
}
############################################
# Model configuration CNN 2019
model_cfg_cnn_2019 = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 64, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 128,
    "nb_fc_stacks": 1, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "n_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "CAUSAL",
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'GlobalAveragePooling1D',
    "loss": "mse", #huber was working great for 2020 and 2021
    "enumerate" : True,
    "metrics": "r_square"
}
#MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn_2019)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 8,
    function = np.min,
    shift_step = 0, #3
    sdev_label =0.1, #0.1
    feat_noise = 0.2, #0.2
    patience = 50,
    forget = True,
    #pretraining_path ='/home/johann/Documents/model_64_Stride_SAME',
    model_directory='/home/johann/Documents/model_512',
)
'''
'''
# Model configuration CNN other years
model_cfg_cnn = {
    "learning_rate": 10e-4,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 64, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 64,
    "nb_fc_stacks": 2, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "n_strides" :2,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "SAME",
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'GlobalAveragePooling1D',
    "loss": "mse", #huber was working great for 2020 and 2021
    "enumerate" : True,
    "metrics": "r_square"
}
#MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn)
# Prepare the model (must be run before training)
model_cnn.prepare()
pretraining = False
cotraining = False
if pretraining:
    x_pretrain = np.concatenate([x_train, x_test], axis = 0)
    model_cnn.pretraining(x_pretrain,
                          pretraining_path='/home/johann/Documents/model_16_Flatten_SAME',
                          num_epochs=100, shift=0, lambda_ = 0.1)
if cotraining:
    model_cnn.cotraining(train_dataset=(x_train, y_train),
                         val_dataset=(x_test, y_test),
                         num_epochs=500,
                         batch_size = 16,
                         lambda_=0.8,
                         patience = 100,
                         pretraining_path ='/home/johann/Documents/model_16_Flatten_SAME')
ts=3
self = model_cnn
x = x_train
batch_size = 8
model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 8,
    function = np.min,
    shift_step = 2, #3
    sdev_label =0.1, #0.1
    feat_noise = 0.2, #0.2
    patience = 50,
    forget = True,
    #pretraining_path ='/home/johann/Documents/model_16_Flatten_SAME',
    model_directory='/home/johann/Documents/model_512',
)

'''
########################################################################################################################
########################################################################################################################

# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.8,
    "nb_conv_filters": 128,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 2048,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [1,1],
    "n_strides" : [1,1],
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
