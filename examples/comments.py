#2019 ; ema 0.99
model_cfg_cnn_stride = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 64, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 256,
    "nb_fc_stacks": 1, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "nb_conv_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "CAUSAL",
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'Flatten',
    "loss": "mse", #huber was working great for 2020 and 2021
    "enumerate" : True,
    "metrics": "r_square"
}
model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 16,
    function = np.min,
    shift_step = 2, #3
    sdev_label =0.1, #0.1
    feat_noise = 0.3, #0.2
    patience = 60,
    forget = 2,
    #finetuning = True,
    #pretraining_path ='/home/johann/Documents/model_64_Stride_SAME',
    model_directory='/home/johann/Documents/model_16',
)
#Achieve around 0.15 at 400 epochs for 2019 ; if fc = 128 : achieve earlier (355) and higher but more unstable => lower lr? but achieve 0.19 :0
#with two fc + global : maybe more epochs (around 0.07 from 400 to 500)

##################################
##################################
#FOR 2020 : Impressive ==> stride 1 : 2 : 2 ; ema 0.99
model_cfg_cnn_stride = {
    "learning_rate": 10e-4,
    "keep_prob" : 0.5, #should keep 0.8
    "nb_conv_filters": 64, #wiorks great with 32
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 128,
    "nb_fc_stacks": 1, #Nb FCN layers
    "fc_activation" : 'relu',
    "kernel_size" : 1,
    "nb_conv_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "CAUSAL",
    "kernel_regularizer" : 1e-6,
    "emb_layer" : 'Flatten',
    "loss": "mse", #huber was working great for 2020 and 2021
    "enumerate" : True,
    "metrics": "r_square"
}

#MODEL 64 128 with drop out 0.5 works great on 2019
model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn_stride)
# Prepare the model (must be run before training)
model_cnn.prepare()

model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_test, y_test),
    num_epochs=500,
    save_steps=5,
    batch_size = 8,
    function = np.min,
    shift_step = 2, #3
    sdev_label =0.1, #0.1
    feat_noise = 0.3, #0.2
    patience = 100,
    forget = 1,
    #finetuning = True,
    #pretraining_path ='/home/johann/Documents/model_64_Stride_SAME',
    model_directory='/home/johann/Documents/model_16',
)
