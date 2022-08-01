import matplotlib.pyplot as plt
import eoflow.models.tempnets_task.rnn_tempnets as rnn_tempnets
import numpy as np
import os



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
           test_x_s1, np.concatenate([test_x_s3, test_x_s2], axis = 1),\
           training_y, val_y, test_y


def reshape_array(x, T=13):
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x

path_experiments = '/home/johann/Documents/Experiments SUBTAB/SSL'

year = 2020
training_x_orig_ts,training_x_orig_st,\
val_orig_x_orig_ts,val_orig_x_orig_st, \
test_orig_x_orig_ts, test_orig_x_orig_st,\
training_orig_y, val_orig_y, test_orig_y = load_gdd_data(path_training=path_experiments,
                                          group=str(year))

training_x_orig_rts, val_orig_x_orig_rts, test_orig_x_orig_rts = reshape_array(training_x_orig_ts), \
                                                              reshape_array(val_orig_x_orig_ts), \
                                                              reshape_array(test_orig_x_orig_ts)

training_x_is_ts,training_x_is_st,\
val_is_x_is_ts,val_is_x_is_st, \
test_is_x_is_ts, test_is_x_is_st,\
training_is_y, val_is_y, test_is_y  = load_gdd_data(path_training=path_experiments,
                                                          group='2020_IS')

training_x_is_rts, val_is_x_is_rts, test_is_x_is_rts = reshape_array(training_x_is_ts[:, :-2], 7), \
                                                              reshape_array(val_is_x_is_ts[:,:-2], 7), \
                                                              reshape_array(test_is_x_is_ts[:,:-2], 7)


model_config = dict(keep_prob = 1.0,
                    loss = 'mse',
                    rnn_blocks = 2,
                    learning_rate= 10e-4,
                    rnn_units = 64,
                    factor = 4,
                    kernel_initializer = 'glorot_uniform',
                    kernel_regularizer = 0,
                    output_shape = list(training_x_orig_rts.shape)[1:],
                    rnn_layer =  'gru')


model_compiled = rnn_tempnets.VAERNN(model_config)
model_compiled.prepare()


model_compiled.fit(
    train_dataset = (training_x_is_rts, training_x_orig_rts),
    val_dataset = (val_is_x_is_rts, val_orig_x_orig_rts),
    test_dataset = (test_is_x_is_rts, test_orig_x_orig_rts),
    num_epochs=100,
    batch_size=8,
    model_directory='/home/johann/Documents/FO_64-128-64_RELU/' + str(year)
)



preds = model_compiled.predict(test_is_x_is_rts)

plt.plot(test_is_x_is_rts[10,:,0])
plt.plot(preds[10,1:,0])
plt.plot(test_orig_x_orig_rts[10,1:,0])
plt.show()



h, x_reco, task = model_compiled.forward_step(test_is_x_is, training=False)
i = 100
plt.plot(x_reco[i,:6])
plt.plot(test_is_x_is[i,:6])
plt.show()

model_compiled.fit_is(
    train_dataset=(training_x_is, training_x_orig, training_is_y),
    val_dataset=(val_is_x_is, val_is_y),
    test_dataset=(test_is_x_is, test_is_y),
    batch_size=8,
    num_epochs=200,
    model_directory = '/home/johann/Documents/FO_64-128-64_RELU/' + str(2020),
    save_steps = 5,
    finetuning = True,
    unfreeze=False,
    #model_kd=model_kd
)
