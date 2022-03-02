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
import numpy as np
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.pipeline import make_pipeline
from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate

########################################################################################################################
########################################################################################################################
def reshape_array(x, T=30):
    x = x.reshape(x.shape[0], x.shape[1] // T, T)
    x = np.moveaxis(x, 2, 1)
    return x


def npy_concatenate(path, prefix='training_x', T=30):
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
    return x

dict = {}

for year in range(2017,2022):
    dict[year] = []
    path = '/home/johann/Documents/Syngenta/cleaned_V2/' + str(year) +  '/fold_1'
    x_train = npy_concatenate(path, 'training_x')
    y_train = np.load(os.path.join(path, 'training_y.npy'))

    x_val = npy_concatenate(path, 'val_x')
    y_val = np.load(os.path.join(path, 'val_y.npy'))

    x_test = npy_concatenate(path, 'test_x')
    y_test = np.load(os.path.join(path, 'test_y.npy'))

    x_train_ = np.concatenate([x_train, x_val], axis=0)
    y_train_ = np.concatenate([y_train, y_val], axis=0)
    x_train_.shape

    y_list_preds = np.array([])

    for i in range(1,11):

        index =  np.random.choice(list(range(x_train_.shape[0])), size=int(x_train_.shape[0]*0.9),
                                   replace=False)
        print(len(set(index)))
        x_train, y_train = x_train_[index,:,:], y_train_[index]

        rocket = Rocket(
            num_kernels=100,normalise=False)

        rocket.fit(x_train)
        X_train_transform = rocket.transform(x_train)
        X_test_transform = rocket.transform(x_test)
        classifier = RidgeCV(alphas=np.logspace(-1, 100, 1000)).fit(X, y)
        classifier.fit(X_train_transform, y_train)
        preds = classifier.predict(X_test_transform)
        print(r2_score(y_test, preds.flatten()))
        if i == 1:
            y_list_preds = np.append(y_list_preds, preds.flatten())
        else:
            y_list_preds += preds.flatten()

    y_list_preds /=10
    print(r2_score(y_test, y_list_preds))
    dict[year].append(r2_score(y_test, y_list_preds))


plt.scatter(y_test, y_list_preds)
plt.show()

r2_score(y_test, y_list_preds)