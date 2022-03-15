import matplotlib.pyplot as plt
import numpy as np
import random

def timeshift(x_, value = 4, proba = 0.5): #0.5 before
    x = x_.copy()

    def _shift(x, rand_unif):
        return np.roll(x, rand_unif)

    shift_list = []
    mask  = np.zeros((x.shape[0]), dtype='float32')

    for i in range(x.shape[0]):
        prob = random.random()
        if prob<proba:
            rand_unif = np.random.randint(value) + 1
            prob /= proba
            mask[i] = 1
            if prob < 0.5:
                rand_unif*=-1
                x[i,] = np.apply_along_axis(_shift, 0, x[i,], **{'rand_unif' : rand_unif})
            else:
                x[i,] = np.apply_along_axis(_shift, 0, x[i,],**{'rand_unif' : rand_unif})
            shift_list.append(rand_unif)
        else:
            shift_list.append(0)

    return x, shift_list, mask




def feature_noise(x_batch, value = 0.2, proba = 0.15):

    ts_masking = x_batch.copy()
    mask = np.zeros((ts_masking.shape[0], ts_masking.shape[1],), dtype='float32')

    for i in range(ts_masking.shape[0]):
        for j in range(ts_masking.shape[1]):
            prob = random.random()
            if prob < proba:
                prob /= proba
                mask[i, j] = 1.0
                if prob < 0.5:
                    ts_masking[i, j, :] += np.random.uniform(low=-value, high=0, size=(x_batch.shape[2]))
                else:
                    ts_masking[i, j, :] += np.random.uniform(low=0, high=value, size=(x_batch.shape[2]))
    return ts_masking, mask


def noisy_label(y_, stdev =0.05, proba = 0.5):
    y = y_.copy()
    y = y.reshape(y.shape[0], 1)

    for i in range(y.shape[0]):
        prob = random.random()
        if prob < proba:
            y[i,:] = y[i,:] + np.random.normal(0, stdev, 1)

        y[i,:] = max(y[i,:], 0)
        y[i,:] = min(y[i,:], 1)

    return y


def fill_gaps(ts, rdm_number, interval = (10, 30), id_argmax = 10):
    import random
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    tmin, tmax = interval
    new_x = ts.copy()
    argmax = np.argmax(ts[tmin:tmax, id_argmax])

    prob = random.random()
    if prob < 0.5:
        lst_add = [argmax + i for i in range(-rdm_number, 0, 1)]
    else:
        lst_add = [argmax + i for i in range(0, rdm_number, 1)]

    for i in lst_add:
        new_x = np.insert(new_x, i, np.nan, axis=0)

    nans, x = nan_helper(new_x)

    np_output = np.zeros((ts.shape[0], ts.shape[1]))

    for d in range(new_x.shape[-1]):
        id_nan, ts_nan = nans[:,d], new_x[:,d]
        ts_nan[id_nan] = np.interp(x(id_nan), x(~id_nan), ts_nan[~id_nan])
        if prob < 0.5:
            np_output[:,d] = ts_nan[rdm_number:]
        else:
            np_output[:, d] = ts_nan[:-rdm_number]

    return np_output

def apply_fill_gaps(x_batch, value = 2, proba = 0.15):

    ts_masking = x_batch.copy()
    mask = np.zeros((ts_masking.shape[0]), dtype='float32')

    for i in range(ts_masking.shape[0]):
        prob = random.random()
        if prob < proba:
            prob /= proba
            mask[i, ] = 1.0
            ts_masking[i, ] = fill_gaps(x_batch[i,],
                                        np.random.randint(value) + 1,
                                        interval = (10, 30),
                                        id_argmax = 10)

    return ts_masking, mask


def data_augmentation(x_train_, y_train_, shift_step, feat_noise, sdev_label, fillgaps):
    if shift_step:
        x_train_, _, _ = timeshift(x_train_, shift_step)
    if feat_noise:
        x_train_, _ = feature_noise(x_train_, feat_noise)
    if sdev_label:
        y_train_ = noisy_label(y_train_, sdev_label)
    if fillgaps:
        x_train_, _ = apply_fill_gaps(x_train_, value = fillgaps)

    return x_train_, y_train_
