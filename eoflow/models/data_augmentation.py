import numpy as np
import random

def timeshift(x_, value = 4, proba = 0.75):
    x = x_.copy()

    def _shift(x, rand_unif):
        return np.roll(x, rand_unif)

    shift_list = []

    for i in range(x.shape[0]):
        prob = random.random()
        if prob<proba:
            rand_unif = np.random.randint(value) +1
            prob /= proba
            if prob < 0.5:
                rand_unif*=-1
                x[i,] = np.apply_along_axis(_shift, 0, x[i,], **{'rand_unif' : rand_unif})
            else:
                x[i,] = np.apply_along_axis(_shift, 0, x[i,],**{'rand_unif' : rand_unif})
            shift_list.append(rand_unif)
        else:
            shift_list.append(0)

    return x, shift_list





def feature_noise(x_batch, value = 0.2, proba = 0.25):

    ts_masking = x_batch.copy()
    mask = np.zeros((ts_masking.shape[0], ts_masking.shape[1],), dtype=float)

    for i in range(x_batch.shape[0]):
        for j in range(x_batch.shape[1]):
            prob = random.random()
            if prob < proba:
                prob /= proba
                mask[i, j] = 1.0
                if prob < 0.5:
                    x_batch[i, j, :] += np.random.uniform(low=-value, high=0, size=(x_batch.shape[2]))
                else:
                    x_batch[i, j, :] += np.random.uniform(low=0, high=value, size=(x_batch.shape[2]))
    return x_batch, mask


def noisy_label(y, stdev =0.2, proba = 0.5):
    y = y.reshape(y.shape[0], 1)

    for i in range(y.shape[0]):
        prob = random.random()
        if prob < proba:
            y[i,:] = y[i,:] + np.random.normal(0, stdev, 1)

        y[i,:] = max(y[i,:], 0)
        y[i,:] = min(y[i,:], 1)

    return y

