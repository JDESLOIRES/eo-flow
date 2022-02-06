import numpy as np
import random

def add_random_shift(x, value = 1, proba = 0.25):
    def shift_pos(x): return np.roll(x, np.random.randint(value))
    def shift_neg(x): return np.roll(x, -np.random.randint(value))

    for i in range(x.shape[0]):
        prob = random.random()
        if prob < proba:
            prob /= proba
            if prob < 0.5:
                x[i,] = np.apply_along_axis(shift_pos, 0, x[i,])
            else:
                x[i,] = np.apply_along_axis(shift_neg, 0, x[i,])
    return x



def add_random_noise(x, value = 0.2, proba = 0.15):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            prob = random.random()
            if prob < proba:
                prob /= proba
                if prob < 0.5:
                    x[i, j, :] += np.random.uniform(low=-value, high=0, size=(x.shape[2]))
                else:
                    x[i, j, :] += np.random.uniform(low=0, high=value, size=(x.shape[2]))
    return x


def add_random_target(y, proba = 0.15):
    y = y.reshape(y.shape[0], 1)
    for i in range(y.shape[0]):
        prob = random.random()
        if prob < proba:
            prob /= proba
            if prob < 0.5:
                y[i,:] = y[i,:] + np.random.rand(1)/5
            else:
                y[i,:] = y[i,:] - np.random.rand(1)/5
        y[i,:] = max(y[i,:], 0)
        y[i,:] = min(y[i,:], 1)
    return y
