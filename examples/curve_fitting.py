import numpy as np
from scipy.optimize import curve_fit
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import pandas as pd
import os
import matplotlib.pyplot as plt

def doubly_logistic(middle, initial_value, scale, a1, a2, a3, a4, a5):
    '''
    α1 is seasonal minimum greenness
    α2 is the seasonal amplitude
    α3 controls the green-up rate
    α4 is the green-up inflection point
    α5 controls the mid-growing season greenness trajectory.
    :return:
    '''
    return initial_value + scale * np.piecewise(
        middle,
        [middle < a1, middle >= a1],
        [lambda y: np.exp(-(((a1 - y) / a4) ** a5)), lambda y: np.exp(-(((y - a1) / a2) ** a3))],
    )


def _fit_optimize_doubly(x_axis, y_axis, initial_parameters=None):
    bounds_lower = [
        np.min(y_axis),
        -np.inf,
        x_axis[0],
        0.15,
        1,
        0.15,
        1,
    ]
    bounds_upper = [
        np.max(y_axis),
        np.inf,
        x_axis[-1],
        np.inf,
        np.inf,
        np.inf,
        np.inf,
    ]
    if initial_parameters is None:
        initial_parameters = [np.mean(y_axis), 0.2, (x_axis[-1] - x_axis[0]) / 2, 0.15, 10, 0.15, 10]

    popt, pcov = curve_fit(
        doubly_logistic,
        x_axis,
        y_axis,
        initial_parameters,
        bounds=(bounds_lower, bounds_upper),
        maxfev=1000000,
        absolute_sigma=True,
    )

    return popt



file = pd.read_csv('./examples/LAI_S2_MPL_2017.csv', sep=';')

def fit_row(row):
    return doubly_logistic(x, row[0], row[1], row[2], row[3], row[4],
                           row[5], row[6])

#Get doy from 0 to 1
x = (file['doy'].values - file['doy'].values[0])/(file['doy'].values[-1] - file['doy'].values[0])

params = _fit_optimize_doubly(x, file['LAI_s2'].values)
fitted_lai = fit_row(params)

#Compare before and after fitting
plt.plot(x, file['LAI_s2'].values)
plt.plot(x, fitted_lai)
plt.show()

file['fitted_LAI'] = fitted_lai

file.to_csv('./examples/LAI_S2_MPL_2017.csv', index = False)

