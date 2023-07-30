import numpy as np
from sklearn import preprocessing

####################################################
## This script contains several filters to apply on
## a numpy array with a single feature
## array shape=(n,)
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def sklearn_normalize(ndarray):
    ndarray = preprocessing.normalize(ndarray.reshape(-1, 1), norm='l1', axis=0).reshape(-1)
    return ndarray

def sklearn_scale(ndarray, f_range=[-1, 1]):
    ndarray = preprocessing.minmax_scale(ndarray, feature_range=f_range)
    return ndarray

def window_normalize(ndarray):
    # set up window size in seconds
    window_size = 3
    step_size = int(window_size/0.01/3)
    window_size = int(window_size/0.01)
    last_shift = 0
    shift_factor = 5
    for idx in range(0, len(ndarray)-window_size, step_size):
        # calculate shift
        shift = ndarray[idx:idx+window_size].sum() / window_size
        shift = (shift_factor*shift + last_shift) / (1 + shift_factor)
        # apply shift
        ndarray[idx:idx+window_size] = ndarray[idx:idx+window_size] - shift
        # update last shift
        last_shift = shift

    return ndarray


def center_data(ndarray):
    mean = ndarray.mean()
    ndarray = ndarray - mean

    return ndarray


def sqrt_scale(ndarray):
    minus = ndarray.copy()
    for idx in range(len(ndarray)):
        if ndarray[idx] > 0:
            minus[idx] = 1
        else:
            minus[idx] = -1
    ndarray = np.sqrt(abs(ndarray))
    ndarray = ndarray*minus

    return ndarray
