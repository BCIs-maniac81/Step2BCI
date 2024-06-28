# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
from sklearn import preprocessing


def min_max_scaler(inputData=None, f_range=(None, None), unit_scal=True, t_ax=0):
    data_std = (inputData - inputData.min(axis=t_ax)) / (
            inputData.max(axis=t_ax) - inputData.min(axis=t_ax))
    if not unit_scal:
        data_scaled = data_std * (f_range[1] - f_range[0]) + f_range[0]
        return data_scaled
    else:
        return data_std


def max_abs_scaler(inputData=None, t_ax=0):
    return inputData / np.abs(inputData).max(axis=t_ax)


def std_scaler(inputData=None, with_std=True, with_mean=True, t_ax=0):
    if with_mean:
        mean0 = np.mean(inputData, axis=t_ax)
    else:
        mean0 = 0
    if with_std:
        std0 = np.std(inputData, axis=t_ax)
    else:
        std0 = 1
    z = (inputData - mean0) / std0
    return z


def robust_scaler(inputData=None):
    if len(inputData.shape) == 1:
        inputData = inputData.reshape(-1, 1)
    scaler = preprocessing.RobustScaler()
    x_scal = scaler.fit_transform(inputData)
    return x_scal


def data_binarize(inputData=None, threshold=0, copy=True, t_ax=0):
    if copy:
        x1 = inputData.copy()
    else:
        x1 = inputData
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if x1[i][j] <= threshold:
                x1[i][j] = 0
            else:
                x1[i][j] = 1
    return x1


def data_normalizer(inputData=None, t_ax=0):
    nrm = np.linalg.norm(inputData, axis=t_ax)
    norm_data = (inputData.T / nrm).T
    return norm_data


if __name__ == '__main__':
    print(f"This is the test section of dataScaler module")