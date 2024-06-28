# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
import peakutils as pk


def subsampler(inputData=None, newsfreq=None, sfreq=256, t_ax=0):
    assert sfreq % newsfreq == 0, 'Erreur, la fréquence d\'échantillonnage doit être un multiple de la nouvelle ' \
                                  'fréquence d\'échantillonnage ! '
    res_ratio = int(np.floor(sfreq / newsfreq))
    new_t_ax = res_ratio * np.linspace(0, (np.size(inputData, axis=t_ax) / res_ratio) - 1,
                                       int(np.size(inputData, axis=t_ax) / res_ratio), dtype=int)
    if inputData.ndim == 1:
        y = np.zeros(new_t_ax.size, dtype=float)
    elif inputData.ndim == 2:
        y = np.zeros((new_t_ax.size, inputData.shape[1]), dtype=float)
    elif inputData.ndim == 3:
        y = np.zeros((inputData.shape[0], new_t_ax.size, inputData.shape[2]), dtype=float)

    for idx1, idx2 in enumerate(new_t_ax):
        if inputData.ndim == 1:
            y[idx1] = inputData[idx2]
        elif inputData.ndim == 2:
            y[idx1] = inputData[idx2]
        elif inputData.ndim == 3:
            y[:, idx1] = inputData[:, idx2]
        else:
            return 'format non-supporté'
    return y, new_t_ax


def baseline_classic_removal(inputData=None, times=None, bl_vector=[None, None], picks=None, t_ax=0):
    edge_min, edge_max = bl_vector
    if type(edge_min) == str or edge_min < 0:
        print("Erreur param. baseline ...")
    else:
        idx_to_right = np.where(times >= edge_min)[0]
        if len(idx_to_right) == 0:
            print("Valeur inf est trés grande!")
        else:
            idx_min = int(idx_to_right[0])
    if type(edge_max) == str or edge_max < 0 or edge_max <= edge_min:
        print("Erreur param. baseline ...")
    else:
        idx_to_left = np.where(times <= edge_max)[0]
        if len(idx_to_left) == 0:
            print("Valeur sup. est trés basse!")
        else:
            idx_max = int(idx_to_left[-1] + 1)  # car l'indice (-1) n'est pas inclut
    mean_value = np.mean(inputData[..., idx_min:idx_max], axis=t_ax, keepdims=True)

    if picks is None:
        inputData = inputData - mean_value
    else:
        for val in picks:
            inputData[..., val, :] = inputData[..., val, :] - mean_value[..., val, :]
    mean_values = np.arange(times.size)
    mean_values[:] = mean_value
    return inputData, mean_values


def baseline_removal(inputData=None, degree=2, max_iterations=50, tolerance=0.5):
    base = pk.baseline(y=inputData, deg=degree, max_it=max_iterations, tol=tolerance)
    y_remove = inputData - base
    return y_remove, base


def baseline_removal_2D(inputData=None, degree=2, max_iterations=50, tolerance=0.5):
    baseline_arr = np.zeros(inputData.shape)
    for jj in range(inputData.shape[-1]):
        baseline_arr[:, jj], _ = pk.baseline(inputData[:, jj], deg=degree, max_it=max_iterations, tol=tolerance)
    return baseline_arr


if __name__ == '__main__':
    print(f"This is the test section of dataEditor module")
