# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
import scipy.stats as stats
import preproc.dataPlotter as plotter


def data_power_n(inputData=None, power=2, t_ax=0):
    data = inputData.copy()
    return data ** power


def data_logarithm(inputData=None, mode='dec'):
    # Il y a "dec", "bin", "nep"
    if mode.lower() == 'dec':
        data = inputData.copy()
        return np.log10(data)
    elif mode.lower() == 'bin':
        data = inputData.copy()
        return np.log2(data)
    elif mode.lower() == 'nep':
        data = inputData.copy()
        return np.log(data)
    else:
        return 'Erreur, mode inconnu !'


def data_variance(inputData=None, t_ax=0):
    data = inputData.copy()
    return np.var(data, axis=t_ax)


def data_covariance(inputData1=None, inputData2=None):
    data1 = inputData1.copy()
    data2 = inputData2.copy()
    return np.cov(data1, data2)


def data_correlation(inputData1=None, inputData2=None, relation="linear", dist="gauss"):
    data1 = inputData1.copy()
    data2 = inputData2.copy()
    if relation.lower() == "linear" and dist.lower() == "gauss":
        pearson = True
        spearman = False
    elif relation.lower() == "nonlinear" and dist.lower() == "nongauss":
        pearson = False
        spearman = True
    else:
        pearson = False
        spearman = True
    if pearson and not spearman:
        corr, p_value = stats.pearsonr(data1, data2)
    else:
        corr, p_value = stats.spearmanr(data1, data2)
    return corr


def signal_average(inputData=None, j=5, t_init=0, plot=True, extend=False, axs=None, sfreq=256, t_ax=0):
    Ts = 1. / sfreq
    N_samples = inputData.shape[t_ax]
    t_val = np.arange(0, N_samples) * Ts + t_init

    if isinstance(inputData, np.ndarray):
        data_arr = inputData
    else:
        data_arr = np.array(inputData)
    t_array = np.array(t_val)
    data_array = np.array(inputData)
    t_array.resize((N_samples // j, j), refcheck=False)
    data_array.resize((N_samples // j, j), refcheck=False)
    t_resh = t_array.reshape((-1, j))
    data_resh = data_array.reshape((-1, j))
    t_avg = t_resh[:, 0]
    data_avg = np.nanmean(data_resh, axis=1)
    if plot:
        plotter.average_data_plot(inputData, data_avg, t_val, t_avg, j, extend, axs=axs)
    return t_avg, data_avg


if __name__ == "__main__":
    print(f"This is the test section of data computing module")
