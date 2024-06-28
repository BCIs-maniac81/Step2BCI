# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt

import preproc.dataPlotter as plotter

matplotlib.use('TkAgg')


def cwt_analysis(inputData=None, t_init=0, scales=None, axs=None, wavelet="cmor", cmap=plt.cm.seismic, extend=False,
                 plot=False, dB=False, sfreq=256, t_ax=0):
    [coeffs, freqs] = pywt.cwt(data=inputData, scales=scales, wavelet=wavelet, sampling_period=1. / sfreq)
    t = np.arange(0, inputData.shape[t_ax]) / sfreq + t_init
    pwr = (np.abs(coeffs)) ** 2
    powerdB = 10 * np.log10(pwr)
    if dB:
        power = powerdB
    else:
        power = pwr

    if plot:
        period = 1. / freqs
        period_log = np.log2(period)
        # levels = [-4, -3, -2, -1, 0, 1, 2, 3]
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        yticks, ylim = plotter.cwt_plot(t, period, period_log, power, contourlevels, extend, cmap,
                                        title="Scalogramme - Puissance du Signal", xlabel="Temps[s]",
                                        ylabel="Echelle", axs=axs)
    else:
        yticks = None
        ylim = None
    return coeffs, freqs, yticks, ylim


# def wavedec_dwt_analysis(inputData=None, data=None, wavelet="db1", level="Auto", mean_c=False, trials_format=True,
#                          sfreq=256, t_ax=0):
#     if data is not None:
#         inputData = data.copy()
#     inputData = np.asarray(inputData)
#     w = pywt.Wavelet(wavelet)
#     coeffs_avg = {}
#     coeffs_diff_temp = []
#     coeffs_diff = {}
#     cA_mean = {}
#     cD_mean_temp = []
#     cD_mean = {}
#     if inputData.ndim > 1:
#         # print("Dimension_2={0}".format(inputData.ndim))
#         if level == "Auto":
#             coeffs2D = pywt.wavedec(inputData, wavelet=w, level=None, axis=t_ax)
#         else:
#             coeffs2D = pywt.wavedec(inputData, wavelet=w, level=level, axis=t_ax)
#         level_ = len(coeffs2D) - 1
#         if not trials_format:
#             n_channels = inputData.shape[-1]
#             k = 0
#             for i in range(n_channels):
#                 coeffs_avg.update({"cA_CH%d" % (i + 1): (coeffs2D[0][:, i])})
#                 cA_mean.update({"cA_mean_CH%d" % (i + 1): np.nanmean(coeffs2D[0][:, i], axis=0)})
#
#                 for j in range(1, level_ + 1):
#                     coeffs_diff_temp.append(coeffs2D[j][:, i])
#                     cD_mean_temp.append(np.nanmean(coeffs2D[j][:, i], axis=t_ax))
#                 coeffs_diff.update({'cD_CH%d' % (i + 1): coeffs_diff_temp[k:k + level_]})
#                 cD_mean.update({'cD_mean_CH%d' % (i + 1): cD_mean_temp[k:k + level_]})
#                 k += level_
#             coeffs1D = None
#         else:
#             n_trials = inputData.shape[0]
#             for i in range(n_trials):
#                 coeffs_avg.update({"cA_trial%d" % (i + 1): (coeffs2D[0][i, :])})
#             cA_mean.update({"cA_mean_trials": np.nanmean(coeffs2D[0], axis=0)})
#             for j, k in enumerate(range(level_, 0, -1), start=1):
#                 coeffs_diff.update({"cD_level%d" % k: coeffs2D[j]})
#                 cD_mean.update({"cD_mean_level%d" % k: np.nanmean(coeffs2D[j], axis=0)})
#             coeffs1D = None
#     else:
#         # print("Dimension_1={0}".format(inputData.ndim))
#         if level == "Auto":
#             coeffs1D = pywt.wavedec(inputData, wavelet=w, level=None, axis=0)
#         else:
#             coeffs1D = pywt.wavedec(inputData, wavelet=w, level=level, axis=0)
#
#         level_ = len(coeffs1D) - 1
#         coeffs_avg.update({"c_avg": list(coeffs1D[0])})
#         cA_mean.update({"cA_mean": np.nanmean(coeffs_avg["c_avg"], axis=0)})
#         coeffs_diff.update({'c_diff': coeffs1D[1:]})
#         cD_mean = {'cD_mean': [np.nanmean(coeffs1D[j]) for j in range(1, level_ + 1)]}
#         coeffs2D = None
#     return coeffs_avg, coeffs_diff, cA_mean, cD_mean

def wavedec_dwt_analysis(inputData=None, data=None, wavelet="db1", level=None, trials_format=True, t_ax=0):
    if data is not None:
        inputData = data.copy()
    inputData = np.asarray(inputData)
    w = pywt.Wavelet(wavelet)
    coeffs_avg = {}
    coeffs_diff_temp = []
    coeffs_diff = {}
    cA_mean = {}
    cD_mean_temp = []
    cD_mean = {}
    cD_std_temp = []
    cD_std = {}
    if inputData.ndim > 1:
        # print("Dimension_2={0}".format(inputData.ndim))
        coeffs2D = pywt.wavedec(inputData, wavelet=w, level=level, axis=t_ax)
        level_ = len(coeffs2D) - 1
        if not trials_format:
            n_channels = inputData.shape[-1]
            k = 0
            for i in range(n_channels):
                coeffs_avg.update({"cA_CH%d" % (i + 1): (coeffs2D[0][:, i])})
                cA_mean.update({"cA_mean_CH%d" % (i + 1): np.nanmean(coeffs2D[0][:, i], axis=0)})

                for j in range(1, level_ + 1):
                    coeffs_diff_temp.append(coeffs2D[j][:, i])
                    cD_mean_temp.append(np.nanmean(coeffs2D[j][:, i], axis=t_ax))
                    cD_std_temp.append(np.std(coeffs2D[j][:, i], axis=t_ax))
                coeffs_diff.update({'cD_CH%d' % (i + 1): coeffs_diff_temp[k:k + level_]})
                cD_mean.update({'cD_mean_CH%d' % (i + 1): cD_mean_temp[k:k + level_]})
                cD_std.update({'cD_std_CH%d' % (i + 1): cD_std_temp[k:k + level_]})
                k += level_
        else:
            n_trials = inputData.shape[0]
            for i in range(n_trials):
                coeffs_avg.update({"cA_trial%d" % (i + 1): (coeffs2D[0][i, :])})
            cA_mean.update({"cA_mean_trials": np.nanmean(coeffs2D[0], axis=0)})
            for j, k in enumerate(range(level_, 0, -1), start=1):
                coeffs_diff.update({"cD_level%d" % k: coeffs2D[j]})
                cD_mean.update({"cD_mean_level%d" % k: np.nanmean(coeffs2D[j], axis=0)})
                cD_std.update({"cD_std_level%d" % k: np.std(coeffs2D[j], axis=0)})
    else:
        # print("Dimension_1={0}".format(inputData.ndim))
        if level == "Auto":
            coeffs1D = pywt.wavedec(inputData, wavelet=w, level=None, axis=0)
        else:
            coeffs1D = pywt.wavedec(inputData, wavelet=w, level=level, axis=0)

        level_ = len(coeffs1D) - 1
        coeffs_avg.update({"c_avg": list(coeffs1D[0])})
        cA_mean.update({"cA_mean": np.nanmean(coeffs_avg["c_avg"], axis=0)})
        coeffs_diff.update({'c_diff': coeffs1D[1:]})
        cD_mean = {'cD_mean': [np.nanmean(coeffs1D[j]) for j in range(1, level_ + 1)]}
        cD_std = {'cD_std': [np.std(coeffs1D[j]) for j in range(1, level_ + 1)]}
    return coeffs_avg, coeffs_diff, cA_mean, cD_mean, cD_std, level_


if __name__ == "__main__":
    print(f"This is the test section of wavelets analysis module")
