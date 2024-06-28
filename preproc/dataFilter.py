# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import scipy.signal as signal
import preproc.dataPlotter as plotter


def low_pass(inputData=None, lowfreq=50, order=6, sfreq=256, t_ax=0):
    cutoff = lowfreq / sfreq
    bl, al = signal.butter(order, 2 * cutoff, btype='lowpass')
    return signal.filtfilt(bl, al, inputData, axis=t_ax)


def high_pass(inputData=None, highfreq=0.5, order=6, sfreq=256, t_ax=0):
    cutoff = highfreq / sfreq
    bl, al = signal.butter(order, 2 * cutoff, btype='highpass')
    return signal.filtfilt(bl, al, inputData, axis=t_ax)


def band_stop(inputData=None, lowfreq=45, highfreq=55, order=6, sfreq=256, t_ax=0):
    low_cutoff = lowfreq / sfreq
    high_cutoff = highfreq / sfreq
    b0, a0 = signal.butter(order, [2 * low_cutoff, 2 * high_cutoff], btype='bandstop')
    return signal.filtfilt(b0, a0, inputData, axis=t_ax)


def band_pass(inputData=None, lowfreq=0.5, highfreq=50, order=6, sfreq=256, t_ax=0):
    low_cutoff = lowfreq / sfreq
    high_cutoff = highfreq / sfreq
    b1, a1 = signal.butter(order, [2 * low_cutoff, 2 * high_cutoff], btype='bandpass')
    return signal.filtfilt(b1, a1, inputData, axis=t_ax)


def notch_filter(inputData=None, notch_freq=50, quality_fact=30, sfreq=256, t_ax=0):
    f0 = notch_freq / sfreq
    bn, an = signal.iirnotch(w0=2 * f0, Q=quality_fact, fs=sfreq)
    return signal.filtfilt(bn, an, inputData, axis=t_ax)


def fir_kaiser_win_filter(inputData=None, ch_num=1, AdB=56, width=None, cutoff=[None, None],
                          pass_zero='bandpass', plot=True, sfreq=256, t_ax=0):
    fmax = sfreq / 2
    f1 = cutoff[0] / fmax
    f2 = cutoff[1] / fmax
    theta_band = [0.5, 4.]
    delta_band = [4., 7.]
    alpha_band = [7., 12.]
    beta_band = [12., 30.]
    M, beta = signal.kaiserord(AdB, width / fmax)  # compute M and beta
    W = signal.kaiser(M, beta)  # kaiser window
    coeffs = signal.firwin(numtaps=M, cutoff=[f1, f2], pass_zero=pass_zero, window='boxcar')  # FIR coeffs
    h = W * coeffs  # impulse response
    w, a = signal.freqz(h)
    signal_out = signal.convolve(h, inputData[:, ch_num])
    L = inputData[:, ch_num].shape[0]
    if cutoff == delta_band:
        rythm = 'DELTA'
    elif cutoff == theta_band:
        rythm = 'THETA'
    elif cutoff == alpha_band:
        rythm = 'ALPHA'
    elif cutoff == beta_band:
        rythm = 'BETA'

    if plot:
        plotter.fir_kaiser_win_plot(inputData, signal_out, rythm, W, M, L, h, w, a, ch_num, sfreq)
    return M, beta, W, h, w, a, signal_out


if __name__ == '__main__':
    print(f"This is the test section of filter module")
