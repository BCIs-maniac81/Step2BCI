# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
from scipy.fftpack import ifft, fft, fftfreq, rfft, rfftfreq
import scipy.signal as signal
from scipy.integrate import simps
import mne.time_frequency as tf
import preproc.dataFilter as filt
import preproc.dataPlotter as plotter


def spectral_analysis(inputData=None, sfreq=256, t_ax=0):
    N_samples = np.size(inputData, axis=t_ax)
    freqs = fftfreq(N_samples)
    freqs_half = np.abs(freqs[:N_samples // 2])
    fft_sig = fft(inputData, axis=t_ax)
    A = 2 * (1 / N_samples) * np.abs(fft_sig[:N_samples // 2])
    return A, freqs_half, freqs_half * sfreq


def periodogram_analysis(inputData=None, dB=False, sfreq=256, t_ax=0):
    c_freqs, c_psd = signal.periodogram(inputData, sfreq, axis=t_ax)
    if dB:
        c_psd = 10 * np.log10(c_psd)
    return c_freqs, c_psd


def welch_periodogram_analysis(inputData=None, sliding_win=None, dB=False, sfreq=256, t_ax=0):
    if len(inputData.shape) == 2:
        if inputData.shape[-1] == 1:
            inputData = inputData[:, 0]
        else:
            return 'Erreur Données > 1D non-supportées'
    # w_freqs, w_psd = signal.welch(inputData, sfreq, window="flattop", nperseg=sliding_win, scaling="spectrum",
    #                               axis=t_ax)
    w_freqs, w_psd = signal.welch(inputData, sfreq, window="hamming", nperseg=sliding_win, scaling="spectrum",
                                  axis=t_ax)
    if dB:
        w_psd = 10 * np.log10(w_psd)
    return w_freqs, w_psd


def multitaper_period_analysis(inputData=None, dB=False, adaptive=True, normalization='full', verbose=None, sfreq=256):
    if len(inputData.shape) == 2:
        if inputData.shape[-1] == 1:
            inputData = inputData[:, 0]
        else:
            return 'Erreur Données > 1D non-supportées'
    mt_psd, mt_freqs = tf.psd_array_multitaper(inputData, sfreq, adaptive=adaptive, normalization=normalization,
                                               verbose=verbose)
    if dB:
        mt_psd = 10 * np.log10(mt_psd)
    return mt_freqs, mt_psd


def spectral_entropy_analysis(inputData=None, method='periodogram', sliding_win=None, dB=False, normalize=False,
                              sfreq=256, t_ax=0):
    if method == 'periodogram':
        _, _psd = periodogram_analysis(inputData, dB, sfreq, t_ax)
    if method == 'welch':
        _, _psd = welch_periodogram_analysis(inputData, sliding_win, dB, sfreq, t_ax)
    psd_n = _psd / _psd.sum()
    se = -(psd_n * np.log2(psd_n)).sum()
    if normalize:
        se /= np.log2(psd_n.size)
    return se


def band_power_analysis(inputData=None, band=None, window_size=None, verbosity=False, normalization='length',
                        method='welch', dB=False, sfreq=256, t_ax=0):
    """Calculer la puissance moyenne du signal x dans une bande de fréquences spécifique.
        """
    band = np.asarray(band)
    low_band, high_band = band
    # Définir la taille de la fenêtre
    if window_size is not None:
        win = window_size * sfreq
    else:
        win = 2. * (1. / low_band) * sfreq

    inputData = inputData.copy()
    inputData = filt.band_pass(inputData, lowfreq=low_band, highfreq=high_band, order=2, sfreq=sfreq)

    if method == 'welch':
        freqs, psd = welch_periodogram_analysis(inputData, sliding_win=win, dB=dB, sfreq=sfreq, t_ax=t_ax)
    elif method == 'multitaper':
        freqs, psd = multitaper_period_analysis(inputData, dB=dB, adaptive=True,
                                                normalization=normalization, verbose=None, sfreq=sfreq)
    else:
        return 'Erreur, méthode inconnue!'
    f_res = freqs[1] - freqs[0]  # Résolution de fréquence
    idx_band = np.logical_and(freqs >= low_band, freqs <= high_band)
    band_power_abs = simps(psd[idx_band], dx=f_res)  # puissance de bande absolue
    total_power = np.abs(simps(psd, dx=f_res))
    band_power_rel = band_power_abs / total_power  # Puissance de bande relative
    if verbosity is True:
        print("La puissance totale est %.3f uV²/Hz " % total_power)
        print("La puissance de bande absolue est %.3f uV²/Hz " % band_power_abs)
        print("La puissance de bande relative est %.3f" % band_power_rel)
        print("Ce qui constitue %.3f de la puissance totale" % (100 * band_power_rel))
    return freqs, psd, idx_band, band_power_abs, band_power_rel


def average_power_analysis(inputData=None, sfreq=256, t_ax=0):
    data = np.asarray(inputData).copy()
    if data.ndim <= 2:
        t_ax = 0
    else:
        t_ax = -1
    return np.power(data, 2).mean(axis=t_ax)


def average_power(inputData=None, single_channel=False, dB=False, sfreq=256, t_ax=0):
    if single_channel:
        t_ax = 0
        nbr_ch = 1
        power = np.power(inputData, 2).mean(axis=t_ax)

    else:
        t_ax = -1
        nbr_ch = inputData.shape[1]
        power = np.zeros((nbr_ch, inputData.shape[0]))
        for j in range(nbr_ch):
            power[j, :] = np.power(inputData[:, j], 2).mean(axis=t_ax)
    return power


def stft_analysis(inputData=None, win='hann', nperseg=256, plot=True, sfreq=256, t_ax=0):
    freqs_00, times_00, Zxx_00 = signal.stft(x=inputData, window=win, fs=sfreq, axis=t_ax)
    if plot:
        plotter.stft_plot(freqs_00, Zxx_00, title='STFT[Zxx(f)] pour chaque segment',
                          xlabel='Fréquences (Hz)',
                          ylabel='Amplitude du spectre (u)')
    return freqs_00, times_00, Zxx_00


def short_time_ft(inputData=None, segment_w=256, N_samples=256, plot=True, sfreq=256, t_ax=0):
    win = signal.windows.hann(segment_w)
    real_fft = np.array(
        [rfft(inputData[i:i + segment_w] * win) for i in range(0, len(inputData) - segment_w, segment_w // 2)])
    Zxx_01 = real_fft * (2. / segment_w)
    duration = N_samples * (1. / sfreq)
    times_01 = np.linspace(0, duration, Zxx_01.shape[t_ax])
    freqs_01 = rfftfreq(segment_w, 1 / sfreq)
    if plot:
        plotter.stft_plot(freqs_01, Zxx_01.T,
                          title='STFT de Zxx(f) pour chaque segment',
                          xlabel='Fréquences (Hz)',
                          ylabel='Amplitude du spectre (U)')
    return freqs_01, times_01, Zxx_01


def spectrogram_analysis(inputData=None, segment_w=512, mode='psd', plot=False, cmap=None, sfreq=256, t_ax=0):
    if not plot:
        freqs, times, Sxx = signal.spectrogram(x=inputData, fs=sfreq, nperseg=segment_w, noverlap=None,
                                               nfft=None, mode=mode)
        r_Sxx = Sxx
        im = None
    else:
        freqs, times, im, r_Sxx = plotter.spectrogram_plot(inputData, segment_w, mode, cmap, sfreq)
    return freqs, times, im, r_Sxx


def fft_Power_Spectrum_analysis(inputData=None, plot=True, plot_method='Horizontal', yticks=None, ylim=None,
                                axs=None, dB=False, sfreq=256, t_ax=0):
    N_samples = np.size(inputData, axis=t_ax)
    N2 = 2 ** (int(np.log2(N_samples)) + 1)
    freqs_nn = np.linspace(0.0, sfreq / 2., N2 // 2)
    freqs_h = freqs_nn / sfreq
    fft_sig = fft(inputData, axis=t_ax)
    A = (2.0 / N2) * np.abs(fft_sig[0:N2 // 2])
    var0 = np.std(inputData) ** 2
    power_spectrum = var0 * np.abs(A) ** 2
    if plot:
        plotter.fft_power_plot(power_spectrum, A, freqs_nn, plot_method, yticks, ylim, axs)
    return power_spectrum


if __name__ == "__main__":
    print(f"This is the test section of data analysis module")
