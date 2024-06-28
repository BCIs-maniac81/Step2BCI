# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def data_trials_plot(inputData=None, trials=None, xlim=None, ylim=[None, None],
                     sfreq=256, ch_ax=-1, t_ax=0, channels=None, **kwargs):
    if kwargs.get('figsize'):
        if kwargs['figsize']:
            figsize = kwargs['figsize']
        else:
            figsize = (10, 5)
    else:
        figsize = (10, 5)
    ax = []
    inputData = np.asarray(inputData).copy()
    if inputData.ndim == 2:
        numb_ch = inputData.shape[ch_ax]
        N_samp = inputData.shape[t_ax]
        t = np.linspace(start=0, stop=N_samp / sfreq, num=N_samp, endpoint=True)
        colors = ['green', 'steelblue', 'red', 'blue', 'orange', 'purple', 'yellow', 'brown']
        if kwargs.get('sharex') and kwargs.get('sharey'):
            fig, ax = plt.subplots(numb_ch, 1, figsize=figsize, sharex=kwargs['sharex'],
                               sharey=kwargs['sharey'])
        else:
            fig, ax = plt.subplots(numb_ch, 1, figsize=figsize)

        for idx in range(numb_ch):
            # ax.append(fig.add_subplot(numb_ch, 1, idx + 1))
            ax[idx].plot(t, inputData[:, idx], color=colors[idx], lw=0.8)
            ax[idx].set_title(kwargs['title'] + ' - Canal: ' + channels[idx], fontsize=14)
            ax[idx].set_xlabel(kwargs['xlabel'], fontsize=12)
            ax[idx].set_ylabel(kwargs['ylabel'], fontsize=12)
            ax[idx].axis([0, xlim, ylim[0], ylim[1]])
            ax[idx].grid(color="green", alpha=0.2)
            if trials is not None:
                for tr in trials:
                    ax[idx].axvline(tr / sfreq, color='black', lw=1.5)
        plt.tight_layout()
        plt.show()
    elif inputData.ndim == 1:
        numb_ch = 1
        N_samp = inputData.shape[t_ax]
        t = np.linspace(start=0, stop=N_samp / sfreq, num=N_samp, endpoint=True)
        colors = ['green', 'steelblue', 'red', 'blue', 'orange', 'purple', 'yellow', 'brown']
        fig = plt.figure(figsize=(10, 2))
        ax.append(fig.add_subplot(numb_ch, 1, 1))
        ax[0].plot(t, inputData[:], color=colors[0], lw=0.8)
        ax[0].set_title(kwargs['title'] + ' - Canal: ' + channels[0], fontsize=14)
        ax[0].set_xlabel(kwargs['xlabel'], fontsize=12)
        ax[0].set_ylabel(kwargs['ylabel'], fontsize=12)
        ax[0].axis([0, xlim, ylim[0], ylim[1]])
        ax[0].grid(color="green", alpha=0.2)
        if trials is not None:
            for tr in trials:
                ax[0].axvline(tr / sfreq, color='black', lw=1.5)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError('Not matching dimension !.')
        pass


def trials_psd_plot(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None, trials_ax=0, **kwargs):
    plt.figure(figsize=(12, 5))
    nchans = len(chan_ind)
    # 3 tracés au max pour chaque ligne
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)
    for i, ch in enumerate(chan_ind):
        plt.subplot(nrows, ncols, i + 1)
        for j, cl in enumerate(trials_PSD.keys()):
            plt.plot(freqs, np.mean(trials_PSD[cl][:, :, ch], axis=trials_ax), label=cl, color=kwargs['colors'][j])
        # All plot decoration below...
        plt.xlim(1, 30)
        if maxy is not None:
            plt.ylim(0, maxy)
        plt.grid()
        plt.xlabel(kwargs['xlabel'], fontsize=16)
        plt.ylabel(kwargs['ylabel'], fontsize=14)
        if chan_lab is None:
            plt.title('Channel %d' % (ch + 1))
        else:
            plt.title(chan_lab[i], fontsize=16)
        plt.legend()
    plt.tight_layout()
    plt.show()


def average_power_plot(inputData=None, channels=None, classes=None, bands=None, trial_period=[0, 8], numb_cls=None,
                       numb_ch=None, colors=None, sfreq=256, figsize=(14, 4), xlabel='Temps ($s$)',
                       ylabel='Puissance Moyenne ($U$)'):
    """
inputData doit être une liste des listes
chaque sous-liste contient le nombre des canaux pour la même classe
    """
    if bands is None:
        bands = ['alpha']
    if numb_ch is None:
        numb_ch = len(inputData[0])
    if numb_cls is None:
        numb_cls = len(inputData)
    for band in bands:
        fig, ax = plt.subplots(nrows=1, ncols=numb_cls, figsize=figsize)
        plt.suptitle(f'Imagination motrice - Plage {band}', fontsize=16)
        for idx in range(numb_cls):
            for ch in range(numb_ch):
                t = np.linspace(trial_period[0], trial_period[1],
                                len(inputData[idx][ch][trial_period[0] * sfreq: trial_period[1] * sfreq]))
                if colors is None:
                    ax[idx].plot(t, inputData[idx][ch][trial_period[0] * sfreq: trial_period[1] * sfreq], lw=0.4,
                                 label=f'{channels[ch]}')
                else:
                    ax[idx].plot(t, inputData[idx][ch][trial_period[0] * sfreq: trial_period[1] * sfreq],
                                 color=colors[ch], lw=1.2, label=f'{channels[ch]}')
            ax[idx].set_title(f'IM: classe {classes[idx]}', fontsize=16)
            ax[idx].set_xlabel(xlabel, fontsize=16)
            ax[idx].set_ylabel(ylabel, fontsize=16)
            ax[idx].grid(alpha=0.5, color="brown")
            # ax[idx].legend(loc='upper right', numpoints=1)
        plt.tight_layout()
        plt.show()


def trial_band_plot(inputData=None, band=None, pos=None, alpha=None, colors=None):
    if colors is None:
        colors = ['green', 'steelblue', 'red', 'blue', 'orange', 'purple', 'yellow', 'brown']
    if alpha is None:
        alpha = {0: 1, 1: 0.8, 2: 0.6}
    if band is None:
        band = {'alpha': [7, 12], 'beta': [13, 24]}
    if pos is None:
        pos = {0: 'C3', 1: 'Cz', 2: 'C4'}
    fig, ax = plt.subplots(nrows=len(band), ncols=1, sharex=True, sharey=True)
    for i in range(len(band)):
        ax[i].set_title(f'Bande {list(band.keys())[i]}')
        for j in range(len(pos)):
            ax[i].plot(inputData[i][:, j], label=str(pos[j]), color=str(colors[j]), alpha=float(alpha[j]))
        ax[i].grid()
        ax[i].legend(loc='best')
        plt.tight_layout()
    plt.show()


def all_trials_classes_plot(inputData=None, bands=None, classes=None, pos=None, xlabel='éch.',
                            ylabel='Amplitude ($u$V)', ylim=[None, None], figsize=(14, 8)):
    if bands is None:
        bands = ['band1', 'band2']
    if classes is None:
        classes = ['class1', 'class2']
    if pos is None:
        pos = ['CH1', 'CH2', 'CH3']

    for i in bands:
        if i in inputData.keys():
            wave = inputData[i]
        else:
            pass
        for idx, class_ in enumerate(wave.keys()):
            fig, ax = plt.subplots(len(pos), 1, figsize=figsize)
            plt.suptitle(f'les essais extraits dans la plage {i} pour la classe {class_}: {classes[idx]}')
            for j in range(wave[class_].shape[-1]):
                ax[j].set_xlabel(xlabel)
                ax[j].set_ylabel(ylabel)
                for k in range(wave[class_].shape[0]):
                    ax[j].plot(np.linspace(0, wave[class_].shape[1] - 1, wave[class_].shape[1]),
                               wave[class_][k, :, j])
                ax[j].grid()
                ax[j].axis([0, wave[class_].shape[1], ylim[0], ylim[1]])
            plt.tight_layout()
            plt.show()


def fir_kaiser_win_plot(inputData=None, signal_out=None, rythm=None, W=None, M=None, L=None, h=None, w=None, a=None,
                        ch_num=None, sfreq=256):
    plt.figure(10)
    plt.subplot(2, 2, 1)
    plt.plot(W, color='indigo', lw=1.5)
    plt.title('Fenêtre Kaiser')
    plt.xlabel("échs.")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(h, color='g', lw=1.5)
    plt.title('Réponse impulsionnelle fenêtrée')
    plt.xlabel('échs.')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(w / (2 * np.pi) * sfreq, 20 * np.log10(np.abs(a)), color="r", lw=0.9)
    plt.title("Réponse fréquentielle")
    plt.xlabel("Fréquence[Hz]")
    plt.ylabel("|H(w)|")
    plt.xlim(0.0, 150)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(inputData[:, ch_num])
    plt.plot(signal_out[M // 2:(M // 2) + L], color='purple', lw=1.2)
    plt.xlabel("échs.")
    plt.ylabel("Amplitude")
    plt.title("EEG + Rythme EEG:{0}".format(rythm))
    plt.grid()
    plt.tight_layout()
    plt.show()


def wavelets_decomp_plot(inputData=None, level=6, classes=None, channels=None, colors=None, alpha=None, savefig=False):
    """
    inputData est sous forme d'un array de 3 dimension (X, Y, Z).
    X : est le nombre des classes
    Y : représente les coefficients de basse fréquence et de haute fréquence.
    Z : est le nombre des canaux.
    Chaque classe contient un nombre quelconque des canaux.
    inputData = np.array([[['lowfreq_c3_class1', 'lowfreq_c4_class1'],
                  ['highfreq_c3_class1', 'highfreq_c4_class1']],
                 [['lowfreq_c3_class2', 'lowfreq_c4_class2'],
                  ['highfreq_c3_class2', 'highfreq_c4_class2']]])
    """
    numb_cls = inputData.shape[0]
    numb_ch = inputData.shape[-1]
    if classes is None:
        classes = ['Left', 'Right']
    if channels is None:
        channels = ['C3', 'C4']
    if colors is None:
        colors = ['green', 'red']
    if alpha is None:
        alpha = [0.95, 0.75]
    for i in range(numb_cls):
        fig, ax = plt.subplots(level + 1, 1, figsize=(18, 12), sharex=False, sharey=False)
        plt.suptitle(f'Coefficients - classe: {classes[i]} - ', fontsize=16)
        for j in range(level + 1):
            if j == 0:
                title = "Coefficient moyens, Niveau=%d" % level
                text = "cA_mean_trials"
                for k in range(numb_ch):
                    ax[j].set_title(title)
                    ax[j].plot(inputData[i, j, k][text], lw=1.2, label=f'{channels[k]}, classe {classes[i]}',
                               color=colors[k], alpha=alpha[k])
                    ax[j].axis([0, inputData[i, j, k][text].shape[0], None, None])
                    ax[j].legend(loc='upper right')
                ax[j].grid()

            else:
                title = "Coefficients de différence, Niveau=%d" % (7 - j)
                text = "cD_mean_level%d" % (7 - j)
                for k in range(numb_ch):
                    ax[j].set_title(title)
                    ax[j].plot(inputData[i, 1, k][text], lw=1.2, label=f'{channels[k]}, classe {classes[i]}',
                               color=colors[k], alpha=alpha[k])
                    ax[j].axis([0, inputData[i, 1, k][text].shape[0], None, None])
                    ax[j].legend(loc='upper right')
                ax[j].grid()
            plt.tight_layout()
        if savefig:
            plt.savefig(f'wavelets_decomp {i}.jpg', dpi=400)
    plt.show()


#
# def psd_semilog_plot(inputData=None, classes=None, channels=None, colors=None, figsize=(8, 4),
#                      xlabel='Fréquence ($Hz$)', ylabel='PSD ($U$)'):
#     """
#     inputData est sous forme d'un array de 3 dimension (X, Y, Z).
#     X : est le nombre des classes
#     Y : est le nombre des canaux.
#     Z : représente les fréquences et les valeurs PSD.
#     inputData = np.array([[[freqs_c3_class1, welch_c3_class1],
#                            [freqs_c4_class1, welch_c4_class1]],
#                           [[freqs_c3_class2, welch_c3_class2],
#                            [freqs_c4_class2, welch_c4_class2]]])
#     """
#     if classes is None:
#         classes = ['Left', 'Right']
#     if channels is None:
#         channels = ['C3', 'C4']
#     if colors is None:
#         colors = ['green', 'red']
#     numb_ch = inputData.shape[1]
#     numb_cls = inputData.shape[0]
#     nrows = int(np.ceil(len(channels) / 2))
#     ncols = 2
#     k = 0
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#     for i in range(numb_ch):
#         if nrows < 2:
#             for j in range(numb_cls):
#                 ax[k].semilogy(inputData[j, i, -1], label=f'{channels[i]}, {classes[j]}')
#                 ax[k].set_xlabel(xlabel, fontsize=16)
#                 ax[k].set_ylabel(ylabel, fontsize=16)
#                 ax[k].legend(loc='upper right')
#             ax[k].grid()
#             if k == 0:
#                 k = 1
#             else:
#                 k = 0
#         else:
#             for j in range(numb_cls):
#                 ax[i, k].semilogy(inputData[j, i, -1], label=f'{channels[i]}, {classes[j]}')
#                 ax[i, k].set_xlabel(xlabel, fontsize=16)
#                 ax[i, k].set_ylabel(ylabel, fontsize=16)
#                 ax[i, k].legend(loc='upper right')
#             ax[i, k].grid()
#             if k == 0:
#                 k = 1
#             else:
#                 k = 0
#     plt.show()
#

def cwt_plot(t=None, period=None, period_log=None, power=None, contourlevels=None,
             extend=True, cmap=None, title=None, xlabel=None, ylabel=None, axs=None):
    if not extend:
        fig, axs = plt.subplots(figsize=(8, 8))
    im = axs.contourf(t, period_log, np.log2(power), contourlevels, extend='both', cmap=cmap)
    axs.set_title(title, fontsize=12)
    axs.set_ylabel(ylabel, fontsize=10)
    axs.set_xlabel(xlabel, fontsize=10)
    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    y_ticks = np.log2(yticks)
    axs.set_yticks(y_ticks)
    axs.set_yticklabels(yticks)
    axs.invert_yaxis()
    ylim = axs.get_ylim()
    axs.set_ylim(ylim[0], -1)
    # axs.tick_params(direction='out', length=6, width=4, colors='k', grid_color='r', grid_alpha=0.3, labelsize=12)
    axs.grid(True)
    if not extend:
        cbar_axs = fig.add_axes([0.95, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_axs, orientation="vertical")
    plt.tight_layout()
    return yticks, ylim


def average_data_plot(inputData=None, data_avg=None, t_val=None, t_avg=None, j=None, extend=True, axs=None):
    if not extend:
        fig, axs = plt.subplots(figsize=(12, 3))
    axs.plot(t_val, inputData, color="b", lw=.8, label="signal d'entrée")
    axs.plot(t_avg, data_avg, color="red", lw=1.8, label="Moyennage (n={})".format(j))
    axs.set_xlim([t_val.min(), t_val.max()])
    axs.set_xlabel('Temps [ech]', fontsize=10)
    axs.set_ylabel('Amplitude [U]', fontsize=10)
    axs.set_title('Signal et Moyennage', fontsize=12)
    axs.grid(True)
    axs.legend(loc="best", fontsize=8)
    plt.tight_layout()


def fft_power_plot(power_spectrum=None, A=None, freqs_nn=None, plot_method='Horizontal', yticks=None,
                   ylim=None, axs=None):
    if plot_method.lower() == 'horizontal':
        extend = False
        fig, axs = plt.subplots(figsize=(12, 3))
        axs.plot(freqs_nn, A, 'r', lw=2, label='FFT')
        axs.plot(freqs_nn, power_spectrum, 'k--', lw=0.8, label='Spectre de Puissance')
        axs.set_xlabel("Fréquence [Hz]", fontsize=10)
        axs.set_ylabel("Amplitude [U]", fontsize=10)
        axs.grid(True)
        axs.legend()
        plt.tight_layout()

    elif plot_method.lower() == 'vertical':
        extend = True
        if freqs_nn[0] == 0:
            freqs_nn[0] = (freqs_nn[2] - freqs_nn[1]) / 2
        scales = 1. / freqs_nn
        scales_log = np.log2(scales)
        # if not extend:
        #     fig, axs = plt.subplots(figsize=(12, 3))
        axs.plot(A, scales_log, 'r', lw=2, label='FFT')
        axs.plot(power_spectrum, scales_log, 'k--', lw=1., label='Spectre de puissance')
        axs.set_ylabel("Fréquence [Hz]", fontsize=10)
        axs.set_xlabel("Amplitude [U]", fontsize=10)
        if yticks is not None:
            axs.set_yticks(np.log2(yticks))
            axs.set_yticklabels(yticks)
        axs.invert_yaxis()
        if ylim is not None:
            axs.set_ylim(ylim[0], -1)
        axs.grid(True)
        axs.legend(loc="best", mode='expand', fontsize='small')
        plt.tight_layout()


def psd_plot(inputData=None, freqs=None, labels=None, channel_names=None, chs=None, xlim=None, ylim=None):
    if channel_names is not None and chs is not None:
        ch_idx = [channel_names.index(ch) for ch in chs]
    elif channel_names is None and chs is None:
        ch_idx = [0, 58, -1]  # c3, cz et c4
        chs = ["First Comp.", "Middle Comp.", "Last Comp."]
    else:
        ch_idx = [26, 28, 30]
        ch_idx = ["C3", "Cz", "C4"]
    nrows = int(np.ceil(len(ch_idx) / 3))
    ncols = int(min(len(ch_idx), 3))

    fig = plt.figure(figsize=(10, 4))
    plt.suptitle("A")
    ax = []
    colors = ["blue", "red"]
    if len(inputData.keys()) > 2:
        colors = []
        colors = colors + [None] * len(inputData.keys())
        print(colors)
    for idx, ch in enumerate(ch_idx):
        ax.append(fig.add_subplot(nrows, ncols, idx + 1))
        for i, [clr_, cls_] in enumerate(zip(colors, inputData.keys())):
            if labels is not None:
                cls_ = labels[i]
            ax[idx].plot(freqs, np.mean(inputData[list(inputData.keys())[i]][:, :, ch], axis=0),
                         color=clr_, lw=1.5, label=cls_)

        ax[idx].set_title("%s Power Spectral Density" % chs[idx])
        ax[idx].set_xlabel("Frequency $(Hz)$")
        ax[idx].set_ylabel("Amplitude $(u)$")
        ax[idx].axis([0, xlim, 0, ylim])
        ax[idx].legend(loc="best")
        ax[idx].grid(color="green", alpha=0.2)
    plt.tight_layout()
    plt.show()


def spectrogram_plot(inputData=None, segment_w=None, mode=None, cmap=None, sfreq=256):
    fig, ax = plt.subplots(figsize=(7, 7))
    Pxx, freqs, times, im = ax.specgram(x=inputData, Fs=sfreq, NFFT=segment_w, mode=mode, noverlap=segment_w // 8,
                                        cmap=cmap)
    ax.set_title("EEG data Spectrogram", fontsize=18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im).set_label('Intensity (dB)')
    plt.tight_layout()
    plt.show()
    return freqs, times, im, Pxx


def stft_plot(freqs, Zxx, title='', xlabel='', ylabel=''):
    fig, ax = plt.subplots()
    for i in range(0, Zxx.shape[-1], 25):
        ax.plot(freqs, np.abs(Zxx[:, i]), lw=0.7)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.axis([freqs.min(), freqs.max(), None, None])
    ax.grid()
    plt.tight_layout()
    plt.show()


def psd_semilog_plot(inputData=None, classes=None, channels=None, colors=None, figsize=(8, 4),
                     xlabel='Fréquence ($Hz$)', ylabel='PSD ($u$)'):
    """
    inputData est sous forme d'un array de 3 dimension (X, Y, Z).
    X : est le nombre des classes
    Y : est le nombre des canaux.
    Z : représente les fréquences et les valeurs PSD.
    inputData = np.array([[[freqs_c3_class1, welch_c3_class1],
                           [freqs_c4_class1, welch_c4_class1]],
                          [[freqs_c3_class2, welch_c3_class2],
                           [freqs_c4_class2, welch_c4_class2]]])
    """
    if classes is None:
        classes = ['Left', 'Right']
    if channels is None:
        channels = ['C3', 'C4']
    if colors is None:
        colors = ['green', 'red']
    numb_ch = inputData.shape[1]
    numb_cls = inputData.shape[0]
    nrows = int(np.ceil(len(channels) / 2))
    ncols = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, class_data in enumerate(inputData):
        for j, channel_data in enumerate(class_data):
            ax[i].semilogy(channel_data[1], label=f'{channels[j]}, {classes[i]}')
            ax[i].set_xlabel(xlabel, fontsize=16)
            ax[i].set_ylabel(ylabel, fontsize=16)
            ax[i].legend(loc='upper right')
        ax[i].grid()
    plt.show()


if __name__ == "__main__":
    print(f"This is the test section of plotter module")
