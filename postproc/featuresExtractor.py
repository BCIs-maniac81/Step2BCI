# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:52:24 2020

@author: Hafed-eddine BENDIB
"""
import sys
import numpy as np
import preproc.waveletAnalyzer as wvlt
from hurst import compute_Hc
from nolds import hurst_rs, dfa
import preproc.dataCalculator as calc
import postproc.featuresPlotter as featplt


def csp_feat(inputData=None, class_names=None, fitted=False, W=None, t_ax=0):
    n_trs = []
    n_samps = []
    n_chs = []
    mat_cov = {}
    csp_trials = {}
    clss_ = list(inputData.keys())
    sigma = 0
    for idx0, cls_ in enumerate(clss_):
        n_trs.append(inputData[cls_].shape[0])
        n_samps.append(inputData[cls_].shape[1])
        n_chs.append(inputData[cls_].shape[2])
        mat_cov.update({cls_: []})
        csp_trials.update({cls_: np.zeros((n_trs[idx0], n_samps[idx0], n_chs[idx0]))})
        for idx1 in range(n_trs[idx0]):
            mat_cov[cls_].append((inputData[cls_][idx1].T.dot(inputData[cls_][idx1])) / n_samps[idx0])
        mat_cov.update({cls_: np.mean(mat_cov[cls_], axis=0)})
        sigma += mat_cov[cls_]
    U, s, _ = np.linalg.svd(sigma)
    P = U.dot(np.diag(s ** -0.5))
    B, _, _ = np.linalg.svd(P.T.dot(mat_cov[clss_[-1]]).dot(P))
    if fitted:
        if W is None:
            print("W must not be None !!")
            sys.exit()
        else:
            W = W
    else:
        W = P.dot(B)

    for idx0, cls_ in enumerate(clss_):
        for idx1 in range(n_trs[idx0]):
            csp_trials[cls_][idx1] = (inputData[cls_][idx1].dot(W))
    return csp_trials, W


def logvar_feat(inputData=None, t_ax=0):
    log_var = np.log(calc.data_variance(inputData, t_ax))
    return log_var


def logvar_csp_composer(trials_csp_train, trials_csp_true, classes, target_codes, plot=False):
    # Calculer Log-var des ensembles d'apprentissage
    csp_train_cl1, csp_train_cl2 = trials_csp_train[classes[0]][:, :, [0, -1]], \
                                   trials_csp_train[classes[1]][:, :, [0, -1]]
    logvar_train_cl1, logvar_train_cl2 = logvar_feat(csp_train_cl1, t_ax=1), logvar_feat(csp_train_cl2, t_ax=1)

    # Calculer Log-var des ensembles de test
    csp_true_cl1, csp_true_cl2 = trials_csp_true[classes[0]][:, :, [0, -1]], trials_csp_true[classes[1]][:, :, [0, -1]]
    logvar_true_cl1, logvar_true_cl2 = logvar_feat(csp_true_cl1, t_ax=1), logvar_feat(csp_true_cl2, t_ax=1)
    # Créer l'ensemble d'apprentissage
    train = np.concatenate((logvar_train_cl1, logvar_train_cl2), axis=0)
    y_train = np.concatenate(
        (np.full((logvar_train_cl1.shape[0],), target_codes[0]), np.full((logvar_train_cl2.shape[0],),
                                                                         target_codes[1])), axis=0)
    # Créer l'ensemble de test
    true = np.concatenate((logvar_true_cl1, logvar_true_cl2), axis=0)
    y_true = np.concatenate((np.full((logvar_true_cl1.shape[0],), target_codes[0]), np.full((logvar_true_cl2.shape[0],),
                                                                                            target_codes[1])), axis=0)
    if plot:
        featplt.features_scatter_plot(logvar_train_cl1, logvar_train_cl2)
        featplt.features_scatter_plot(logvar_true_cl1, logvar_true_cl2)
    return train, true, y_train, y_true


def subBP_feat(inputData=None, trials=None, timeStep=4, window_a=None, window_b=None, sfreq=256, selected_band2=0):
    # nous devons retrouver les punchs de données dans une liste
    band1, band2 = inputData[0], inputData[1]
    if band1.ndim != 3 or band2.ndim != 3:
        print("Format de données inconnu !!")
    if band1.shape[1] > band2.shape[1]:
        band1 = band1[:, : band2.shape[1], :]
    elif band1.shape[1] < band2.shape[1]:
        band2 = band2[:, : band2.shape[1], :]
    set_nbr = band1.shape[0]
    samples_nbr = band1.shape[1]
    ch_nbr = band1.shape[2]
    features = np.zeros((len(trials), (band1.shape[0] * band1.shape[2]) + (band2.shape[2] * band2.shape[0])))
    # features = np.zeros((len(trials), (band1.shape[0] * band1.shape[2]) + band2.shape[2]))
    for idx, trial_nbr in enumerate(trials):
        idx0_a = trial_nbr + (sfreq * timeStep) + window_a[0]
        idx1_a = trial_nbr + (sfreq * timeStep) + window_a[1]
        idx0_b = trial_nbr + (sfreq * timeStep) + window_b[0]
        idx1_b = trial_nbr + (sfreq * timeStep) + window_b[1]
        if band1 is not None:
            epochs_a = np.zeros((band1.shape[0], (idx1_a - idx0_a), band1.shape[2]))
        if band2 is not None:
            epochs_b = np.zeros((band2.shape[0], (idx1_b - idx0_b), band2.shape[2]))
        for idxi in range(ch_nbr):
            epochs_a[:, :, idxi] = band1[:, idx0_a: idx1_a, idxi]
            epochs_b[:, :, idxi] = band2[:, idx0_b: idx1_b, idxi]
        epochs_a = (np.power(epochs_a, 2).mean(axis=1)).flatten()
        # epochs_b = (np.power(epochs_b, 2).mean(axis=1)).flatten()
        # epochs_b = (np.power(epochs_b, 2).mean(axis=1))[selected_band2]
        epochs_b = (np.power(epochs_b, 2).mean(axis=1)).flatten()
        features[idx, :] = np.log(np.concatenate((epochs_a, epochs_b)))
    print(features.shape)
    return features


def hjorth_feat(inputData=None):
    if not isinstance(inputData, np.ndarray):
        signal = np.asarray(inputData).copy
    else:
        signal = inputData.copy()
    n = len(signal)
    dy = np.gradient(signal)
    ddy = np.gradient(dy)
    sigma_y2 = (1 / n) * np.sum(signal ** 2)
    sigma_dy2 = (1 / n) * np.sum(dy ** 2)
    sigma_ddy2 = (1 / n) * np.sum(ddy ** 2)
    activity = sigma_y2
    mobility = np.sqrt(sigma_dy2 / sigma_y2)
    complexity = np.sqrt(sigma_ddy2 / sigma_dy2) / mobility
    return activity, mobility, complexity


def pfd_feat(inputData=None, t_ax=0):
    """
    Pterosian Fractal Dimension
    """
    if not isinstance(inputData, np.ndarray):
        signal = np.asarray(inputData).copy()
    else:
        signal = inputData.copy()
    n_samples = np.size(signal, axis=t_ax)
    d_signal = np.diff(signal, axis=t_ax)
    n_diff = np.size(d_signal, axis=t_ax)
    delta = 0
    for kk in range(1, n_diff):
        if d_signal[kk] * d_signal[kk - 1] < 0:
            delta = delta + 1
    n = np.log10(n_samples)
    n_ = np.log10(n_samples / (n_samples + 0.4 * delta))
    pfd = n / (n + n_)
    return pfd


def hurst_m1_feat(inputData=None, kind="random_walk", min_window=10, simplified=True, plot=True):
    """
    Exposant Hurst suivant le module hurst
    """
    H, c, [window_sizes, RS] = compute_Hc(series=inputData, kind=kind, min_window=min_window,
                                          simplified=simplified)
    data = [window_sizes, RS]
    if plot:
        featplt.hurst_m1_plot(H, c, RS, window_sizes, 'log', 'Intervalle de temps', 'Taux R/S', 'Droite de Régression')
    return H, c, data


def hurst_m2_feat(inputData=None, nvals=None, fit_method="RANSAC", plot=True, corrected=True, save_file=None):
    ret = hurst_rs(data=inputData, nvals=nvals, fit=fit_method, debug_plot=plot,
                   debug_data=True, corrected=corrected, plot_file=save_file)
    H = ret[0]
    xvals = ret[1][0]
    rsvals = ret[1][1]
    [m, c] = ret[1][2]
    return H, [m, c], xvals, rsvals


def hfd_pr_feat(inputData=None, kmax=None, rcond=None):
    """
    Higuchi Fractal Dimension
    """
    L = []
    x = []
    N = len(inputData)
    for k in range(1, kmax):
        Lk = 0
        for m in range(0, k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            Lmk = np.sum(np.abs(inputData[m + idxs * k] - inputData[m + k * (idxs - 1)]))
            Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
            Lk += Lmk
        L.append(np.log(Lk / (m + 1)))
        x.append([np.log(1.0 / k), 1])
    (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=rcond)
    return p[0]


def dfa_feat(inputData=None, fit="RANSAC", plot=True, save_fig=None):
    """
    Detrended Fluctuation Analysis
    """
    pol0 = dfa(data=inputData, fit_exp=fit, debug_plot=plot, plot_file=save_fig)
    return pol0


def sampEntropy_feat(inputData=None, emb_dim=2, tol=0.2, plot=False, save_fig=None, ch_ax=-1, t_ax=0):
    """
    Sample Entropy
    """
    ext_data = [[], []]
    n_samp = inputData.shape[t_ax]
    chan = inputData.shape[ch_ax]
    if tol is None:
        tol = 0.2 * np.std(inputData[:, chan - 1])
    A = 0.
    B = 0.
    sig_i = np.array([inputData[i:i + emb_dim] for i in range(n_samp - emb_dim)])
    sig_j = np.array([inputData[i:i + emb_dim] for i in range(n_samp - emb_dim + 1)])
    emb_dim_p = emb_dim + 1
    sig_k = np.array([inputData[i:i + emb_dim_p] for i in range(n_samp - emb_dim_p + 1)])

    # Calculer A
    smlr = []
    for (ii, kk) in zip(sig_k, range(0, len(sig_k))):
        maxim = np.abs(ii - sig_k).max(axis=1)
        ext_data[1].extend(maxim[kk + 1:])
        sum_max_less_r = np.sum(maxim <= tol) - 1  # tol <=> tolerance
        smlr.append(sum_max_less_r)
    A = np.sum(smlr)

    # Calculer B
    smlr = []
    for (jj, kk) in zip(sig_i, range(0, len(sig_i))):
        maxim = np.abs(jj - sig_j).max(axis=1)
        ext_data[0].extend(maxim[kk + 1:])
        sum_max_less_r = np.sum(maxim <= tol) - 1  # tol <=> tolerance
        smlr.append(sum_max_less_r)
    B = np.sum(smlr)

    sampEn_ = -np.log(A / B)
    if plot:
        featplt.sampEn_dists_plot(distances=ext_data, tol=tol, m=emb_dim + 1,
                                  title="Sample Entropy {:.3f}".format(sampEn_), save_fig=save_fig)
    return sampEn_


def wavelets_feat(inputData=None, trials=None, timeStep=4, window=None, wavelet="db4",
                  max_level=None, level_to_extract=2, sfreq=256, t_ax=0):
    ch_nbr = inputData.shape[-1]
    epochs = np.array([[[]]], dtype=float)
    features = np.zeros((len(trials), 9))
    for idx, trial_nbr in enumerate(trials):
        idx0 = trial_nbr + (sfreq * timeStep) + window[0]
        idx1 = trial_nbr + (sfreq * timeStep) + window[1]
        epoch = inputData[idx0: idx1, :].reshape(1, idx1 - idx0, ch_nbr)
        if idx == 0:
            epochs = inputData[idx0: idx1, :].reshape(1, idx1 - idx0, ch_nbr)
        else:
            epochs = np.append(epochs, epoch, axis=0)
    for j in range(len(trials)):
        c_Avg, c_diff, cA_mean, cD_mean, cD_std, n_levels = \
            wvlt.wavedec_dwt_analysis(None, data=epochs[j, :, :], wavelet=wavelet,
                                      level=max_level, trials_format=False, t_ax=t_ax)

        # c3std = np.std(c_diff["cD_CH1"][n_levels - level_to_extract])
        # czstd = np.std(c_diff["cD_CH2"][n_levels - level_to_extract])
        # c4std = np.std(c_diff["cD_CH3"][n_levels - level_to_extract])
        c3std = cD_std['cD_std_CH1'][n_levels - level_to_extract]
        czstd = cD_std['cD_std_CH2'][n_levels - level_to_extract]
        c4std = cD_std['cD_std_CH3'][n_levels - level_to_extract]
        c3median = np.median(c_diff["cD_CH1"][n_levels - level_to_extract])
        czmedian = np.median(c_diff["cD_CH2"][n_levels - level_to_extract])
        c4median = np.median(c_diff["cD_CH3"][n_levels - level_to_extract])
        c3mean = cD_mean['cD_mean_CH1'][n_levels - level_to_extract]
        czmean = cD_mean['cD_mean_CH2'][n_levels - level_to_extract]
        c4mean = cD_mean['cD_mean_CH3'][n_levels - level_to_extract]
        # c3mean = np.mean(c_diff["cD_CH1"][n_levels - level_to_extract])
        # czmean = np.mean(c_diff["cD_CH2"][n_levels - level_to_extract])
        # c4mean = np.mean(c_diff["cD_CH3"][n_levels - level_to_extract])
        features[j, :] = np.array([c3median, c3std, c3mean, czmedian, czstd, czmean, c4median, c4std, c4mean])
        # features[j, :] = np.array([c3std, czstd, c4std, c3mean, czmean, c4mean])
    return features


if __name__ == "__main__":
    print(f"This is the test section of feature extraction module")
