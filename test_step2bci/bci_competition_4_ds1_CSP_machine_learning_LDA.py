# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:45:28 2021

@author: hafed-eddine BENDIB
"""

import numpy as np
import preproc.IOfiles as iof
import preproc.trialsProcessor as trproc
import preproc.dataPlotter as plotter
import preproc.dataAnalyzer as analysis
import preproc.dataFilter as filt
import postproc.featuresExtractor as featext
import postproc.featuresPlotter as featplt
import postproc.classifierTools as tools
import postproc.classification as clf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load data
X, trials, targets, attributes = iof.read_matlab_file(full_path='../datasets/berlindata/comp4_calib_ds1d/BCICIV_calib_ds1d.mat',
                                                      verbosity=False, dtype='EEG')

nsamples, nchannels = X.shape
sfreq = attributes['sfreq']
classes = attributes['Classes']
channels = attributes['Channels']
target_codes = np.unique(targets)
nclasses = len(classes)
cl1, cl2 = classes[0], classes[1]
"""
targets: -1: Imagination de la main gauche (Left) , 1: Imagination de la main droite (Right)
"""


print(f'Format des données EEG: {X.shape}')
print(f'Fréquence d\'échantillonnage: {sfreq}')
print(f'Nombre des canaux: {nchannels}')
print(f'Nombre d\'échantillons: {nsamples}')
print(f'Nom des canaux: {channels}')
print(f'Nombre d\'évènements: {len(trials)}')
print(f'Codes d\'évènements: {np.unique(targets)}')
print(f'classes cibles: {classes}')
print(f'Nombre de classes: {nclasses}')

'''
Les essais sont coupés dans l'intervalle [0,5-2,5 s] après l'apparition du marqueur.
'''


trials_data = trproc.trials_extract(inputData=X, trials=trials, targets=targets, window=[0.5, 2.5], version='berlin',
                                    sfreq=sfreq)
print('format des données pour la classe (left) -1 (n_trials, data, channel)', trials_data[target_codes[0]].shape)
print('format des données pour la classe (right) 1 (n_trials, data, channel)', trials_data[target_codes[1]].shape)

trials_filt_f = filt.band_pass(trials_data[target_codes[0]], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)
trials_filt_r = filt.band_pass(trials_data[target_codes[1]], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)

trials_filt = trproc.data_class_composer(trials_filt_f, trials_filt_r, classes=classes)

trials_csp, W = featext.csp_feat(trials_filt, fitted=False, t_ax=1)
'''Pour voir le résultat de l'algorithme CSP, nous traçons le log-var comme nous l'avons fait précédemment.'''

logvar_f = featext.logvar_feat(trials_csp[classes[0]], t_ax=1)
logvar_r = featext.logvar_feat(trials_csp[classes[1]], t_ax=1)

trials_logvar = trproc.data_class_composer(logvar_f, logvar_r, classes=classes)

params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var', xlabel='Canaux'
                                                                                                   '/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['g', 'purple'], **params)

freqs, psd_f = analysis.welch_periodogram_analysis(inputData=trials_csp[classes[0]], sliding_win=200, sfreq=100, t_ax=1)
freqs, psd_r = analysis.welch_periodogram_analysis(inputData=trials_csp[classes[1]], sliding_win=200, sfreq=100, t_ax=1)

trials_PSD = trproc.data_class_composer(psd_f, psd_r, classes=classes)
colors = ['green', 'red']

plotter.trials_psd_plot(trials_PSD, freqs, chan_ind=[0, 28, -1],
                        chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'], maxy=0.75, xlabel='Fréquence ('
                                                                                                       '$Hz$)',
                        ylabel='PSD', colors=colors)

featplt.features_scatter_plot(logvar_f, logvar_r)

# =========================================== Classification ==================================================
X_train, X_true = trproc.split_data_classes(trials_filt, classes=classes, target_codes=target_codes,
                                            ts=0.25, rs=1)

# Formation CSP seulement sur l'ensemble de formation X_train (left & right)
trials_csp_train, W = featext.csp_feat(X_train, fitted=False, t_ax=1)
trials_csp_train, _ = featext.csp_feat(X_train, fitted=True, W=W, t_ax=1)
# Appliquer CSP formé à l'ensemble de test

trials_csp_true, W_ = featext.csp_feat(X_true, fitted=True, W=W, t_ax=1)

# Calculer Log-var des deux ensembles
train_f = trials_csp_train[classes[0]][:, :, [0, -1]]
train_r = trials_csp_train[classes[1]][:, :, [0, -1]]

logvar_train_f = featext.logvar_feat(train_f, t_ax=1)
logvar_train_r = featext.logvar_feat(train_r, t_ax=1)
logvar_train = trproc.data_class_composer(logvar_train_f, logvar_train_r, classes=classes)
featplt.features_scatter_plot(logvar_train_f, logvar_train_r)

true_f = trials_csp_true[classes[0]][:, :, [0, -1]]
true_r = trials_csp_true[classes[1]][:, :, [0, -1]]
logvar_true_f = featext.logvar_feat(true_f, t_ax=1)
logvar_true_r = featext.logvar_feat(true_r, t_ax=1)
logvar_true = trproc.data_class_composer(logvar_true_f, logvar_true_r, classes=classes)
featplt.features_scatter_plot(logvar_true_f, logvar_true_r)

W_, b = clf.train_lda(logvar_train_f, logvar_train_r)
print(W_)
print(b)

featplt.features_scatter_plot(logvar_train_f, logvar_train_r, boundary_decision_plot=True, W=W_, b=b,
                              offset_xy=[0.5, 0.5])
featplt.features_scatter_plot(logvar_true_f, logvar_true_r, boundary_decision_plot=True, W=W_, b=b,
                              title='Données de Test', offset_xy=[0.5, 0.5])

# Matrice de confusion
conf_matrix = np.array([
    [(clf.apply_lda(logvar_true_f.T, W_, b) == 1).sum(), (clf.apply_lda(logvar_true_r.T, W_, b) == 1).sum()],
    [(clf.apply_lda(logvar_true_f.T, W_, b) == 2).sum(), (clf.apply_lda(logvar_true_r.T, W_, b) == 2).sum()]
])

print('Confusion matrix:')
print(conf_matrix)
print()
print('Accuracy: %.3f' % (np.sum(np.diag(conf_matrix)) / float(np.sum(conf_matrix))))

# ==================================== Classification par des apprentissage Automatique ================================
X_train, y_train = trproc.split_data_classes(logvar_train, for_classification=True)
X_true, y_true = trproc.split_data_classes(logvar_true, for_classification=True)
y_pred, LDA = clf.lda_classifier(X_train, y_train, X_true, solver='lsqr', shrinkage=True)
accuracy = tools.accuracy_clf(y_pred, y_true)
conf_ = tools.confusion_matrix(y_true, y_pred)
print('Précision LDA: ', accuracy)
print('Matrice de confusion: \n', conf_)

weights = LDA.coef_[0]
intercept = LDA.intercept_[0]
featplt.features_scatter_plot(logvar_train_f, logvar_train_r, boundary_decision_plot=True, W=weights, b=intercept,
                              offset_xy=[0.5, 0.5])
featplt.features_scatter_plot(logvar_true_f, logvar_true_r, boundary_decision_plot=True, W=weights, b=intercept,
                              title='Données de Test', offset_xy=[0.5, 0.5])
# =======================================================================================================================
