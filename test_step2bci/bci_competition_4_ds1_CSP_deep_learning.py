# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:45:28 2021

@author: Hafed-eddine BENDIB
"""

import numpy as np
import preproc.IOfiles as iof
import preproc.trialsProcessor as trproc
import preproc.dataPlotter as plotter
import preproc.dataAnalyzer as analysis
import preproc.dataFilter as filt
import postproc.featuresExtractor as featext
import postproc.featuresPlotter as featplt
import matplotlib.pyplot as plt

import torch.optim
from deepproc.deepBCI import NetPytorch, TorchTrainer, TorchEvaluator, TorchPredictor
from deepproc.neuroDataset import CustomDataset, LoadSplitDataset
from deepproc.neuroDevice import DeviceSwitcher



# Load data
X, trials, targets, attributes = iof.read_matlab_file('datasets/berlindata/comp4_calib_ds1d/BCICIV_calib_ds1d.mat',
                                                      verbosity=False, dtype='EEG')

nsamples, nchannels = X.shape
sfreq = attributes['sfreq']
classes = attributes['Classes']
channels = attributes['Channels']
target_codes = np.unique(targets)
"""
targets: -1: Imagination de la main gauche (Left) , 1: Imagination de la main droite (Right)
"""
nclasses = len(classes)
cl1 = classes[0]
cl2 = classes[1]

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
print('format des données pour la classe -1 (n_trials, data, channel)', trials_data[target_codes[0]].shape)
print('format des données pour la classe 1 (n_trials, data, channel)', trials_data[target_codes[1]].shape)

trials_filt_f = filt.band_pass(trials_data[target_codes[0]], lowfreq=9, highfreq=15, order=6, sfreq=100, t_ax=1)
trials_filt_r = filt.band_pass(trials_data[target_codes[1]], lowfreq=9, highfreq=15, order=6, sfreq=100, t_ax=1)

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

featplt.features_scatter_plot(logvar_f, logvar_r, title='Ensemble des données')

# =========================================== Classification ==================================================
X_train, X_true = trproc.split_data_classes(trials_filt, classes=classes, target_codes=target_codes,
                                            ts=0.25, rs=1)

# Formation CSP seulement sur l'ensemble de formation X_train (left & right)
trials_csp_train, W = featext.csp_feat(X_train, fitted=False, t_ax=1)

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

X_train, y_train = trproc.split_data_classes(logvar_train, for_classification=True)
X_true, y_true = trproc.split_data_classes(logvar_true, for_classification=True)
y_train[y_train == -1] = 0
y_true[y_true == -1] = 0
dataset_train = CustomDataset((X_train, y_train), None, None)
dataset_true = CustomDataset((X_true, y_true), None, None)

print(dataset_train)


# ==================================== Classification par Apprentissage approfondi =====================================
# Hyper-paramètres
n_features = X_train.shape[1]
hidden_sizes = [32, 32, 32]
batch_size1 = 16  # dataset_train.data.shape[0]
batch_size2 = 16  # dataset_test.data.shape[0]
learning_rate = 0.1
dropout_prob = .2
n_epochs = 1000
momentum = .94
w_d = 0.0

print(f'Le nombre de classes: {nclasses}')
print(f'Le nombre de caractéristiques: {n_features}')
print(f'la taille des couches cachées: {hidden_sizes}')
print(f'La taille de chaque batch (train): {batch_size1}')
print(f'La taille de chaque batch (test): {batch_size2}')

# =====================================================================================================================
# Set the device to use for training and inference
device = DeviceSwitcher('cpu').device
print(device)
# Define the model architecture
model = NetPytorch(n_features, nclasses, *hidden_sizes, dropout_prob=.5)
# # Define the trainer and train the model
trainer = TorchTrainer(model=model, train_data=dataset_train, batch_size=batch_size1, max_epochs=n_epochs,
                       learning_rate=learning_rate, momentum=momentum, weight_decay=w_d, device=device,
                       optimizer=torch.optim.Adam)

trained_model, train_accuracy, epoch_losses, eval_losses = trainer.train(eval_train=True, patience=10, save_model=True)

# Define the evaluator and evaluate the model on the validation set

evaluator = TorchEvaluator(model=trained_model, criterion=torch.nn.CrossEntropyLoss, batch_size=batch_size2,
                           device=device)
test_accuracy, test_loss = evaluator.evaluate(val_data=dataset_true, eval_train=False)

# Define the predictor and make predictions on the test set
predictor = TorchPredictor(model=model, input_size=n_features, device=device)
predictions = predictor.predict_batch(dataset_true.data[20:50])


print('=================')
print(predictions)
print(dataset_true.targets[20:50])
print('==============')

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
plt.suptitle('Performance evaluation during Training step')
ax.plot(epoch_losses, label='Training loss', color='blue')
ax.plot(eval_losses, label='Evaluation loss', color='orange')

y_max = 1
ax.vlines(trainer.early_stopping_epoch, 0, y_max, linestyle='dashed', color='red')
ax.text(trainer.early_stopping_epoch, y_max + 0.1, 'early stopping', ha='center', va='bottom',
        rotation=90, fontsize=10, color='#1076BB')
step = 5
ax.set_xticks(list(range(0, trainer.early_stopping_epoch - 3, step)) + [trainer.early_stopping_epoch])

ax.set_xlabel("Epochs", fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
plt.grid(alpha=0.4)
plt.legend()
plt.show()
