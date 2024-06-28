# -*- coding: utf-8 -*-
"""
@author: hafed-eddine BENDIB
"""
import numpy as np
import matplotlib.pyplot as plt
import preproc.trialsProcessor as trproc
import preproc.dataAnalyzer as analysis
import postproc.featuresPlotter as featplt
import preproc.IOfiles as iof
import preproc.dataFilter as filt
import preproc.dataPlotter as plotter
import postproc.featuresExtractor as featext
import postproc.classifierTools as tools

import torch.optim
from deepproc.deepBCI import NetPytorch, TorchTrainer, TorchEvaluator, TorchPredictor
from deepproc.neuroDataset import CustomDataset
from deepproc.neuroDevice import DeviceSwitcher

# Load data
X, trials, targets, attributes = iof.read_matlab_file(
    full_path='../datasets/berlindata/comp4_calib_ds1d/BCICIV_calib_ds1d.mat',
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

'''
Le code fourni concatène les données d'essais de deux classes en un seul ensemble de données,
tout en créant les étiquettes cibles correspondantes.
'''

# Concatenate trials data
data = np.concatenate((trials_data[target_codes[0]], trials_data[target_codes[1]]), axis=0)
# Create corresponding target labels
targets = np.concatenate((np.full((trials_data[target_codes[0]].shape[0],), target_codes[0]), \
                          np.full((trials_data[target_codes[1]].shape[0],), target_codes[1])), axis=0)

'''
En introduisant la fonction train_test_clf du module tools, cette ligne de code divise les caractéristiques et les étiquettes
en ensembles d'apprentissage et de test. Les paramètres features et targets représentent respectivement les données d'entrée
et les étiquettes. En spécifiant un test_size de 0,25 (25% des données) et un random_state de 0 pour la reproductibilité,
la variable split_features_targets contient les ensembles divisés prêts à être utilisés dans les étapes suivantes de
l'analyse ou de la classification.
'''
splitted_data = tools.train_test_clf(features=data, targets=targets,
                                     test_size=0.25, random_state=123456789)
'''
# La fonction train_true_composer est utilisée pour préparer les données en vue de l'extraction des caractéristiques à l'aide
# de la fonction csp_feat(). Elle effectue une transformation des données pour les mettre dans un format approprié pour être
# traitées par csp_feat(), en les organisant dans des dictionnaires distincts pour les classes 'left' et 'right' en fonction
# des codes cibles fournis.
# '''
train, true = trproc.train_true_composer(splitted_data, classes=classes, target_codes=target_codes)

# Périodogramme de Welch pour estimer la PSD
freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=train[classes[0]], sliding_win=200, sfreq=sfreq,
                                                       t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=train[classes[1]], sliding_win=200, sfreq=sfreq,
                                                       t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

plotter.trials_psd_plot(trials_psd, freqs_f, [channels.index(ch) for ch in ['C3', 'Cz', 'C4']], \
                        chan_lab=['C3', 'Cz', 'C4'], maxy=800, xlabel='Fréquence ($Hz$)', \
                        ylabel='PSD de Welch', colors=['green', 'red'])
# plt.savefig('trials_psd.jpeg', dpi=400)

filtered_train_f = filt.band_pass(train[classes[0]], lowfreq=9, highfreq=15, order=8, sfreq=100, t_ax=1)
filtered_train_r = filt.band_pass(train[classes[1]], lowfreq=9, highfreq=15, order=8, sfreq=100, t_ax=1)
train_filt = trproc.data_class_composer(filtered_train_f, filtered_train_r, classes=classes)

filtered_true_f = filt.band_pass(true[classes[0]], lowfreq=9, highfreq=15, order=8, sfreq=100, t_ax=1)
filtered_true_r = filt.band_pass(true[classes[1]], lowfreq=9, highfreq=15, order=8, sfreq=100, t_ax=1)
true_filt = trproc.data_class_composer(filtered_true_f, filtered_true_r, classes=classes)

freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=filtered_train_f, sliding_win=200, sfreq=sfreq, t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=filtered_train_r, sliding_win=200, sfreq=sfreq, t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

plotter.trials_psd_plot(trials_psd, freqs_f, [channels.index(ch) for ch in ['C3', 'Cz', 'C4']], \
                        chan_lab=['C3', 'Cz', 'C4'], maxy=400, xlabel='Fréquence ($Hz$)', \
                        ylabel='PSD de Welch', colors=['green', 'red'])
# plt.savefig('trials_psd2.jpeg', dpi=400)

logvar_f = featext.logvar_feat(filtered_train_f, t_ax=1)
logvar_r = featext.logvar_feat(filtered_train_r, t_ax=1)

trials_logvar = trproc.data_class_composer(logvar_f, logvar_r, classes=classes)

params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var',
              xlabel='Canaux/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['green', 'red'], **params)
# plt.savefig('logvar1.jpeg', dpi=400)

# Formation CSP seulement sur l'ensemble d'apprentissage train
trials_csp_train, W = featext.csp_feat(train_filt, fitted=False, t_ax=1)

'''Pour voir le résultat de l'algorithme CSP, nous traçons le log-var comme nous l'avons fait précédemment.'''
train_logvar_f = featext.logvar_feat(trials_csp_train[classes[0]], t_ax=1)
train_logvar_r = featext.logvar_feat(trials_csp_train[classes[1]], t_ax=1)

train_logvar = trproc.data_class_composer(train_logvar_f, train_logvar_r, classes=classes)

params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var',
              xlabel='Canaux/composantes (#)')
featplt.plot_logvar(train_logvar, trial_axis=0, colors=['green', 'red'], **params)
# plt.savefig('logvar2.jpeg', dpi=400)

freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=trials_csp_train[classes[0]],
                                                       sliding_win=200, sfreq=sfreq, t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=trials_csp_train[classes[1]],
                                                       sliding_win=200, sfreq=sfreq, t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

plotter.trials_psd_plot(trials_psd, freqs_f, chan_ind=[0, 29, -1],
                        chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'],
                        maxy=0.75, xlabel='Fréquence (''$Hz$)', ylabel='PSD de Welch', colors=['green', 'red'])
# plt.savefig('csp_welch1.jpeg', dpi=400)

'''Pour voir dans quelle mesure nous pouvons différencier les deux classes, un diagramme de dispersion est un outil
utile. Ici, les deux classes sont représentées sur un plan à deux dimensions : l'axe des x correspond à la première
composante du CSP, l'axe des y à la dernière. '''

featplt.features_scatter_plot(train_logvar_f, train_logvar_r)
plt.savefig('scatter_csp1.jpeg', dpi=400)

# =========================================== Classification ==================================================
'''Nous allons appliquer un classifieur linéaire à ces données. On peut considérer qu'un classifieur linéaire
dessine une ligne dans le graphique ci-dessus pour séparer les deux classes. Pour déterminer la classe d'un nouvel
essai, il suffit de vérifier de quel côté de la ligne l'essai se situerait s'il était représenté comme ci-dessus.
Les données sont divisées en un ensemble de formation et un ensemble de test. Le classifieur ajuste un modèle (
dans ce cas, une ligne droite) sur l'ensemble d'apprentissage et utilise ce modèle pour faire des prédictions sur
l'ensemble de test (voir de quel côté de la ligne se situe chaque essai de l'ensemble de test). Notez que
l'algorithme CSP fait partie du modèle et que, par souci d'équité, il doit donc être calculé en utilisant uniquement
les données d'apprentissage.'''
# ==================================

# Appliquer CSP formé à l'ensemble de test
trials_csp_true, _ = featext.csp_feat(true_filt, fitted=True, W=W, t_ax=1)

# établir les données et les targets d'apprentissage et de test pour alimenter la fonction tools.estimator_select()
composed_data = featext.logvar_csp_composer(trials_csp_train, trials_csp_true, classes, target_codes, plot=True)

# Decomposition des données
train, true, train_targets, true_targets = composed_data

# Re-encodage des étiquettes cibles -1 --> 0 & 1 --> 1
train_targets = np.where(train_targets == -1, 0, train_targets)
true_targets = np.where(true_targets == -1, 0, true_targets)

dataset_train = CustomDataset((train, train_targets), None, None)
dataset_true = CustomDataset((true, true_targets), None, None)

# ==================================== Classification par Apprentissage approfondi =====================================
# Hyper-paramètres
n_features = train.shape[1]
nclasses = len(np.unique(true_targets))
hidden_sizes = [8]
batch_size1 = len(train_targets) // 8  # dataset_train.data.shape[0]
batch_size2 = len(true_targets) // 8  # dataset_test.data.shape[0]
learning_rate = 0.19
dropout_prob = .2
n_epochs = 1000
momentum = .94
weight_decay = 0.1

print(f'Le nombre de classes: {nclasses}')
print(f'Le nombre de caractéristiques: {n_features}')
print(f'la taille des couches cachées: {hidden_sizes}')
print(f'La taille de chaque batch (train): {batch_size1}')
print(f'La taille de chaque batch (test): {batch_size2}')

# =====================================================================================================================
# =====================================================================================================================
# Set the device to use for training and inference
device = DeviceSwitcher('cpu').device
print(device)
# Define the model architecture
model = NetPytorch(n_features, nclasses, *hidden_sizes, dropout_prob=.5)
# # Define the trainer and train the model
trainer = TorchTrainer(model=model, train_data=dataset_train, batch_size=batch_size1, max_epochs=n_epochs,
                       learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, device=device,
                       optimizer=torch.optim.Adam)

trained_model, train_accuracy, epoch_losses, eval_losses = trainer.train(eval_train=True, patience=15, save_model=False)

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
# plt.suptitle('Évaluation des performances au cours de la phase d\'apprentissage')
ax.plot(epoch_losses, label='Perte d\'entraînement', color='blue')
ax.plot(eval_losses, label='Perte d\'évaluation', color='orange')

y_max = 0.8
ax.vlines(trainer.early_stopping_epoch, 0, 0.6, linestyle='dashed', color='red')
ax.text(trainer.early_stopping_epoch, 0.6 , 'Arrêt précoce (Early Stopping)', ha='center', va='bottom',
        rotation=0, fontsize=10, color='#1076BB')
step = 5
ax.set_xticks(list(range(0, trainer.early_stopping_epoch - 3, step)) + [trainer.early_stopping_epoch])

ax.set_xlabel("époques", fontsize=14)
ax.set_ylabel('Perte', fontsize=14)
plt.grid(alpha=0.4)
plt.legend(loc='lower left')
plt.show()

plt.savefig('torch_model.jpeg', dpi=400)