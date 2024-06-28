# -*- coding: utf-8 -*-
"""
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
Il s'agit d'un enregistrement important : 59 électrodes Puisque la caractéristique que nous recherchons 
(une diminution de l'activité 𝜇) est une caractéristique de fréquence, traçons la DSP des essais de la même manière 
qu'avec les données SSVEP. Le code ci-dessous définit une fonction qui calcule la DSP pour chaque essai 
(nous en aurons de nouveau besoin plus tard) : ici, elle est répartie sur l'ensemble du cuir chevelu. 
Le sujet a reçu un indice et a ensuite imaginé soit le mouvement de sa main droite, soit le mouvement de ses pieds. 
Comme le montre l'homoncule, le mouvement du pied est contrôlé au centre du cortex moteur 
(ce qui rend difficile la distinction entre le pied gauche et le pied droit), 
tandis que le mouvement de la main est contrôlé plus latéralement.

'''
# freqs_f, mt_psd_f = analysis.multitaper_period_analysis(inputData=trials_data[-1], dB=False, adaptive=True, sfreq=sfreq)
# freqs_r, mt_psd_r = analysis.multitaper_period_analysis(inputData=trials_data[1], dB=False, adaptive=True, sfreq=sfreq)

freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=trials_data[target_codes[0]], sliding_win=200, sfreq=sfreq, t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=trials_data[target_codes[1]], sliding_win=200, sfreq=sfreq, t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

'''
Utilisons la fonction trials_psd_plot() pour tracer trois canaux :
C3 : central, gauche
Cz : Central, central
C4 : central, droit
'''
plotter.trials_psd_plot(trials_psd, freqs_f, [channels.index(ch) for ch in ['C3', 'Cz', 'C4']], \
                        chan_lab=['C3', 'Cz', 'C4'], maxy=500, xlabel='Fréquence ($Hz$)', \
                        ylabel='PSD de Welch', colors=['green', 'red'])
# plt.savefig('trials_psd.jpeg', dpi=300)

trials_filt_f = filt.band_pass(trials_data[target_codes[0]], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)
trials_filt_r = filt.band_pass(trials_data[target_codes[1]], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)

trials_filt = trproc.data_class_composer(trials_filt_f, trials_filt_r, classes=classes)

freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=trials_filt_f, sliding_win=200, sfreq=sfreq, t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=trials_filt_r, sliding_win=200, sfreq=sfreq, t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

plotter.trials_psd_plot(trials_psd, freqs_f, [channels.index(ch) for ch in ['C3', 'Cz', 'C4']], \
                        chan_lab=['C3', 'Cz', 'C4'], maxy=500, xlabel='Fréquence ($Hz$)', \
                        ylabel='PSD de Welch', colors=['green', 'red'])

# plt.savefig('trials_psd2.jpeg', dpi=300)

logvar_f = featext.logvar_feat(trials_filt_f, t_ax=1)
logvar_r = featext.logvar_feat(trials_filt_r, t_ax=1)

trials_logvar = trproc.data_class_composer(logvar_f, logvar_r, classes=classes)
params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var',
              xlabel='Canaux/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['green', 'red'], **params)

# plt.savefig('logvar1.jpeg', dpi=300)

"""
Nous constatons que la plupart des canaux présentent une faible différence dans le log-var du signal entre les
deux classes. L'étape suivante consiste à passer de 118 canaux à seulement quelques mélanges de canaux. L'algorithme
CSP calcule des mélanges de canaux conçus pour maximiser la différence de variation entre deux classes. Ces mélanges
sont appelés filtres spatiaux.
"""

trials_csp, W = featext.csp_feat(trials_filt, fitted=False, t_ax=1)

'''Pour voir le résultat de l'algorithme CSP, nous traçons le log-var comme nous l'avons fait précédemment.'''
logvar_f = featext.logvar_feat(trials_csp[classes[0]], t_ax=1)
logvar_r = featext.logvar_feat(trials_csp[classes[1]], t_ax=1)

trials_logvar = trproc.data_class_composer(logvar_f, logvar_r, classes=classes)

params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var',
              xlabel='Canaux/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['green', 'red'], **params)

# plt.savefig('logvar2.jpeg', dpi=300)

"""Au lieu de 118 canaux, nous avons maintenant 118 mélanges de canaux, appelés composantes. Elles sont le résultat
de 118 filtres spatiaux appliqués aux données.

Les premiers filtres maximisent la variation de la première classe, tout en minimisant la variation de la seconde.
Les derniers filtres maximisent la variation de la deuxième classe, tout en minimisant la variation de la première.

Ceci est également visible dans un graphique PSD. Le code ci-dessous trace la DSP pour les premières et dernières
composantes, ainsi que pour une composante centrale : """

freqs_f, welch_f = analysis.welch_periodogram_analysis(inputData=trials_csp[classes[0]], sliding_win=200, sfreq=sfreq,
                                                       t_ax=1)
freqs_r, welch_r = analysis.welch_periodogram_analysis(inputData=trials_csp[classes[1]], sliding_win=200, sfreq=sfreq,
                                                       t_ax=1)

trials_psd = trproc.data_class_composer(welch_f, welch_r, classes=classes)

plotter.trials_psd_plot(trials_psd, freqs_f, chan_ind=[0, 29, -1],
                        chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'],
                        maxy=0.75, xlabel='Fréquence (''$Hz$)', ylabel='PSD de Welch', colors=['green', 'red'])
# plt.savefig('csp_welch1.jpeg', dpi=300)

'''Pour voir dans quelle mesure nous pouvons différencier les deux classes, un diagramme de dispersion est un outil
utile. Ici, les deux classes sont représentées sur un plan à deux dimensions : l'axe des x correspond à la première
composante du CSP, l'axe des y à la dernière. '''

featplt.features_scatter_plot(logvar_f, logvar_r)
# plt.savefig('scatter_csp1.jpeg', dpi=300)

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
# Former les ensembles de formation et de test à partir des données trials_filt

data = np.concatenate((trials_filt[classes[0]], trials_filt[classes[1]]), axis=0)
print(data.shape)

targets = np.concatenate((np.full((trials_filt[classes[0]].shape[0],), target_codes[0]),
                          np.full((trials_filt[classes[1]].shape[0],), target_codes[1])),
                         axis=0)
print(targets.shape)


splitted_data = tools.train_test_clf(features=data, targets=targets, test_size=0.25, random_state=123456789)

'''
La fonction train_true_composer est utilisée pour préparer les données en vue de l'extraction des caractéristiques à l'aide
de la fonction csp_feat(). Elle effectue une transformation des données pour les mettre dans un format approprié pour être
traitées par csp_feat(), en les organisant dans des dictionnaires distincts pour les classes 'left' et 'right' en fonction
des codes cibles fournis.
'''
train, true = trproc.train_true_composer(splitted_data, classes=classes, target_codes=target_codes)

# Formation CSP seulement sur l'ensemble de formation X_train (left & right)
trials_csp_train, W = featext.csp_feat(train, fitted=False, t_ax=1)

# Appliquer CSP formé à l'ensemble de test
trials_csp_true, W_ = featext.csp_feat(true, fitted=True, W=W, t_ax=1)

# data = np.concatenate((train, true), axis=0)
# targets = np.concatenate((y_train, y_true), axis=0)
#
# Sélection des estimateurs
splitted_data = featext.logvar_csp_composer(trials_csp_train, trials_csp_true,
                                            classes=classes, target_codes=target_codes)
scores, y_pred = tools.estimator_select(None, None, cv=4, ts=None, rs=None, splitted_data=splitted_data,
                                        with_predict=True)
for item in scores:
    print(item)
