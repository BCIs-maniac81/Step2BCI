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
import postproc.classification as classifier

"""Il s'agit d'un enregistrement de grande envergure : 59 électrodes ont été utilisées, réparties sur l'ensemble du 
cuir chevelu. Le sujet a reçu un signal et a ensuite imaginé soit le mouvement de la main droite, soit le mouvement 
de ses pieds. Comme le montre l'homoncule, le mouvement du pied est contrôlé au centre du cortex moteur (ce qui rend 
difficile la distinction entre le pied gauche et le pied droit), tandis que le mouvement de la main est contrôlé plus 
latéralement. 
"""

X, trials, targets, attributes = iof.read_matlab_file('../datasets/berlindata/comp4_calib_ds1d/BCICIV_calib_ds1d.mat',
                                                      verbosity=False, dtype='EEG')

nsamples, nchannels = X.shape
sfreq = attributes['sfreq']
classes = attributes['Classes']
channels = attributes['Channels']

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
Les essais sont coupés dans l'intervalle [0,5 - 2,5 s] après l'apparition du marqueur.
'''
print(sfreq)
trials_data = trproc.trials_extract(inputData=X, trials=trials, targets=targets, window=[0.5, 2.5], version='berlin',
                                    sfreq=sfreq)
print('format Berlin trials data (n_trials, data, channel)', trials_data[-1].shape)
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
freqs, A_f = analysis.welch_periodogram_analysis(inputData=trials_data[-1], sliding_win=200, sfreq=sfreq, t_ax=1)
freqs, A_r = analysis.welch_periodogram_analysis(inputData=trials_data[1], sliding_win=200, sfreq=sfreq, t_ax=1)

trials_PSD = {'left': A_f, 'right': A_r}
colors = ['green', 'red']
'''
Utilisons la fonction trials_psd_plot() pour tracer trois canaux :
C3 : central, gauche
Cz : Central, central
C4 : central, droit
'''
plotter.trials_psd_plot(trials_PSD, freqs,
                        [channels.index(ch) for ch in ['C3', 'Cz', 'C4']],
                        chan_lab=['C3', 'Cz', 'C4'], maxy=500, xlabel='Fréquence ($Hz$)', ylabel='PSD', colors=colors)
'''
Un pic d'activité mu peut être observé sur chaque canal pour les deux classes. Dans l'hémisphère droit,
l'activité mu pour le mouvement de la main gauche est plus faible que pour le mouvement de la main droite
en raison de l'ERD. Au niveau de l'électrode gauche, l'activité mu pour le mouvement de la main droite
est réduite et au niveau de l'électrode centrale, l'activité mu est à peu près égale pour les deux classes.
Ceci est conforme à la théorie selon laquelle la main gauche est contrôlée par l'hémisphère droit et les pieds 
sont contrôlés au niveau central.
'''
'''
Classification des données
Nous utiliserons un algorithme d'apprentissage automatique pour construire un modèle capable de distinguer
les mouvements de la main droite et du pied de ce sujet. Pour ce faire, nous devons trouver un moyen de quantifier la 
quantité d'activité mu présente dans un essai créer un modèle qui décrit les valeurs attendues de l'activité mu pour chaque
classe enfin, tester ce modèle sur des données inédites pour voir s'il est capable de prédire l'étiquette de la classe 
correcte. Nous suivrons la conception classique de l'ICB de Blankertz et al [1], qui utilisent le logarithme de la variance
du signal dans une certaine bande de fréquence comme caractéristique pour le classificateur. 

[1] Blankertz, B., Dornhege, G., Krauledat, M., Müller, K.-R., & Curio, G. (2007). L'interface cerveau-ordinateur
non invasive de Berlin : acquisition rapide de performances efficaces chez des sujets non entraînés. NeuroImage,
37(2), 539-550. doi:10.1016/j.neuroimage.2007.01.051

Le script ci-dessous conçoit un filtre passe-bande à l'aide de scipy.signal.irrfilter qui élimine les fréquences
 en dehors de la fenêtre 8--15Hz. Le filtre est appliqué à tous les essais :
'''

trials_filt_f = filt.band_pass(trials_data[-1], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)
trials_filt_r = filt.band_pass(trials_data[1], lowfreq=8, highfreq=15, order=6, sfreq=100, t_ax=1)

freqs, psd_f = analysis.welch_periodogram_analysis(inputData=trials_filt_f, sliding_win=200, sfreq=100, t_ax=1)
freqs, psd_r = analysis.welch_periodogram_analysis(inputData=trials_filt_r, sliding_win=200, sfreq=100, t_ax=1)

trials_filt_psd = {'left': psd_f, 'right': psd_r}
colors = ['green', 'red']

plotter.trials_psd_plot(trials_filt_psd, freqs,
                        [channels.index(ch) for ch in ['C3', 'Cz', 'C4']],
                        chan_lab=['C3', 'Cz', 'C4'], maxy=500, xlabel='Fréquence ($Hz$)', ylabel='PSD', colors=colors)
'''
Comme caractéristique pour le classifieur, nous utiliserons le logarithme de la variance de chaque canal.
'''

logvar_f = featext.logvar_feat(trials_filt_f, t_ax=1)
logvar_r = featext.logvar_feat(trials_filt_r, t_ax=1)

trials_logvar = {'left': logvar_f, 'right': logvar_r}
params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var', xlabel='Canaux'
                                                                                                   '/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['g', 'purple'], **params)

"""Nous constatons que la plupart des canaux présentent une faible différence dans le log-var du signal entre les 
deux classes. L'étape suivante consiste à passer de 118 canaux à seulement quelques mélanges de canaux. L'algorithme 
CSP calcule des mélanges de canaux conçus pour maximiser la différence de variation entre deux classes. Ces mélanges 
sont appelés filtres spatiaux. """

trials_filt = {'left': trials_filt_f, 'right': trials_filt_r}
trials_csp, W = featext.csp_feat(trials_filt, fitted=False, t_ax=1)
'''Pour voir le résultat de l'algorithme CSP, nous traçons le log-var comme nous l'avons fait précédemment.'''

logvar_f = featext.logvar_feat(trials_csp['left'], t_ax=1)
logvar_r = featext.logvar_feat(trials_csp['right'], t_ax=1)

trials_logvar = {'left': logvar_f, 'right': logvar_r}
params = dict(title='caractéristique Log-var de chaque canal/composante', ylabel='Log-var', xlabel='Canaux'
                                                                                                   '/composantes (#)')
featplt.plot_logvar(trials_logvar, trial_axis=0, colors=['g', 'purple'], **params)

"""Au lieu de 118 canaux, nous avons maintenant 118 mélanges de canaux, appelés composantes. Elles sont le résultat 
de 118 filtres spatiaux appliqués aux données. 

Les premiers filtres maximisent la variation de la première classe, tout en minimisant la variation de la seconde. 
Les derniers filtres maximisent la variation de la deuxième classe, tout en minimisant la variation de la première. 

Ceci est également visible dans un graphique PSD. Le code ci-dessous trace la DSP pour les premières et dernières 
composantes, ainsi que pour une composante centrale : """

freqs, welch_f = analysis.welch_periodogram_analysis(inputData=trials_csp['left'], sliding_win=200, sfreq=sfreq, t_ax=1)
freqs, welch_r = analysis.welch_periodogram_analysis(inputData=trials_csp['right'], sliding_win=200, sfreq=sfreq,
                                                     t_ax=1)

trials_PSD = {'left': welch_f, 'right': welch_r}

plotter.trials_psd_plot(trials_PSD, freqs, chan_ind=[0, 28, -1],
                        chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'], maxy=0.75,
                        xlabel='Fréquence (''$Hz$)',
                        ylabel='PSD', colors=['green', 'red'])

'''Pour voir dans quelle mesure nous pouvons différencier les deux classes, un diagramme de dispersion est un outil 
utile. Ici, les deux classes sont représentées sur un plan à deux dimensions : l'axe des x correspond à la première 
composante du CSP, l'axe des y à la dernière. '''

featplt.features_scatter_plot(logvar_f, logvar_r)

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

# Former les ensembles de formation et de test à partir des données trials_filt

features = np.concatenate((trials_filt['left'], trials_filt['right']), axis=0)
print(features.shape)

targets = np.concatenate((np.full((trials_filt['left'].shape[0],), -1),
                          np.full((trials_filt['right'].shape[0],), 1)),
                         axis=0)
print(targets.shape)

split_features_targets = tools.train_test_clf(features=features, targets=targets,
                                              test_size=0.25, random_state=0)

X_train, X_true, y_train, y_true = split_features_targets
print(X_train.shape)
print(X_true.shape)
print(y_train.shape)
print(y_true.shape)

X_train_f = X_train[y_train == -1, :, :]
X_train_r = X_train[y_train == 1, :, :]
X_train = {'left': X_train_f, 'right': X_train_r}
X_true_f = X_true[y_true == -1, :, :]
X_true_r = X_true[y_true == 1, :, :]
X_true = {'left': X_true_f, 'right': X_true_r}

# Formation CSP seulement sur l'ensemble de formation X_train (left & right)
trials_csp_train, W = featext.csp_feat(X_train, fitted=False, t_ax=1)

# freqs_tr, psd_tr_f = analysis.welch_periodogram_analysis(inputData=trials_csp_train['left'], sliding_win=200, sfreq=100, t_ax=1)
# freqs_tr, psd_tr_r = analysis.welch_periodogram_analysis(inputData=trials_csp_train['right'], sliding_win=200, sfreq=100, t_ax=1)
#
# trials_tr_PSD = {'left': psd_tr_f, 'right': psd_tr_r}
#
# colors = ['green', 'red']
#
# plotter.trials_psd_plot(trials_tr_PSD, freqs_tr, chan_ind=[0, 28, -1],
#                         chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'], maxy=0.75, xlabel='Fréquence ('
#                                                                                                        '$Hz$)',
#                         ylabel='PSD', colors=colors)

# Appliquer CSP formé à l'ensemble de test
trials_csp_true, W_ = featext.csp_feat(X_true, fitted=True, W=W, t_ax=1)

# freqs_true, psd_true_f = analysis.welch_periodogram_analysis(inputData=trials_csp_true['left'], sliding_win=200, sfreq=100, t_ax=1)
# freqs_true, psd_true_r = analysis.welch_periodogram_analysis(inputData=trials_csp_true['right'], sliding_win=200, sfreq=100, t_ax=1)
#
# trials_true_PSD = {'left': psd_true_f, 'right': psd_true_r}
#
# colors = ['green', 'red']
#
# plotter.trials_psd_plot(trials_true_PSD, freqs_tr, chan_ind=[0, 28, -1],
#                         chan_lab=['1ère Comp.', 'Comp. centrale', 'Dernière Comp.'], maxy=0.75, xlabel='Fréquence ('
#                                                                                                        '$Hz$)',
#                         ylabel='PSD', colors=colors)

# Calculer Log-var des deux ensembles
train_f = trials_csp_train['left'][:, :, [0, -1]]
logvar_train_f = featext.logvar_feat(train_f, t_ax=1)
train_r = trials_csp_train['right'][:, :, [0, -1]]
logvar_train_r = featext.logvar_feat(train_r, t_ax=1)
featplt.features_scatter_plot(logvar_train_f, logvar_train_r)
train = np.concatenate((logvar_train_f, logvar_train_r), axis=0)
y_train = np.concatenate((np.full((logvar_train_f.shape[0],), -1), np.full((logvar_train_r.shape[0],), 1)), axis=0)

true_f = trials_csp_true['left'][:, :, [0, -1]]
logvar_true_f = featext.logvar_feat(true_f, t_ax=1)
true_r = trials_csp_true['right'][:, :, [0, -1]]
logvar_true_r = featext.logvar_feat(true_r, t_ax=1)
featplt.features_scatter_plot(logvar_true_f, logvar_true_r)
true = np.concatenate((logvar_true_f, logvar_true_r), axis=0)
y_true = np.concatenate((np.full((logvar_true_f.shape[0],), -1), np.full((logvar_true_r.shape[0],), 1)), axis=0)

features_ = np.concatenate((train, true), axis=0)
targets_ = np.concatenate((y_train, y_true), axis=0)

# Sélection des estimateurs
# splitted_data = train, true, y_train, y_true

# ******************************* Classifieur - Analyse discriminate linéaire (LDA) ************************************

print('********************  Classification: Analyse discriminante linéaire ******************************************')
y_pred_lda, lda = classifier.lda_classifier(X_train=train, y_train=y_train,
                                            X_true=true, shrinkage=None, solver='svd')
reports_lda = tools.report_clf(estimator=lda, y_true=y_true, y_pred=y_pred_lda,
                               conf_matrix_plot=True)
accuracy_lda = tools.accuracy_clf(y_true=y_true, y_pred=y_pred_lda)
print("Type du classifieur:", lda)
print("Rapport de la classification: ", reports_lda)
print("Précision de la classification: %.2f" % accuracy_lda)

# ***************************** Classifieur - La régression logistique (LogReg) ****************************************
print('********************************  Classification: Logistic regression *****************************************')
y_pred_logreg, logreg = classifier.logreg_classifier(X_train=train, y_train=y_train,
                                                     X_true=true, C=5)
reports_logreg = tools.report_clf(estimator=logreg, y_true=y_true, y_pred=y_pred_logreg,
                                  conf_matrix_plot=True)
accuracy_logreg = tools.accuracy_clf(y_true=y_true, y_pred=y_pred_logreg)
print("Type du classifieur:", logreg)
print("Rapport de la classification: ", reports_logreg)
print("Précision de la classification: %.2f" % accuracy_logreg)

# ***************************** Classifieur - Machine à Vecteurs de Support (SVM) ************************************
print('********************  Classification: Support Vector Machine SVM **********************************************')
params = {'C': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
y_pred_svm, svm = classifier.svm_classifier(X_train=train, y_train=y_train, X_true=true, fastmode=True,
                                            gridsearch=False, cv=10, max_iter=50, **params)
reports_svm = tools.report_clf(estimator=svm, y_true=y_true, y_pred=y_pred_svm,
                               conf_matrix_plot=True)
accuracy_svm = tools.accuracy_clf(y_true=y_true, y_pred=y_pred_svm)
print("Type du classifieur:", svm)
print("Rapport de la classification: ", reports_svm)
print("Précision de la classification: %.2f" % accuracy_svm)

# ***************************** Classifieur - K voisins les plus proches KNN ****************************************
print('********************  Classification: K-Nearest Neighbours *****************************************')

y_pred_knn, knn = classifier.knn_classifier(X_train=train, y_train=y_train,
                                            X_true=true, n_neighbors=17)
reports_knn = tools.report_clf(estimator=knn, y_true=y_true, y_pred=y_pred_knn,
                               conf_matrix_plot=True)
accuracy_knn = tools.accuracy_clf(y_true=y_true, y_pred=y_pred_knn)
print("Type du classifieur:", knn)
print("Rapport de la classification: ", reports_knn)
print("Précision de la classification: %.2f" % accuracy_knn)
