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
# plt.savefig('scatter_csp1.jpeg', dpi=400)

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

# data = np.concatenate((train, true), axis=0)
# targets = np.concatenate((y_train, y_true), axis=0)

# Sélection des estimateurs
scores, y_pred = tools.estimator_select(features=None, targets=None, cv=6, ts=None, rs=None,
                                        splitted_data=composed_data, with_predict=False)
for item in scores:
    print(item)

train, true, train_targets, true_targets = composed_data

print('******************** lassifieur (KNN) - K plus proches voisins *****************************************')
params = {'n_neighbors': 8}
y_pred_knn, knn = clf.knn_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_knn = tools.report_clf(estimator=knn, y_true=true_targets, y_pred=y_pred_knn, conf_matrix_plot=True)
plt.savefig('knn_conf_matrix.jpeg', dpi=400)
accuracy_knn = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_knn)
# print("Type du classifieur:", knn)
# print("Rapport de la classification: ", reports_knn)
print("Précision de la classification: %.2f" % accuracy_knn)

print('*******************************  Classifieur (LogReg): Régression Logistique  ******************************')
params = {'C':0.1}
y_pred_logreg, logreg = clf.logreg_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_logreg = tools.report_clf(estimator=logreg, y_true=true_targets, y_pred=y_pred_logreg, conf_matrix_plot=True)
plt.savefig('logreg_conf_matrix.jpeg', dpi=400)
accuracy_logreg = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_logreg)
# print("Type du classifieur:", logreg)
# print("Rapport de la classification: ", reports_logreg)
print("Précision de la classification: %.2f" % accuracy_logreg)

print('********************************  Classifieur (GNB) - Naive Bayes Gaussien ***********************************')
params = {'var_smoothing': 1.0}
y_pred_gnb, gnb = clf.gnb_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_gnb = tools.report_clf(estimator=gnb, y_true=true_targets, y_pred=y_pred_gnb, conf_matrix_plot=True)
plt.savefig('gnb_conf_matrix.jpeg', dpi=400)
accuracy_gnb = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_gnb)
# print("Type du classifieur:", gnb)
# print("Rapport de la classification: ", reports_gnb)
print("Précision de la classification: %.2f" % accuracy_gnb)

print('********************  Classifieur (LDA) - Analyse discriminate linéaire  ***********************************')
params = {'shrinkage': None, 'solver': 'svd'}
y_pred_lda, lda = clf.lda_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_lda = tools.report_clf(estimator=lda, y_true=true_targets, y_pred=y_pred_lda, conf_matrix_plot=True)
plt.savefig('lda_conf_matrix.jpeg', dpi=400)
accuracy_lda = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_lda)
# print("Type du classifieur:", lda)
# print("Rapport de la classification: ", reports_lda)
print("Précision de la classification: %.2f" % accuracy_lda)

print('********************  Classification (DT): Arbre de Décision **********************************************')
params = {'max_depth':10, 'splitter':'random'}
y_pred_dt, dt = clf.dt_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_dt = tools.report_clf(estimator=dt, y_true=true_targets, y_pred=y_pred_dt, conf_matrix_plot=True)
plt.savefig('dt_conf_matrix.jpeg', dpi=400)
accuracy_dt = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_dt)

# print("Type du classifieur:", dt)
# print("Rapport de la classification: ", reports_dt)
print("Précision de la classification: %.2f" % accuracy_dt)

print('********************  Classification (RF): Forêt aléatoire **********************************************')
params = {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 4, 'n_estimators': 200}
y_pred_rf, rf = clf.rf_classifier(X_train=train, y_train=train_targets, X_true=true, **params)
reports_rf = tools.report_clf(estimator=rf, y_true=true_targets, y_pred=y_pred_rf, conf_matrix_plot=True)
plt.savefig('rf_conf_matrix.jpeg', dpi=400)
accuracy_rf = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_rf)
# print("Type du classifieur:", rf)
# print("Rapport de la classification: ", reports_rf)
print("Précision de la classification: %.2f" % accuracy_rf)

print('********************  Classifieur (SVM): Support Vector Machine  **********************************************')
params = {'C': 20, 'gamma': 'auto', 'kernel': 'rbf'}
y_pred_svm, svm = clf.svm_classifier(X_train=train, y_train=train_targets, X_true=true, fastmode=True,
                                     gridsearch=False, max_iter=50, **params)
reports_svm = tools.report_clf(estimator=svm, y_true=true_targets, y_pred=y_pred_svm, conf_matrix_plot=True)
plt.savefig('svm_conf_matrix.jpeg', dpi=400)
accuracy_svm = tools.accuracy_clf(y_true=true_targets, y_pred=y_pred_svm)
# print("Type du classifieur:", svm)
# print("Rapport de la classification: ", reports_svm)
print("Précision de la classification: %.2f" % accuracy_svm)
