# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:45:28 2021

@author: hafed-eddine BENDIB
"""
import numpy as np
import matplotlib.pyplot as plt
import preproc.IOfiles as iof
import preproc.trialsProcessor as trproc
import preproc.dataScaler as scal
import preproc.dataFilter as filt
import preproc.dataPlotter as plotter
import preproc.dataAnalyzer as analyzer


# étape 1: méta-données Graz B01T
"""
chemain d'accés ../datasets/grazdata/
Nom du fichier: B01T
Classes cibles (targets): 1 (Left), 2 (Right)
Paradigme: Imagination motrice
électrodes: C3, Cz, C4
Colonnes: 1-C3, 2-Cz, 3-C4
"""
# Preprocessing
# étape 2: Charger les données Graz
X, trials, targets, attributes = iof.read_graz_file('../datasets/grazdata/B01T.mat')
print(X.shape)
X = X[:, :3]
print(X.shape)

sfreq = attributes['sfreq']
classes = attributes['Classes']
target_codes = np.sort(np.unique(targets))
trial_period = [0, 8]
channels = ['C3', 'Cz', 'C4']

# étape2 (Optionnelle): re-référencer les données EEG par rapport l'électrode Cz
cond = False
if cond:
    X_ = X.copy()
    X_[:, 0] = X_[:, 0] - 2 * X_[:, 1]
    X_[:, 2] = X_[:, 0] - 2 * X_[:, 2]

# étape3: Filtrage des données EEG
# a) Application du filtre passe-bande [2, 60]
signal0 = filt.band_pass(inputData=X, lowfreq=2, highfreq=60, order=4, sfreq=sfreq, t_ax=0)
# b) Application du notch à 50 Hz
signal1 = filt.notch_filter(inputData=signal0, notch_freq=50, quality_fact=50, sfreq=sfreq, t_ax=0)


kwargs = dict(title='Tracé EEG', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(signal1[0 * sfreq:400 * sfreq, :], trials=trials, sfreq=sfreq, xlim=None, ylim=[None, None],
                         channels=channels, **kwargs)
# plt.savefig('400s_trials_eeg_c3.jpeg', dpi=300)
# étape4: STFT
# Deux méthodes
freqs1, times1, Zxx1 = analyzer.stft_analysis(signal1[:, 0], plot=True, sfreq=sfreq, t_ax=0)
freqs2, times2, Zxx2 = analyzer.short_time_ft(signal1[:, 0], plot=True, sfreq=sfreq, t_ax=0)
# étape5: Spectrogramme
freqs, times, im, r_Sxx = analyzer.spectrogram_analysis(signal1[:, 0], 256, plot=True, cmap=plt.cm.gnuplot2)

fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(0, r_Sxx.shape[-1], 1):
    # ax1.plot(freqs, (r_Sxx[:, i]), lw=1.2)
    ax1.plot(freqs, r_Sxx[:, i], lw=1.2)
# ax1.set_title("Densité spectrale de puissance DSP EEG-C3 pour chaque essais", fontsize=14)
ax1.set_xlabel("Fréquences ($Hz$)", fontsize=12)
ax1.set_ylabel("Spectre STFT", fontsize=12)
ax1.axis([freqs.min(), freqs.max(), None, None])
ax1.grid()
mesh = ax2.pcolormesh(times, freqs, 10 * np.log10(r_Sxx), cmap=plt.cm.gnuplot2, shading="auto")  # plt.cm.gist_heat)
colorbar = plt.colorbar(mesh)
ax2.set_title("Spectrogramme 3D du signal EEG-C3", fontsize=12)
ax2.set_xlabel("Temps (s)", fontsize=12)
ax2.set_ylabel("Fréquences ($Hz$)", fontsize=12)
plt.tight_layout()
plt.show()
# plt.savefig('sttf_spectrogramme.jpeg', dpi=400)
# étape6: Normalisation
normalized_signal = scal.std_scaler(signal1, with_std=True, with_mean=0)
print("Caractéristiques du signal normalisé: \n Moyenne{:.2f} \n Max {:.2f} \n Min {:.2f} \n Std_dev {:.2f}" \
      .format(np.nanmean(normalized_signal), np.nanmax(normalized_signal),
              np.nanmin(normalized_signal), np.nanstd(normalized_signal)))

kwargs = dict(title='Tracé de l\'EEG normalisé', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(normalized_signal[:3000, :], trials=None, sfreq=sfreq, xlim=None,
                         ylim=[None, None], channels=channels, **kwargs)
# étape7: Représenter le premier epoch
first_epoch = signal1[trials[0]: trials[1], :]
kwargs = dict(title='Tracé EEG du $1^{er}$ epoch', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(inputData=first_epoch, trials=None, sfreq=sfreq, xlim=None, ylim=[None, None],
                         channels=channels, **kwargs)
# plt.savefig('first_epoch_c3.jpeg', dpi=300)
# étape8: Visualiser la bande Alpha et Beta du dernier epoch
# bande Alpha = [7, 13] Hz, Beta = [16, 24]
bands = ['alpha', 'beta']
alpha = [7, 13]
beta = [16, 24]
# étape9: éxtraction des ondes alpha et beta
alpha_wave = filt.band_pass(signal0, lowfreq=alpha[0], highfreq=alpha[1], order=6, sfreq=sfreq, t_ax=0)
beta_wave = filt.band_pass(signal0, lowfreq=beta[0], highfreq=beta[1], order=6, sfreq=sfreq, t_ax=0)
# Tracé
kwargs = dict(title='Tracé de l\'éxtrait de la bande alpha', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(inputData=alpha_wave[100000:102000, :], trials=trials, sfreq=sfreq,
                         channels=channels, **kwargs)
kwargs = dict(title='Tracé de l\'éxtrait de la bande beta', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(inputData=beta_wave[100000:102000, :], trials=trials, sfreq=sfreq,
                         channels=channels, **kwargs)

# vérifier si l'Intervalle d'occurence de l'imagination motrice [-4s, +4s]
trial_n = -1
mi_range = [4 * attributes['sfreq'], 4 * attributes['sfreq']]

alpha_last_trial = [0, 0, 0]
beta_last_trial = [0, 0, 0]

for i in range(3):
    alpha_last_trial[i] = alpha_wave[trials[trial_n] - mi_range[0]: trials[trial_n] + mi_range[1], i]
    beta_last_trial[i] = beta_wave[trials[trial_n] - mi_range[0]: trials[trial_n] + mi_range[1], i]
alpha_last_trial = np.asarray(alpha_last_trial).T.copy()
beta_last_trial = np.asarray(beta_last_trial).T.copy()

kwargs = dict(title='Dernier epoch Alpha', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(alpha_last_trial, trials=None, sfreq=sfreq, channels=channels, **kwargs)
kwargs = dict(title='Dernier epoch Beta', xlabel="Temps ($s$)", ylabel="Amplitude ($uV$)",
              sharex=True, sharey=True, figsize=None)
plotter.data_trials_plot(beta_last_trial, trials=None, sfreq=sfreq, channels=channels, **kwargs)

band_data = {bands[0]: alpha, bands[1]: beta}

# colors = {0: 'green', 1: 'steelblue', 2: 'red'}
# alpha = {0: 1, 1: 0.8, 2: 0.6}
last_trials = [alpha_last_trial, beta_last_trial]
plotter.trial_band_plot(last_trials)

# étape10: Extraction des éssais (trials)
print(type(trials))
print(type(targets))

trials_alpha_band = trproc.trials_extract(inputData=alpha_wave, trials=trials, targets=targets, trial_size=8,
                                          n_channels=3, offset=0, version='graz', sfreq=sfreq)

trials_beta_band = trproc.trials_extract(inputData=beta_wave, trials=trials, targets=targets, trial_size=8,
                                         n_channels=3, offset=0, version='graz', sfreq=sfreq)
print(type(trials_alpha_band))
print('Les classes (cibles, targets) sont: ', trials_alpha_band.keys())
print('Format des essais (trials) pour la classe 1 (Left): ',
      trials_alpha_band[list(trials_alpha_band.keys())[0]].shape)
"""
trials_alpha_band et trials_beta_band sont 2 dictionnaires ayant des clés cibles 1, 2 [classes Left & Right]
chaque trial est d'une durée de 8 secondes.
chaque classe contient n trials (éssais) de longueur 8s * 250Hz = 2000 éch.
le format de chaque classes est (z, x, y)
z: est le nombre des canaux
x: est le nombre des éssais (trials)
y: est le nombre des échantillons dans chaque éssai.
"""
all_trials = {'alpha': trials_alpha_band, 'beta': trials_beta_band}
plotter.all_trials_classes_plot(all_trials, bands=bands, classes=classes, pos=channels)

# Calcul de la densité spectrale moyenne des essais extraits pour les cibles 1 et 2, dans la plage Alpha [7, 13]
new_channels = ['C3', 'C4']


power_c3_class1_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[0]][:, :, 0], t_ax=-1)
power_cz_class1_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[0]][:, :, 1], t_ax=-1)
power_c4_class1_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[0]][:, :, 2], t_ax=-1)
power_c3_class2_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[1]][:, :, 0], t_ax=-1)
power_cz_class2_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[1]][:, :, 1], t_ax=-1)
power_c4_class2_a = analyzer.average_power(inputData=all_trials[bands[0]][target_codes[1]][:, :, 2], t_ax=-1)

average_power_data_a = [[power_c3_class1_a, power_c4_class1_a], [power_c3_class2_a, power_c4_class2_a]]

# Tracé
plotter.average_power_plot(average_power_data_a, classes=classes, channels=new_channels, trial_period=[0, 8],
                           sfreq=sfreq, colors=['purple', 'steelblue'], bands=bands[0])
# plt.show()
#
power_c3_class1_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[0]][:, :, 0], t_ax=-1)
power_cz_class1_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[0]][:, :, 1], t_ax=-1)
power_c4_class1_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[0]][:, :, 2], t_ax=-1)
power_c3_class2_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[1]][:, :, 0], t_ax=-1)
power_cz_class2_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[1]][:, :, 1], t_ax=-1)
power_c4_class2_b = analyzer.average_power(inputData=all_trials[bands[1]][target_codes[1]][:, :, 2], t_ax=-1)

average_power_data_b = [[power_c3_class1_b, power_c4_class1_b], [power_c3_class2_b, power_c4_class2_b]]

# Tracé
plotter.average_power_plot(average_power_data_b, classes=classes, channels=new_channels, trial_period=[0, 8],
                           sfreq=sfreq, colors=['purple', 'steelblue'], bands=bands[1])
plt.show()
