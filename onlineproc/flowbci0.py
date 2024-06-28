import numpy as np
from scipy.signal import welch
import pywt
from scipy.signal import butter, filtfilt
import time
from fakeStreamData import EEGGenerator
import matplotlib.pyplot as plt

# Define general parameters
sfreq = 250
num_channels = 8
time_buffer = 4
device_started = False

# Define pass band filter parameters
nyquist = 0.5 * sfreq  # La fréquence de Nyquist
mu_low_cutoff = 7  # low cutoff frequency of mu band
mu_high_cutoff = 12  # high cutoff frequency of mu band

beta_low_cutoff = 13  # low cutoff frequency of mu band
beta_high_cutoff = 30  # high cutoff frequency of mu band

notch_cutoff = [50, 60]
order = 4

b_mu, a_mu = butter(order, [mu_low_cutoff / nyquist, mu_high_cutoff / nyquist], btype='band')
b_beta, a_beta = butter(order, [beta_low_cutoff / nyquist, beta_high_cutoff / nyquist], btype='band')

# Define variables for pre-processing and post-processing functions
buffer_size = sfreq * time_buffer  # Le nombre d'échantillons à mettre dans le tampon.
data_buffer = np.zeros((buffer_size, num_channels))  # Initialisation du tampon


# window_size = 250  # Le nombre d'échantillons à mettre dans chaque fenêtre.
# window_overlap = 125  # Le nombre d'échantillons qui se chevauchent entre les fenêtres.
# window_buffer = np.zeros((window_size, num_channels))  # Initialisation de la fenêtre.
# window_index = 0  # indice de la fenêtre courante.
# window_count = 0  # Compteur du nombre de fenêtres traitées
# clf = LinearDiscriminantAnalysis()  # Initialisation du classifieur

# Define function to filter EEG data between f_low and f_high using a Butterworth filter
def filter_eeg_data(eeg_data, f_low, f_high):
    nyquist_freq = 0.5 * sfreq
    filter_order = 4
    b, a = butter(filter_order, [f_low / nyquist_freq, f_high / nyquist_freq], btype='bandpass')
    filtered_eeg_data = filtfilt(b, a, eeg_data, axis=0)
    return filtered_eeg_data


def extract_psd_features(eeg_signal, fs=250, nperseg=250):
    f, pxx = welch(eeg_signal, fs=fs, nperseg=nperseg, noverlap=4, axis=0)
    print(pxx.shape)
    # Extract the desired PSD features (e.g., mean, median, max) for each band
    psd_features = [np.max(pxx, axis=0), np.mean(pxx, axis=0), np.median(pxx, axis=0)]
    return psd_features


def classify_eeg_data(data_buffer_):
    pass


def button_push(push_=True):
    return push_


if __name__ == '__main__':
    buffer_index = 0
    eeg = EEGGenerator(num_channels=8, sampling_rate=250)
    while True:
        push = button_push()
        if push:
            offset = 3
            time.perf_counter()
            while time.time() - time.perf_counter() <= offset:
                pass
            device_started = True
            break

    s = 0
    while buffer_index < buffer_size:
        sample = next(eeg.data_generator())
        data_buffer[buffer_index, :] = sample
        buffer_index += 1

        if buffer_size == buffer_index:
            buffer_index = 0
            mu_filtered_data_buffer = filter_eeg_data(data_buffer, mu_low_cutoff, mu_high_cutoff)
            mu_psd_features = extract_psd_features(mu_filtered_data_buffer, fs=250, nperseg=250)
            print(mu_psd_features)
            beta_filtered_data_buffer = filter_eeg_data(data_buffer, beta_low_cutoff, beta_high_cutoff)
            beta_psd_features = extract_psd_features(beta_filtered_data_buffer, fs=250, nperseg=250)
            print(beta_psd_features)
            print(mu_psd_features + beta_psd_features)
            break


fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(data_buffer[:, 0])
ax[1].plot(mu_filtered_data_buffer[:, 0])
ax[2].plot(beta_filtered_data_buffer[:, 0])
plt.show()
