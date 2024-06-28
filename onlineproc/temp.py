import numpy as np
from scipy.signal import welch
import pywt
import time
from fakeStreamData import EEGGenerator



def extract_psd_features(eeg_signal, fs=250, nperseg=250):
    f, pxx = welch(eeg_signal, fs=fs, nperseg=nperseg, noverlap=4)

    # Define the frequency ranges for the mu and beta bands
    mu_band = (f >= 8) & (f < 13)
    beta_band = (f >= 13) & (f < 30)

    # Compute the PSD in the mu and beta band
    mu_psd = pxx[mu_band]
    beta_psd = pxx[beta_band]

    # Extract the desired PSD features (e.g., mean, median, max) for each band
    mu_psd_features = [np.max(mu_psd, axis=0), np.mean(mu_psd, axis=0), np.median(mu_psd, axis=0)]
    beta_psd_features = [np.max(beta_psd, axis=0), np.mean(beta_psd, axis=0), np.median(beta_psd, axis=0)]

    # Concatenate the PSD features for each band into a single feature vector
    psd_features_ = mu_psd_features + beta_psd_features
    print(psd_features.shape)
    return psd_features_




if __name__ == '__main__':
    n_samples = 1
    sample_ = 0
    epoch_length = 1 # in seconds
    n_epochs = 10
    sfreq = 250
    buffer = np.zeros((epoch_length * sfreq * ))
    eeg = EEGGenerator(num_channels=8, sampling_rate=250)

    while sample_ < n_samples:
        sample = next(eeg.data_generator())
        print(sample)
        psd_features = extract_psd_features(sample, nperseg=8)
        # eeg.append_to_csv(sample)
        sample_ += 1
    # eeg.close_csv_file()

print(psd_features)
