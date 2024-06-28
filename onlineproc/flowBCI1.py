import numpy as np
import pywt
from scipy.signal import butter, filtfilt
import time

# Define parameters
sampling_rate = 250  # Hz
n_channels = 8
n_samples = 1000  # number of samples to generate for simulation
f_low = 8  # Hz
f_high = 30  # Hz
n_wavelet_coeffs = 30  # number of wavelet coefficients to use for feature extraction


# Define function to simulate generating EEG data at 250 Hz with 8 channels
def generate_eeg_data():
    # Generate random EEG data with mean 0 and variance 1
    eeg_data = np.random.randn(n_channels)
    # Wait for 4 ms to simulate 250 Hz sampling rate
    time.sleep(0.004)
    return eeg_data


# Define function to filter EEG data between f_low and f_high using a Butterworth filter
def filter_eeg_data(eeg_data_):
    nyquist_freq = 0.5 * sampling_rate
    filter_order = 4
    b, a = butter(filter_order, [f_low / nyquist_freq, f_high / nyquist_freq], btype='bandpass')
    filtered_eeg_data_ = filtfilt(b, a, eeg_data_)
    return filtered_eeg_data_


# Define function to normalize EEG data between -1 and 1 using min-max normalization
def normalize_eeg_data(eeg_data_):
    min_val = np.min(eeg_data_)
    max_val = np.max(eeg_data_)
    normalized_eeg_data_ = (eeg_data_ - min_val) / (max_val - min_val) * 2 - 1
    return normalized_eeg_data_


# Define function to extract wavelet-based features from EEG data
def extract_wavelet_features(eeg_data_):
    wavelet_name = 'db4'
    coeffs = pywt.wavedec(eeg_data_, wavelet_name)
    wavelet_features_ = np.concatenate([coeff[:n_wavelet_coeffs] for coeff in coeffs])
    return wavelet_features_


# Define function to classify EEG data into left (class 0) or right (class 1) using a simple threshold
def classify_eeg_data(eeg_data_):
    threshold = 0
    if np.mean(eeg_data_) > threshold:
        return 1
    else:
        return 0


# Simulate generating EEG data and processing it for online BCI
for i in range(n_samples):
    # Generate EEG data
    eeg_data_ = generate_eeg_data()
    # Filter EEG data
    filtered_eeg_data_ = filter_eeg_data(eeg_data_)
    print(filtered_eeg_data_)
    # # Normalize EEG data
    # normalized_eeg_data_ = normalize_eeg_data(filtered_eeg_data_)
    # # Extract wavelet-based features from EEG data
    # wavelet_features_ = extract_wavelet_features(normalized_eeg_data_)
    # # Classify EEG data into left (class 0) or right (class 1)
    # class_label = classify_eeg_data(wavelet_features_)
    # print('Class label:', class_label)
