from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pywt
import mne


def generate_fake_eeg(duration, sampling_rate=256):
    # Generate random EEG data with 8 channels
    num_samples = duration * sampling_rate
    eeg_data = np.random.normal(0, 1, size=(num_samples, 8))
    # Generate timestamps
    timestamps = np.arange(num_samples) / sampling_rate
    return eeg_data, timestamps


# def preprocess_eeg_data(data):
#     # Apply a high-pass filter to remove baseline drift
#     data = data - np.mean(data, axis=0)
#     data = np.dot(data, np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1], [1, -2, 1]]))
#     # Apply a notch filter to remove 60Hz noise
#     notch_freq = 60.0
#     for i in range(data.shape[1]):
#         freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / 256)
#         fft = np.fft.rfft(data[:, i])
#         fft[np.where(np.abs(freqs - notch_freq) < 1)] = 0
#         data[:, i] = np.fft.irfft(fft)
#     # Apply standard scaling
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)
#     return data


def preprocess_eeg_data(eeg_data, low_freq, high_freq, notch_freq=50, sfreq=256):
    # Bandpass filter between 4-30 Hz
    eeg_data = mne.filter.filter_data(eeg_data.T, sfreq, low_freq, high_freq, method='iir').T
    # Baseline correction
    eeg_data = mne.baseline.rescale(eeg_data.T, sfreq=sfreq, window=None, baseline=(None, 0),
                                    mode='zscore', copy=True).T
    # Notch filter at 50 Hz or 60 Hz
    eeg_data = mne.filter.notch_filter(eeg_data.T, sfreq=sfreq, freqs=notch_freq, method='iir', verbose=False).T
    # Standard scaling
    scaler = StandardScaler()
    eeg_data = scaler.fit_transform(eeg_data)
    return eeg_data


def feature_extraction(eeg_data, wavelet='db4', level=5):
    # Decompose EEG data using wavelet decomposition
    coeffs = pywt.wavedec(eeg_data, wavelet, level=level)
    # Extract approximation and detail coefficients from each level
    features = []

    for i in range(level + 1):
        features.append(coeffs[i].mean(axis=0))
        features.append(coeffs[i].std(axis=0))

    return np.concatenate(features)


def online_bci(duration, eeg_generator):
    # Get EEG data and timestamps from the generator
    eeg_data, timestamps = eeg_generator(duration)
    # Preprocess the data
    preprocessed_data = preprocess_eeg_data(eeg_data)
    # Define the labels for our classification model
    labels = np.zeros(preprocessed_data.shape[0])
    labels[timestamps > timestamps[0] + duration / 2] = 1
    # Train a logistic regression model
    clf = LogisticRegression(random_state=0).fit(preprocessed_data, labels)
    # Get new EEG data and classify it in real-time
    for i in range(eeg_data.shape[0]):
        sample = eeg_data[i:i + 1]
        data = preprocess_eeg_data(sample)
        prediction = clf.predict(data)[0]
        if prediction == 0:
            print("Relaxation")
        else:
            print("Concentration")


if __name__ == '__main__':
    eeg_data, timestamp = generate_fake_eeg(1)

    eeg_data = preprocess_eeg_data(eeg_data, 4, 30, notch_freq=50, sfreq=256)
    print(eeg_data.shape)
