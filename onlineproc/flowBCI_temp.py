import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal

class Preprocessing:
    def __init__(self, sfreq=250, highcut=40, lowcut=0.5, notch=True, notch_freq=50):
        self.sfreq = sfreq
        self.highcut = highcut
        self.lowcut = lowcut
        self.notch = notch
        self.notch_freq = notch_freq
        self.b, self.a = self._design_filter()

    def _design_filter(self):
        nyquist = 0.5 * self.sfreq
        high = self.highcut / nyquist
        low = self.lowcut / nyquist

        b, a = signal.butter(4, [low, high], btype='band')

        if self.notch:
            notch_freq = self.notch_freq / nyquist
            q = 30.0
            b_notch, a_notch = signal.iirnotch(notch_freq, q)
            b = np.convolve(b, b_notch)
            a = np.convolve(a, a_notch)

        return b, a
    
    def _apply_filter(self, data):
        filtered_data = signal.filtfilt(self.b, self.a, data, axis=0)
        return filtered_data
    


    def remove_baseline(self, eeg_data):
        baseline = np.mean(eeg_data, axis=0, keepdims=True)
        baseline_removed_data = eeg_data - baseline
        return baseline_removed_data

    def _standardize_data(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        standardized_data = (data - mean) / std
        return standardized_data
    
    def epoching_data(self, eeg_data, epoch_size=1):
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(-1, 1)
        num_epochs = eeg_data.shape[0] // (self.sfreq * epoch_size)
        if num_epochs == 0:
            perpared_data = self._apply_filter(eeg_data)
            #prepared_data = self.apply_notch_filter(prepared_data)
            prepared_data = self.remove_baseline(prepared_data)
            prepared_data = self._standardize_data(prepared_data)
            return prepared_data
        prepared_data = np.zeros((num_epochs, self.sfreq * epoch_size, eeg_data.shape[1]))
        for i in range(num_epochs):
            epoch_start = i * self.sfreq * epoch_size
            epoch_end = epoch_start + self.sfreq * epoch_size
            epoch_data = eeg_data[:, epoch_start:epoch_end]
            epoch_data = self._apply_filter(epoch_data)
            #epoch_data = self.apply_notch_filter(epoch_data)
            epoch_data = self.remove_baseline(epoch_data)
            epoch_data = self._standardize_data(epoch_data)
            prepared_data[i] = epoch_data
        return prepared_data



# Generate a simulated EEG signal with 2 seconds of data at 250 Hz
t = np.arange(0, 2, 1/250)
eeg_data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + np.random.randn(1, len(t))
# print(eeg_data.shape)
# Create a Preprocessing object and preprocess the data
preprocessor = Preprocessing()
preprocessed_data = preprocessor.epoching_data(eeg_data)
# print(preprocessed_data.shape)

# Plot the original and preprocessed signals
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(t, eeg_data[0])
axs[0].set_title('Original EEG signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
#axs[1].plot(np.arange(0, 2, 1/250), preprocessed_data[0][0])
axs[1].set_title('Preprocessed EEG signal')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
plt.show()

