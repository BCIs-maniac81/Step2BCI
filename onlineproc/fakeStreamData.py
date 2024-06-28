import numpy as np
import time
import datetime
import csv


class EEGGenerator(object):
    def __init__(self, num_channels=8, sampling_rate=250):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        # Initialize variables for generating fake EEG data
        self.time = 0
        self.count = 0

        # Initialize variables for saving data to CSV file
        self.filename = None
        self.csv_file = None
        self.csv_writer = None

    def generate_sample(self):
        # self.count += 1
        eeg_signal = np.zeros((self.num_channels,))
        for ch in range(self.num_channels):
            # Generate random coefficients for a 4th order autoregressive (AR) model
            amp = np.random.randint(2, 5)
            ar_coeffs = np.random.normal(0, 0.1, size=4)
            # Generate random white noise
            noise = np.random.normal(0, 1)
            # Generate the AR signal by filtering the white noise
            eeg_signal[ch] = amp * np.sum(ar_coeffs * eeg_signal[ch]) + noise
            # print('count: ', self.count)
        return eeg_signal  # return 1 sample array

    def data_generator(self):
        # Generator that yields one sample at a time, with a delay to simulate a sampling rate of 250 Hz
        while True:
            yield self.generate_sample()
            start_time = time.perf_counter()  # get the current time in seconds with high resolution
            while time.perf_counter() - start_time <= 0.004:  # loop until 4 millisecond has elapsed
                pass  # do nothing

    def create_csv_file(self):
        # Create a new CSV file with a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"./fakeEEG/fake_eeg_{timestamp}.csv"
        self.csv_file = open(self.filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write the header row with channel names
        channel_names = [f"Channel {i + 1}" for i in range(self.num_channels)]
        self.csv_writer.writerow(channel_names)

    def append_to_csv(self, sample):
        # Append one sample of data to the CSV file
        if self.csv_writer is None:
            self.create_csv_file()
        self.csv_writer.writerow(sample)

    def close_csv_file(self):
        # Close the CSV file when finished
        if self.csv_file is not None:
            self.csv_file.close()


if __name__ == '__main__':

    start_ = time.time()
    n_samples = 2000
    sample_ = 0
    eeg = EEGGenerator(num_channels=8, sampling_rate=250)

    while sample_ <= n_samples:
        sample = next(eeg.data_generator())
        print(sample)
        eeg.append_to_csv(sample)
        sample_ += 1
    eeg.close_csv_file()
    end_ = time.time()
    print(end_ - start_)
