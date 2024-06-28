import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from fakeStreamData import EEGGenrator

class RealtimePlot:
    def __init__(self, num_channels=8, sample_rate=250, x_range=5):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.x_range = x_range

        # Create empty array to store the data
        self.data = np.zeros((self.x_range * self.sample_rate, self.num_channels))

        # Create the plot
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Realtime EEG Plot")
        self.plots = [self.win.addPlot(row=i, col=1) for i in range(self.num_channels)]
        self.win.nextRow()

        # Set the range and axis labels for each plot
        for plot in self.plots:
            plot.setYRange(-1, 1)
            plot.setXRange(0, 1200)
            plot.setLabel('left', 'Amplitude')
            plot.setLabel('bottom', 'Time (s)')

        # Create the curves to plot the data
        self.curves = [plot.plot(pen=pg.mkPen(i, width=1)) for i, plot in enumerate(self.plots)]

        # Create a timer to update the plot every 40 ms (25 Hz)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

        # Start the application
        self.app.exec_()

    def update_plot(self):
        # Get the latest data from the EEG generator
        latest_data = next(data_gen)

        # Shift the existing data to the left and append the new data
        self.data[:-1] = self.data[1:]
        self.data[-1] = latest_data

        # Update the curves with the new data
        for i, curve in enumerate(self.curves):
            curve.setData(self.data[:, i], pen=pg.mkPen(i, width=1))

if __name__ == '__main__':
    fake_eeg = EEGGenrator()

    # Start the data generator and the realtime plot
    data_gen = fake_eeg.data_generator()
    realtime_plot = RealtimePlot()

    # Stop the data generator and close the CSV file when finished
    fake_eeg.close_csv_file()
