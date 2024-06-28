# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:04:49 2023

@author: bendi
"""

import numpy as np

import pyqtgraph as pg
from fakeStreamData import EEGGenerator


class RealTimeEEGPlot(object):
    def __init__(self, num_channels=8, sample_rate=250, duration=5):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.duration = duration
        self.ptr=0
        self.eeg = EEGGenerator()
        
        # Create the plot
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('Realtime EEG Plot')
        self.win.setGeometry(100, 100, 1024, 728)
        
        # Initialize the plot
        self.plot_items = []
        self.labels = [f'CH {i}' for i in range(1, self.num_channels + 1)]
        
        # Create empty array to store the data
        self.data = np.zeros((self.duration * self.sample_rate, self.num_channels))
        
        
        for i in range(self.num_channels):
            self.plot_items.append(self.win.addPlot(row=i, col=1, title=self.labels[i]))
        self.win.nextRow()
        
        
        # set the range and axis label of each plot
        for plot in self.plot_items:
            # plot.setYRange(-1, 1)
            plot.setLabel('left', 'uV')
            plot.setLabel('bottom', 'Time (s)')
            #plot.setXRange(0, self.plot_duration * self.sample_rate, padding=0)
            #plot.setYRange(-1, 1)
            #plot.showGrid(x=True, y=True)
            #plot.setMouseEnabled(x=True, y=True)
            #plot.setDownsampling(mode='peak')
            #plot.setClipToView(True)
        
        # Create the curves to plot the data
        self.curves = [plot.plot(pen=pg.mkPen(i, width=1)) for i, plot in enumerate(self.plot_items)]
        
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(4)
        
    
    def update(self):
        sample = eeg.data_generator()
        # self.ptr += 1
        # Get the latest data from the EEG generator
        latest_data = next(sample)
        eeg.append_to_csv(latest_data)
        print(latest_data)

        # Shift the existing data to the left and append the new data
        self.data[1:] = self.data[0:-1]
        self.data[0] = next(sample)

        # Update the curves with the new data
        for i, curve in enumerate(self.curves):
            curve.setData(self.data[:, i], pen=pg.mkPen(i, width=1))
            # curve.setPos(self.ptr, 0)
            
        
    
    def run(self):
        pg.exec()
        
if __name__ == '__main__':
    
    eeg = EEGGenerator()

    # Start the data generator and the realtime plot
    #sample = eeg.data_generator()
    #print(sample)
    #eeg.append_to_csv(sample)
    eeg_plot = RealTimeEEGPlot()
    eeg_plot.run()
    
    # Stop the data generator and close the CSV file when finished
    eeg.close_csv_file()