import numpy as np

class fast_fourier_transform():
    def __init__(self, channel_data, freq=500):
        self.freq = freq
        self.channel_data = channel_data
        self.bands = {"Delta" : [0, 4],
                     "Theta" : [4, 7],
                     "Alpha" : [7, 13],
                     "Beta" : [13, 39],
                     "Gamma" : [39, 100]}

    def FFT(self, MAX_Hz):
        # calculates the high/amplatude of each frequency
        return np.abs(np.fft.rfft(self.channel_data))[:MAX_Hz]

    def get_bands(self):
        fft_amp = np.absolute(np.fft.rfft(self.channel_data))
        # gets the frequeinces present
        fft_freq = np.fft.rfftfreq(len(self.channel_data), 1.0/self.freq)

        band_fft = dict()
        for band in self.bands:
            freq = np.where((fft_freq >= self.bands[band][0]) & (fft_freq > self.bands[band][1]))
            band_fft[band] = np.mean(fft_amp[freq])
        
        return band_fft