from data_handler import data_handler
import numpy as np
import matplotlib.pyplot as plt

class channel_names():
    def __init__(self, file_path):
        self.names = dict()
        dl = data_handler(file_path)
        list_names = dl.get_channel_names()
        for name in list_names[:64]:
            self.names[name] = None

    def get_names_dict(self):
        return self.names


class channel_band_amps():
    def __init__(self, file_path):
        self.file_path = file_path
        self.bands = {"Delta" : None,
                     "Theta" : None,
                     "Alpha" : None,
                     "Beta" : None,
                     "Gamma" : None}
        for band in self.bands.keys():
            c_names = channel_names(self.file_path)
            self.bands[band] = c_names.get_names_dict()
    
    def add_band_amp(self, single_channel, channel_name):
        for band in single_channel.keys():
            self.bands[band][channel_name] = single_channel[band]

    def get_bands_amp(self):
        return self.bands

class qEEG():
    def __init__(self, bands, file_path):
        self.bands = bands
        self.file_path = file_path
    
    def graph_XY(self):
        SIZE = 5

        dl = data_handler(self.file_path)
        XYZ = dl.get_channel_locations()
        channel_names = dl.get_channel_names()

        for band in self.bands.keys():
            QEEG = np.zeros((SIZE*2+1, SIZE*2+1))
            
            X = np.array(XYZ[0])
            X = (X + abs(np.amin(X)))
            X = (X / np.amax(X)) * SIZE*2

            Y = np.array(XYZ[1])
            Y = (Y + abs(np.amin(Y)))
            Y = (Y / np.amax(Y)) * SIZE*2

            for x, y, channel_name in zip(X[:64], Y[:64], channel_names[:64]):
                QEEG[int(x)][int(y)] = self.bands[band][channel_name]
            
            plt.imshow(QEEG, interpolation = 'bessel')
            plt.show()


