from scipy.io import loadmat
import numpy as np
import h5py

class data_handler():
    # file_path: path to the matlab file
    # data_name: name of data array under matlab file
    # channels_included: name of channel information array under data in the matlab file

    def __init__(self, file_path, data_name='EEG', channels_included=None):
        try:
            self.raw_data = loadmat(file_path)
        except:
            self.raw_data = h5py.File(file_path, 'r')
        print(self.raw_data.keys())
        self.EEG = self.r_indices(self.raw_data[data_name])
        if channels_included is not None:
            self.raw_channel_info = self.r_indices(self.raw_data[data_name][channels_included])

    def r_indices(self, array):
        try:
            while(len(array)==1):
                array = array[0]
            return array
        except:
            return array

    def get_EEG(self):
        return self.EEG

    def get_channel_locations(self, x_inx, y_inx, z_inx):
        xyz = []
        for channel in self.raw_channel_info:
            try:
                xyz.append([self.r_indices(channel[x_inx]), self.r_indices(channel[y_inx]), self.r_indices(channel[z_inx])])
            except:
                xyz.append([None, None, None])
        return xyz

    def get_channel_names(self, name_idx=0):
        return [self.r_indices(info[name_idx]) for info in self.raw_channel_info]