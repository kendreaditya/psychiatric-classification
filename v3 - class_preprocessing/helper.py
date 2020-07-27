from scipy.io import loadmat
import numpy as np
import h5py

class Database:
    def __init__(self, len_sec, sampling_freq, class_value, buffer_sec, path, data_filenames=None):
        self.len_sec = len_sec
        self.sampling_freq = sampling_freq
        self.class_value = class_value
        self.path = path
        self.buffer_sec = buffer_sec
        self.sample_len = int(self.len_sec/(1/self.sampling_freq))
        self.buffer_len = int(self.buffer_sec/(1/self.sampling_freq))
        self.data_filenames = data_filenames
    
    # Helper Method
    def set_file(self, file_path, data_name='EEG', channels_included=None):
        try:
            self.raw_data = loadmat(file_path)
        except:
            self.raw_data = h5py.File(file_path, 'r')
        print(self.raw_data.keys())
        self.EEG = self.r_indices(self.raw_data[data_name])
        if channels_included is not None:
            self.raw_channel_info = self.r_indices(self.raw_data[data_name][channels_included])

    # Helper Method
    def r_indices(self, array):
        try:
            while(len(array)==1):
                array = array[0]
            return array
        except:
            return array

    def get_EEG(self, file_path, data_name='EEG', channels_included=None):
        self.set_file(file_path, data_name, channels_included)
        return self.EEG      

class EEG:
    def __init__(self, rEEG, channel_locations, CLASS, seq_len=5, cutoff_freq=60):
        self.rEEG = rEEG
        self.channel_locations = channel_locations
        self.seq_len = seq_len
        self.CLASS = CLASS
        self.pEEG = preprocessing(self.rEEG)
        self.topoQEEG = topograph(self.pEEG)
        self.imgQEEG = img(self.topoQEEG)

    def preprocessing(self, EEG):
        return EEG
    
    def source_reconstruction(EEG, channel_locations, remaped_channel_locations):
        return EEG
    
    def topograph(self, EEG):
        return topoQEEG
    
    def img(self, topoQEEG):
        return imgQEEG

    def get_dict(self):
        return {'rawEEG':self.rEEG, 
                'preprocessedEEG':self.pEEG, 
                'topograph':self.topoQEEG, 
                'image':self.imgQEEG, 
                'channel_locations':self.channel_locations, 
                'class':self.CLASS, 
                'sequence_length':self.seq_len}