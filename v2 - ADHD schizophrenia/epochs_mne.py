import mne
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

chan = loadmat("C:/Users/akend/Downloads/data/ADHD/ADH0/chan.mat")['chan'][0]
d1 = h5py.File("C:/Users/akend/Downloads/data/ADHD/ADH0/d1.mat", 'r')['d1']

# Info object
ch_names = [str(obj[0][0]) for obj in chan]
[ch_names.remove(bad_ch) for bad_ch in ["P9", "P10", "P11", "P12"]]
ch_types = 'eeg'
info = mne.create_info(ch_names, 500, ch_types=ch_types)

# Creating Montage
pos = [[float(obj[8][0]), float(obj[7][0]), float(obj[9][0])] for obj in chan] # x&y are flited to rate the graph
montage = mne.channels.Montage(pos=pos, ch_names=ch_names, kind='motor-cap', selection=range(57))
#montage.plot()

# Alternate montage
"""dig_ch_pos = dict(zip(ch_names,pos))
montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame='head')
"""
# Loading data
epochs = mne.EpochsArray([d1[0]], info)
epochs.set_montage(montage, set_dig=True)
#epochs.info['bads'] = ["P8"]

# Topomap
epochs.plot_psd_topomap(normalize=True)
#img = epochs.plot_psd_topomap(normalize=True)

