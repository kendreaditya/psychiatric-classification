# DEPR (class:1) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from helper import *

# Instanciate Database class
# DEPR0 and DEPR1 don't corrospond to database DEPR0 and DEPR1, but rather are a divison between
# 1-MDD & 2-had MDD
DEPR0 = Database(len_sec=10, sampling_freq=500, class_value=1, buffer_sec=0.5, path="../database/depression/DEPR0/")
DEPR1 = Database(len_sec=10, sampling_freq=500, class_value=2, buffer_sec=0.5, path="../database/depression/DEPR0/")


# Added files containing data to Database class
DEPR0_list, DEPR1_list = [], []
data_files = np.genfromtxt("../database/depression/DEPR0/trial-information/Data_4_Import_REST.csv", delimiter=",", dtype=str)
for file in data_files[1:]:
    if file[1] == '1':
        DEPR0_list.append(f"{file[0]}_Depression_REST.mat")
    elif file[1] == '2':
        DEPR1_list.append(f"{file[0]}_Depression_REST.mat")
DEPR0.data_filenames = DEPR0_list
DEPR1.data_filenames = DEPR1_list

# Iterate though files
for filename in tqdm(DEPR0.data_filenames):
    try:
        epochs = DEPR0.get_EEG(f"{DEPR0.path}/{filename}", data_name='data', channels="chanlocs", field='EEG')
        channel_locations = DEPR0.get_channel_locations()
        segmented_EEG = DEPR0.EEG_segmentation(epochs)
        print(segmented_EEG)
    except:
        pass

for filename in tqdm(DEPR1.data_filenames):
    try:
        epochs = DEPR1.get_EEG(f"{DEPR1.path}/{filename}", data_name='data', channels="chanlocs", field='EEG')
        channel_locations = DEPR1.get_channel_locations()
        segmented_EEG = DEPR1.EEG_segmentation(epochs)
        print(segmented_EEG)
    except:
        pass
