# SCHI (class:4) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from helper import *

# Instanciate Database class
SCHI0 = Database(len_sec=10, sampling_freq=128, class_value=4, buffer_sec=0.5, path="../database/depression/sch0/")


# Added files containing data to Database class
SCHI0_list= []
for filename in os.listdir(SCHI0.path):
    if filename[-4:] == ".eea":
        SCHI0_list.append(filename)

SCHI0.data_filenames = SCHI0_list

# Iterate though files
for fileame in tqdm(SCHI0.data_filenames):
    try:
        epochs = SCHI0.get_EEG(f"{SCHI0.path}/{filename}", data_name=None, channels=None, field=None, file_type='txt 7680'
        channel_locations = SCHI0.get_channel_locations()
        segmented_EEG = SCHI0.EEG_segmentation(epochs)
        print(segmented_EEG)
    except:
        pass
