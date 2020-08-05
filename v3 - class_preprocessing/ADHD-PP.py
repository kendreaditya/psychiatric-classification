# ADHD (class:0) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from helper import *

# Instanciate Database class
ADHD0 = Database(len_sec=10, sampling_freq=500, class_value=0, buffer_sec=0.5, path="../database/ADHD/ADH0")

# Added files containing data to Database class
dir = []
for fn in os.listdir(ADHD0.path):
    if fn[-3:]=='mat' and fn!='chan.mat':
        dir.append(fn)
ADHD0.data_filenames = dir

# Iterate though files
for filename in tqdm(ADHD0.data_filenames):
    epochs = ADHD0.get_EEG(f"{ADHD0.path}/{filename}", data_name=filename[:-4],
                           channels_included="../database/ADHD/ADH0/chan.mat")
    channel_locations = ADHD0.get_channel_locations()

    for trial in epochs:
        data = np.transpose(trial) # shape: (5000, 56)
        try:
            for i in range(ADHD0.buffer_len, len(data)-ADHD0.buffer_len, ADHD0.sample_len): # iterates by len
                EEG = np.transpose(data[i:i+ADHD0.sample_len]) # before shape (len, 56) : after shape (56, len)
                #batcher([EEG, [ADHD0.class_value]])
        except Exception as e:
            print(e)
