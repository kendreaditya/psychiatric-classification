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
                           channels="../database/ADHD/ADH0/chan.mat")
    channel_locations = ADHD0.get_channel_locations()
    segmented_EEG = ADHD0.EEG_segmentation(epochs)
    print(segmented_EEG.shape)
