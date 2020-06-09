# SCHIZOPHRENIA (class:2) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from data_handler import data_handler
ORGIN_PATH = "C:/OneDrive - Cumberland Valley School District/EEG ScienceFair"
CLASS_VALUE = {'ADHD':0, 'MDD':1, 'SCHIZO':2, 'NORM':3}['ADHD']

def batcher(array):
    if len(BATCH) == 99:
        BATCH.append(array)
        np.save(f'{BATCH_PATH}/raw{len(os.listdir(BATCH_PATH))}.npy', BATCH)
        BATCH.clear()
    else:
        BATCH.append(array)

sfreq = 500
LEN_SEC = 9 # Change length
BUFF_SEC = 0.5

LENGTH = int(LEN_SEC / (1/sfreq))
BUFFER_ZONE = int(BUFF_SEC / (1/sfreq))
BATCH_PATH = f"{ORGIN_PATH}/database/pre-processed_data/raws/SCHI"
PATH = f"{ORGIN_PATH}/database/schizophrenia/"

BATCH = []