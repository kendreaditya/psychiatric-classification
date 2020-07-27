# ADHD (class:0) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from data_handler import data_handler
ORGIN_PATH = "C:/OneDrive - Cumberland Valley School District/EEG ScienceFair"

def batcher(array):
    if len(BATCH) == 99:
        BATCH.append(array)
        np.save(f'{BATCH_PATH}/raw{len(os.listdir(BATCH_PATH))}.npy', BATCH)
        BATCH.clear()
    else:
        BATCH.append(array)


CLASS_VALUE = {'ADHD':0, 'MDD':1, 'SCHIZO':2, 'NORM':3}['ADHD']

sfreq = 500
LEN_SEC = 9 # Change length
BUFF_SEC = 0.5

LENGTH = int(LEN_SEC / (1/sfreq))
BUFFER_ZONE = int(BUFF_SEC / (1/sfreq))
BATCH_PATH = f"{ORGIN_PATH}/database/pre-processed_data/raws/ADHD"
PATH = f"{ORGIN_PATH}/database/ADHD/ADH0"

dir = []
for fn in os.listdir(PATH):
    if fn[-3:]=='mat' and fn!='chan.mat':
        dir.append(fn)
    else:
        continue

BATCH = []

for filename in tqdm(dir):
    dl = data_handler(f"{PATH}\{filename}", data_name=filename[:-4])
    epochs = dl.get_EEG()

    for trial in epochs:
        data = np.transpose(trial) # shape: (5000, 56)
        try:
            for i in range(BUFFER_ZONE, len(data)-BUFFER_ZONE+LENGTH, LENGTH): # iterates by len
                EEG = np.transpose(data[i:i+LENGTH]) # before shape (len, 56) : after shape (56, len)
                batcher([EEG, [CLASS_VALUE]])
        except Exception as e:
            print(e)

np.save(f'{BATCH_PATH}/raw{len(os.listdir(BATCH_PATH))}.npy', BATCH)
