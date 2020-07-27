# ADHD (class:0) Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

# Local modules
from data_handler import data_handler

numRaw = 0

def batcher(array):
    numRaw += 1
    if len(BATCH) == 99:
        BATCH.append(array)
        np.save(f'v2 - ADHD schizophrenia/pre-processed_data/raws/raw{numRaw}.npy', BATCH)
        BATCH.clear()
    else:
        BATCH.append(array)
    
        


CLASS_VALUE = {'ADHD':0, 'MDD':1, 'SCHIZO':2, 'NORM':3}['ADHD']

sfreq = 500
LEN_SEC = 4
BUFF_SEC = 0.5

LENGTH = int(LEN_SEC / (1/sfreq))
BUFFER_ZONE = int(BUFF_SEC / (1/sfreq))
PATH = "C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\ADHD\ADH0"

dir = []
for fn in os.listdir(PATH):
    if fn[-3:]=='mat' and fn!='chan.mat':
        dir.append(fn)
    else:
        continue

BATCH = []

for filename in tqdm(dir):
    dl = data_handler(f"{PATH}\{filename}", data_name=filename[:-4])
    fake_data = np.array([i for i in range(300*56*5000)]).reshape((300,56,5000))
    print(fake_data.shape)
    for trial in tqdm(fake_data):
        data = np.transpose(trial) # shape: (5000, 56)
        for i in range(BUFFER_ZONE, len(data)-BUFFER_ZONE+LENGTH, LENGTH): # iterates by len
            batcher([np.transpose(data[i:i+LENGTH]), [CLASS_VALUE]]) # before shape (len, 56) : after shape (56, len)
    break


