import numpy as np
from scipy.io import loadmat
import os
from data_handler import * 

PATH = "C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Matlab Files"

for filename in os.listdir(PATH):
    dl = data_handler(f"{PATH}/{filename}")
    dl.plot_channel_locations()
    break
