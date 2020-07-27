import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from data_handler import *
from fast_fourier_transform import  *
from channel_band_amps import channel_band_amps, channel_names, qEEG
import os

PATH = "C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Matlab Files"
band_amps = []

for filename in os.listdir(PATH):
	file_path = f"{PATH}/{filename}"

	dl = data_handler(file_path)		# Add inheritance between data_handler & channel_bands_amp
	cba = channel_band_amps(file_path)

	for channel_data, channel_name in zip(dl.get_EEG()[:64], dl.get_channel_names()[:64]):
		fft = fast_fourier_transform(channel_data[:15000])
		channel_bands = fft.FFT()
		cba.add_band_amp(channel_bands, channel_name)
		
	band_amps = cba.get_bands_amp()
	QEEG = qEEG(band_amps, file_path)
	QEEG.graph_XY()
	break
