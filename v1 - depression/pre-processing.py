import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from data_handler import data_handler
from fast_fourier_transform import  fast_fourier_transform
from channel_band_amps import channel_band_amps, channel_names, qEEG

def is_MDD(filename):
	val = [1,2,50,99]
	subject = filename[:3]
	for fields in MDD_list:
		if subject == fields[0] and fields[1]!='50':
			return np.eye(len(val))[val.index(int(fields[1]))] # possible values are 1,2,50, 99
	return [0,0,0,0]

def len_check(sliced_channel):
	if len(sliced_channel) < LENGTH:
		while(sliced_channel != LENGTH):
			sliced_channel.append(0)
	return sliced_channel

def balance():
	classes = [0, 0, 0, 0] # [1, 2, 50 , 90]
	for _, one_hot in band_amps:
		classes[np.argmax(one_hot)] += 1
	min_class = min([classes[0], classes[-1]])
	classes = [0, 0, 0, 0] # [1, 2, 50 , 90]
	balanced = []
	for data in band_amps:
		if classes[np.argmax(data[1])] < min_class:
			balanced.append(data)
			classes[np.argmax(data[1])] += 1
	return balanced

LENGTH = 15000
ELECTRODES = 64
MAX_Hz = 60
PATH = "C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Matlab Files"
MDD_list = np.genfromtxt("C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Data_4_Import_REST.csv", delimiter=',', dtype=str)
band_amps = []

for filename in tqdm(os.listdir(PATH)):
	class_ = is_MDD(filename)
	if max(class_) == 0:
		continue
	file_path = f"{PATH}\{filename}"
	dl = data_handler(file_path)		# Add inheritance between data_handler & channel_bands_amp
	channel_data = dl.get_EEG()[:ELECTRODES]

	sliced_channels = []
	for channel in channel_data:
		sliced_channel = []
		for i in range(0,len(channel)-LENGTH,LENGTH):
			fft = fast_fourier_transform(channel[i:i+LENGTH])
			sliced_channel.append(fft.FFT(MAX_Hz)/LENGTH) # INCREASE AMOUNT OF GRANULAIRTY (scale of Hz)
		sliced_channels.append(sliced_channel) # [electrode][sample #][hz]

	for s in range(len(sliced_channels[0])):
		montage = []
		for e in range(len(sliced_channels)):
			montage.append(sliced_channels[e][s])
		band_amps.append([montage, class_])

np.random.shuffle(balance())
np.save("v1 - depression\\pre-processed_data\\EEG_balanced.npy", band_amps)
