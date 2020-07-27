import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import griddata

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
	channel_locations = dl.get_channel_locations()

	sliced_channels = []
	for channel in channel_data:
		fft = fast_fourier_transform(channel, MAX_Hz)
		sliced_channels.append(fft.get_bands()['Delta'])
	#break

    #print(sliced_channels)


	x = channel_locations[0][:ELECTRODES]
	y = channel_locations[1][:ELECTRODES]
	
	z = sliced_channels
	print(z)
	#print(len(x), len(y), len(z))

	xi = np.arange(min(x), max(x), .1)
	yi = np.arange(min(y), max(y), .1)
	xi,yi = np.meshgrid(xi,yi)
	zi = griddata((x,y),z,(xi,yi), method='cubic')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	contourf_ = plt.contourf(xi,yi,zi)
	plt.scatter(x,y)
	plt.title(str(class_))
	fig.colorbar(contourf_)
	plt.show()
	plt.imshow(zi)
	plt.show()
	break