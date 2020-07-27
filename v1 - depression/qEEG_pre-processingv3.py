import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy import interpolate

from data_handler import data_handler
from fast_fourier_transform import  fast_fourier_transform
from channel_band_amps import channel_band_amps, channel_names, qEEG

def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return [rho, phi]

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
	if max(class_) == 0 or class_[-1]!=1:
		continue
	file_path = f"{PATH}\{filename}"
	dl = data_handler(file_path)		# Add inheritance between data_handler & channel_bands_amp
	channel_data = dl.get_EEG()[:ELECTRODES]
	channel_locations = dl.get_channel_locations()

	sliced_channels = []
	for channel in channel_data:
		fft = fast_fourier_transform(channel, MAX_Hz)
		#sliced_channels.append(fft.get_bands()['Alpha'])
		sliced_channels.append(channel[500])
	#break

    #print(sliced_channels)

	x,y = np.transpose(np.array([(x_temp,y_temp) for x_temp, y_temp in zip(channel_locations[0][:ELECTRODES], channel_locations[1][:ELECTRODES])]))
	z = sliced_channels

	xnew, ynew = np.mgrid[-85:85:85j, -85:85:85j]
	tck = interpolate.bisplrep(x, y, z, s=0)
	znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
	#spline = interpolate.RectBivariateSpline(x, y, z, s=0, kx=3, ky=3)
	#znew = spline(xnew,ynew)
	plt.figure()
	plt.pcolor(xnew,ynew,znew)	

	plt.scatter(x,y,c='r',marker='+')
	plt.title(str(class_))
	plt.colorbar()
	plt.show()
	break
