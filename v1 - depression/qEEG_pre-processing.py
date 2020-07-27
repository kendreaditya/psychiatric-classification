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

LENGTH = 5000
ELECTRODES = 64
MAX_Hz = 60
PATH = "C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Matlab Files"
MDD_list = np.genfromtxt("C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\database\depression\Data_4_Import_REST.csv", delimiter=',', dtype=str)
band_amps = [[], []]

def is_MDD(filename):
	val = [1,99]
	subject = filename[:3]
	for fields in MDD_list:
		if subject == fields[0] and fields[1]!='50' and fields[1]!='2':
			return np.eye(len(val))[val.index(int(fields[1]))] # possible values are 1,2,50, 99
	return [0,0]


for filename in tqdm(os.listdir(PATH)):
    binary = is_MDD(filename)
    file_path = f"{PATH}\{filename}"
    dl = data_handler(file_path)		# Add inheritance between data_handler & channel_bands_amp
    channel_data = dl.get_EEG()[:ELECTRODES]
    x, y, bad_channels = dl.get_polor_channel_locations(rotate=True)

    SCALE = 63
    x = (abs(min(x))+np.array(x))
    x =  (x/max(x))*SCALE
    y = (abs(min(y))+np.array(y))
    y = (y/max(y))*(SCALE)
    
    split_z = []
    for channel in channel_data:
        channel = channel[50:-50]
        temp_split = []
        for i in range(0, len(channel), LENGTH):
            fft = fast_fourier_transform(channel[i:i+LENGTH], 100)
            temp_split.append(fft.get_bands()['Theta']/LENGTH)
        split_z.append(temp_split)
    
    #split_z = (np.transpose(split_z)-np.amin(split_z)) / (np.amax(split_z)-np.amin(split_z))
    split_z = np.transpose(split_z)
    for z in split_z:
        xi = yi = np.arange(0,SCALE+1,1)
        xi,yi = np.meshgrid(xi,yi)


        rbf = interpolate.Rbf(x,y,z, function='cubic', smooth=0)
        zi = rbf(xi,yi)
    
        band_amps[np.argmax(binary)].append([zi, binary])

'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    contourf_ = plt.contourf(xi,yi,zi, cmap='jet')
    plt.scatter(x,y,c='r',marker='+')
    fig.colorbar(contourf_)
    plt.show()

    plt.imshow(zi)
    plt.show()
'''

def balance(data):
    balanced_data = []
    least = min([len(i) for i in band_amps])
    for class_ in data:
        for obj in class_[:least]:
            balanced_data.append(obj)
    return balanced_data

band_amps = balance(band_amps)
print(len(band_amps))
np.random.shuffle(band_amps)
np.save("v1 - depression\\pre-processed_data\\qEEGs_balanced.npy", band_amps)