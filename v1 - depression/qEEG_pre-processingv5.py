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

LENGTH = 15000
ELECTRODES = 64
MAX_Hz = 60
PATH = "C:\\Users\\akend\\Downloads\\sampleEEGdata.mat"

data = (loadmat(PATH)['EEG']['data'][0][0]) # [channles][trials][trials]

def pol2cart(t, r):
    t=round(t[0][0],4)
    r=r[0][0]

    x = (r*(1+np.tan(t)**2)**0.5)/(1+(np.tan(t)**2))
    y = ((r**2-x**2)**0.5)

    if round(np.arctan2(-y,x),4) == t:
        return (x,-y)
    elif round(np.arctan2(y,-x),4) == t:
        return (-x, y)
    elif round(np.arctan2(-y,-x),4) == t:
        return (-x, -y)
    return(x, y)


raw_data = loadmat("C:\\Users\\akend\\Downloads\\sampleEEGdata.mat")['EEG']['chanlocs'][0][0][0]

x, y = np.transpose(np.array([[pol2cart((np.pi/180)*i[1], i[2])] for i in (raw_data)]))
x = x[0]
y = y[0]

SCALE = 100

x = (abs(min(x))+np.array(x))
x =  (x/max(x))*SCALE
y = (abs(min(y))+np.array(y))
y = (y/max(y))*(SCALE)

z = []

for channel in data:
	fft = fast_fourier_transform(channel, 100)
	z.append(fft.get_bands()['Delta'])
xi = yi = np.arange(0,SCALE+1,1)
xi,yi = np.meshgrid(xi,yi)

rbf = interpolate.Rbf(x,y,z, function='cubic', smooth=0)
zi = rbf(xi,yi)

fig = plt.figure()
ax = fig.add_subplot(111)
contourf_ = plt.contourf(xi,yi,zi, cmap='jet')
plt.scatter(x,y,c='r',marker='+')
fig.colorbar(contourf_)
plt.show()

plt.imshow(zi)
plt.show()