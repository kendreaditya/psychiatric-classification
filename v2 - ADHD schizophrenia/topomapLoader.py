import numpy as np
import matplotlib.pyplot as plt

"""data = np.load('rawData.npy', allow_pickle=True)
plt.imshow(data[0].reshape((7,8)))
plt.show()"""

data = np.load('C:\OneDrive - Cumberland Valley School District\EEG ScienceFair\src\Zi1.npy', allow_pickle=True)

plt.imshow(data)
plt.colorbar()
plt.show()