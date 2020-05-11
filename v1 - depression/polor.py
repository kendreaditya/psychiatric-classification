from scipy.io import loadmat
import numpy as np

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

X, Y = np.transpose(np.array([[pol2cart((np.pi/180)*i[1], i[2])] for i in (raw_data)]))
X = X[0]
Y = Y[0]

x = loadmat("C:\\Users\\akend\\Downloads\\x.mat")['elocsX'][0]
y = loadmat("C:\\Users\\akend\\Downloads\\y.mat")['elocsY'][0]

for x1, x2, y1, y2 in zip(X, x, Y, y):
    if (round(x1,4)!=round(x2,4)) or (round(y1,4)!=round(y2,4)):
        print(x1, x2, y1, y2)

import matplotlib.pyplot as plt
plt.scatter(x,y,c='r',marker='+')
plt.scatter(X,Y,c='b',marker='_')
plt.show()