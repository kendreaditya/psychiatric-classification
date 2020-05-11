from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

############################ Annotation3D #############################
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
#######################################################################

class data_handler():    
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = loadmat(file_path)
        self.raw_channel_info = self.raw_data['EEG']['chanlocs'][0][0][0]
    
    def get_channel_names_locations(self):
        names = []
        XYZ = []
        for channel, name in zip(self.raw_channel_info[:64], self.get_channel_names()[:64]):
            try:
                XYZ.append([channel[4][0][0]/1000, channel[5][0][0]/1000, channel[6][0][0]/1000])
                names.append(name)
            except:
                pass
        print(len(names), len(XYZ))
        return zip(names, XYZ)



    def get_EEG(self):
        return self.raw_data['EEG']['data'][0][0]

    def get_raw_data(self):
        return self.raw_data

    def get_channel_locations(self):
        X, Y, Z = [], [], []
        for channel in self.raw_channel_info:
            try:
                X.append(channel[4][0][0])
            except:
                X.append(0.01)
            
            try:
                Y.append(channel[5][0][0])
            except:
                Y.append(0.01)

            try:
                Z.append(channel[6][0][0])
            except:
                Z.append(0.01)
        return [X, Y, Z]

    def pol2cart(self, t, r):
        t=round(t,4)

        x = (r*(1+np.tan(t)**2)**0.5)/(1+(np.tan(t)**2))
        y = ((r**2-x**2)**0.5)

        if round(np.arctan2(-y,x),4) == t:
            return (x,-y)
        elif round(np.arctan2(y,-x),4) == t:
            return (-x, y)
        elif round(np.arctan2(-y,-x),4) == t:
            return (-x, -y)
        return(x, y)

    def get_polor_channel_locations(self, rotate=False, theta_index=2, radius_index=3):
        x,y = [], []
        removed_channels = []
        for channel in self.raw_channel_info:
            try:
                xy = self.pol2cart((np.pi/180)*channel[theta_index][0][0],channel[radius_index][0][0])
                x.append(xy[0])
                y.append(xy[1])
            except:
                removed_channels.append(channel[0][0])
        if rotate:
            return (y,x, removed_channels)
        return (x, y, removed_channels)

    def get_channel_names(self):
        return [info[0][0] for info in self.raw_channel_info][:64]

    def plot_channel_locations(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        channel_locations = self.get_channel_locations()
        channel_names = self.get_channel_names()

        ax.scatter(channel_locations[0], channel_locations[1], channel_locations[2])

        for name, xyz in zip(channel_names, zip(channel_locations[0], channel_locations[1], channel_locations[2])):
            annotate3D(ax, s=name, xyz=xyz, fontsize=5, xytext=(-3,3),
                textcoords='offset points', ha='right',va='bottom')

        plt.show()
        