import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.signal import periodogram

from data_handler import data_handler
from fast_fourier_transform import  fast_fourier_transform
from channel_band_amps import channel_band_amps, channel_names, qEEG

############################
def plot_psds_topomap(
        psds, freqs, pos, agg_fun=None, vmin=None, vmax=None, bands=None,
        cmap=None, dB=True, normalize=False, cbar_fmt='%0.3f', outlines='head',
        axes=None, show=True, sphere=None):
    """Plot spatial maps of PSDs.
    Parameters
    ----------
    psds : np.ndarray of float, shape (n_channels, n_freqs)
        Power spectral densities
    freqs : np.ndarray of float, shape (n_freqs)
        Frequencies used to compute psds.
    pos : numpy.ndarray of float, shape (n_sensors, 2)
        The positions of the sensors.
    agg_fun : callable
        The function used to aggregate over frequencies.
        Defaults to np.sum. if normalize is True, else np.mean.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None np.min(data) is used. If callable, the output equals
        vmin(data).
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    bands : list of tuple | None
        The lower and upper frequency and the name for that band. If None,
        (default) expands to:
            bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                     (12, 30, 'Beta'), (30, 45, 'Gamma')]
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None (default), 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True).
    dB : bool
        If True, transform data to decibels (with ``10 * np.log10(data)``)
        following the application of `agg_fun`. Only valid if normalize is
        False.
    normalize : bool
        If True, each band will be divided by the total power. Defaults to
        False.
    cbar_fmt : str
        The colorbar format. Defaults to '%%0.3f'.
    %(topomap_outlines)s
    axes : list of axes | None
        List of axes to plot consecutive topographies to. If None the axes
        will be created automatically. Defaults to None.
    show : bool
        Show figure if True.
    %(topomap_sphere)s
    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure distributing one image per channel across sensor topography.
    """
    import matplotlib.pyplot as plt

    if bands is None:
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]

    if agg_fun is None:
        agg_fun = np.sum if normalize is True else np.mean

    if normalize is True:
        psds /= psds.sum(axis=-1)[..., None]
        assert np.allclose(psds.sum(axis=-1), 1.
    
    n_axes = len(bands)
    if axes is not None:
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, n_axes, figsize=(2 * n_axes, 1.5))
        if n_axes == 1:
            axes = [axes]

    for ax, (fmin, fmax, title) in zip(axes, bands):
        try:
            freq_mask = (fmin < freqs) & (freqs < fmax)
            print(freq_mask)
            if freq_mask.sum() == 0:
                raise RuntimeError('No frequencies in band "%s" (%s, %s)'% (title, fmin, fmax))
            data = agg_fun(psds[:, 1], axis=1)
            #print(psds)
            #data = agg_fun(psds, axis=1)
            if dB is True and normalize is False:
                data = 10 * np.log10(data)
                unit = 'dB'
            else:
                unit = 'power'
        except:
            print('fail')

    fig.canvas.draw()
    plt_show(show)
    return fig

def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.
    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)
############################

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
	PSDS = []
	sliced_channels = []
	POS = []
	for channel, name, pos in zip(channel_data, dl.get_channel_names(), channel_locations):
		fft = fast_fourier_transform(channel)
		FFT = fft.FFT(MAX_Hz)
		PSDS.append([periodogram(FFT, 500)[0]])
		POS.append([pos[0], pos[1]])

	plot_psds_topomap(np.array(PSDS), 500, np.array(POS))
	break