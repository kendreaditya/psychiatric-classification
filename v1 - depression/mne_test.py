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
    import numpy as np
    #sphere = _check_sphere(sphere)

    if bands is None:
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]

    if agg_fun is None:
        agg_fun = np.sum if normalize is True else np.mean

    if normalize is True:
        psds /= psds.sum(axis=-1)[..., None]
        assert np.allclose(psds.sum(axis=-1), 1.)

    n_axes = len(bands)
    if axes is not None:
        #_validate_if_list_of_axes(axes, n_axes)
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, n_axes, figsize=(2 * n_axes, 1.5))
        if n_axes == 1:
            axes = [axes]

    for ax, (fmin, fmax, title) in zip(axes, bands):
        freq_mask = (fmin < freqs) & (freqs < fmax)
        if freq_mask.sum() == 0:
            raise RuntimeError('No frequencies in band "%s" (%s, %s)'
                               % (title, fmin, fmax))
        data = agg_fun(psds[:, freq_mask], axis=1)
        if dB is True and normalize is False:
            data = 10 * np.log10(data)
            unit = 'dB'
        else:
            unit = 'power'

    fig.canvas.draw()
    #plt.show(show)
    return fig
