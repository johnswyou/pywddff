import numpy as np
from filters import equiv_scaling_filter, equiv_wavelet_filter, scaling_filter
from utils import circ_conv

def scaling_coefs(x, filter, j, remove_bc = True, max_L = None, max_J = None):
    equivalent_scaling_filter = equiv_scaling_filter(filter, j) / (2**(j/2))
    if x.shape[0] < equivalent_scaling_filter.shape[0]:
        raise ValueError("x should be longer than the equivalent scaling filter")
    n_pads = x.shape[0] - equivalent_scaling_filter.shape[0]
    equivalent_scaling_filter = np.pad(equivalent_scaling_filter, (0, n_pads), mode='constant')

    scaling_coef = circ_conv(x, equivalent_scaling_filter)

    # Handling boundary coefficients
    if remove_bc:

        g = scaling_filter(filter)
        L = g.shape[0]

        if max_L is None:
            max_L = L

        if max_J is None:
            max_J = j

        num_bc = (2**max_J - 1) * (max_L - 1)
        scaling_coef = scaling_coef[num_bc:]

    return scaling_coef

def wavelet_coefs(x, filter, j, remove_bc = True, max_L = None, max_J = None):
    equivalent_wavelet_filter = equiv_wavelet_filter(filter, j) / (2**(j/2))
    if x.shape[0] < equivalent_wavelet_filter.shape[0]:
        raise ValueError("x should be longer than the equivalent wavelet filter")
    n_pads = x.shape[0] - equivalent_wavelet_filter.shape[0]
    equivalent_wavelet_filter = np.pad(equivalent_wavelet_filter, (0, n_pads), mode='constant')

    wavelet_coef = circ_conv(x, equivalent_wavelet_filter)

    # Handling boundary coefficients
    if remove_bc:

        g = scaling_filter(filter)
        L = g.shape[0]

        if max_L is None:
            max_L = L

        if max_J is None:
            max_J = j

        num_bc = (2**max_J - 1) * (max_L - 1)
        wavelet_coef = wavelet_coef[num_bc:]

    return wavelet_coef