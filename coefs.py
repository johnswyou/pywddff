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
        num_bc = n_boundary_coefs(filter, j, max_L, max_J)
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
        num_bc = n_boundary_coefs(filter, j, max_L, max_J)
        wavelet_coef = wavelet_coef[num_bc:]

    return wavelet_coef

def n_boundary_coefs(filter, j, max_L=None, max_J=None):

    if max_L is None:
        g = scaling_filter(filter)
        L = g.shape[0]
        max_L = L

    if max_J is None:
        max_J = j

    num_bc = (2**max_J - 1) * (max_L - 1)

    return num_bc

def make_output_names(n_inputs, j):
    orig_input_names = ["X" + str(i) for i in range(1, n_inputs+1)]
    wavelet_names = ["W" + str(i) for i in range(1, j+1)]
    scaling_name = "V" + str(j)
    wavelet_names.append(scaling_name)
    return [i + "_" + j for i in orig_input_names for j in wavelet_names]
