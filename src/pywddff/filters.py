from pywddff.utils import insert_zeros_between
from pywddff.datasets import get_scaling_dict, get_wavelet_dict
import numpy as np

# Dictionaries containing scaling and wavelet filters, respectively
scaling_dict = get_scaling_dict()
wavelet_dict = get_wavelet_dict()

# Level 1 scaling and wavelet filters

def scaling_filter(filter, modwt = False):
    """
    Level 1 scaling filter.

    Args:

        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        modwt (bool): If True, the level 1 scaling filter is divided by 2^(1/2).

    Returns:
        np.ndarray: A 1D numpy array.
    """
    if modwt:
        return scaling_dict[filter] / np.sqrt(2)
    elif not modwt:
        return scaling_dict[filter]
    else:
        raise TypeError

def wavelet_filter(filter, modwt = False):
    """
    Level 1 wavelet filter.

    Args:

        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        modwt (bool): If True, the level 1 wavelet filter is divided by 2^(1/2).

    Returns:
        np.ndarray: A 1D numpy array.
    """
    if modwt:
        return wavelet_dict[filter] / np.sqrt(2)
    elif not modwt:
        return wavelet_dict[filter]
    else:
        raise TypeError

# Level j scaling and wavelet filters

def equiv_scaling_filter(filter, j):
    """
    Level j equivalent scaling filter.

    Args:

        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        j (int): Decomposition level.

    Returns:
        np.ndarray: A 1D numpy array of length (2^j -1)*(L-1)+1, where L is the length of the level 1 
                    scaling/wavelet filter.
    """
    prev_filter = scaling_filter(filter)
    running_filter = prev_filter.copy()
    for i in range(j-1):
        next_filter = insert_zeros_between(prev_filter, 1)
        running_filter = np.convolve(next_filter, running_filter)
        prev_filter = next_filter
    return running_filter

def equiv_wavelet_filter(filter, j):
    """
    Level j equivalent wavelet filter.

    Args:

        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        j (int): Decomposition level.

    Returns:
        np.ndarray: A 1D numpy array of length (2^j -1)*(L-1)+1, where L is the length of the level 1 
                    scaling/wavelet filter.
    """
    prev_filter = scaling_filter(filter)
    running_filter = prev_filter.copy()
    for i in range(j-2):
        next_filter = insert_zeros_between(prev_filter, 1)
        running_filter = np.convolve(next_filter, running_filter)
        prev_filter = next_filter
    
    prev_filter = wavelet_filter(filter)
    next_filter = insert_zeros_between(prev_filter, (2**(j-1))-1)
    running_filter = np.convolve(next_filter, running_filter)
    
    return running_filter