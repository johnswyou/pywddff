from utils import load_pickle, insert_zeros_between
import numpy as np

# Dictionaries containing scaling and wavelet filters, respectively
scaling_dict = load_pickle('scaling_dict.pkl')
wavelet_dict = load_pickle('wavelet_dict.pkl')

# Level 1 scaling and wavelet filters

def scaling_filter(filter, modwt = False):
    if modwt:
        return scaling_dict[filter] / np.sqrt(2)
    elif not modwt:
        return scaling_dict[filter]
    else:
        raise TypeError

def wavelet_filter(filter, modwt = False):
    if modwt:
        return wavelet_dict[filter] / np.sqrt(2)
    elif not modwt:
        return wavelet_dict[filter]
    else:
        raise TypeError

# Level j scaling and wavelet filters

def equiv_scaling_filter(filter, j):
    '''filter is a string
       j is an integer greater than 0
       kwargs is to specify value of modwt (either modwt=True or modwt=False)
    '''
    prev_filter = scaling_filter(filter)
    running_filter = prev_filter.copy()
    for i in range(j-1):
        next_filter = insert_zeros_between(prev_filter, 1)
        running_filter = np.convolve(next_filter, running_filter)
        prev_filter = next_filter
    return running_filter

def equiv_wavelet_filter(filter, j, **kwargs):
    '''filter is a string
       j is an integer greater than 0
       kwargs is to specify value of modwt (either modwt=True or modwt=False)
    '''
    prev_filter = scaling_filter(filter, **kwargs)
    running_filter = prev_filter.copy()
    for i in range(j-2):
        next_filter = insert_zeros_between(prev_filter, 1)
        running_filter = np.convolve(next_filter, running_filter)
        prev_filter = next_filter
    
    prev_filter = wavelet_filter(filter, **kwargs)
    next_filter = insert_zeros_between(prev_filter, (2**(j-1))-1)
    running_filter = np.convolve(next_filter, running_filter)
    
    return running_filter