import pickle
import numpy as np

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        file = pickle.load(f)

    return file

def insert_zeros_between(x, j):
    '''x should be a 1D numpy array
       j should be an integer
    '''
    assert len(x.shape) == 1
    new_x = np.zeros(len(x) + (len(x)-1)*(j))
    new_x[::j+1] = x
    return new_x

def circ_conv(signal, ker):
    '''
        Reference: https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker)))