import pickle
import numpy as np
import pandas as pd

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

def add_lags(x, n_lags, pandas_output = False):

    x = pd.Series(x)
    
    cols = [x]
    
    for i in range(1, n_lags+1):
        cols.append(x.shift(i))

    if pandas_output:
        return pd.concat(cols, axis=1).dropna()
    else:
        return pd.concat(cols, axis=1).dropna().to_numpy()

def make_lag_names(n_inputs, n_lags):
    orig_input_names = ["X" + str(i) for i in range(1, n_inputs+1)]
    lag_names = ["lag_" + str(i) for i in range(n_lags+1)]
    out = [i + "_" + j for i in orig_input_names for j in lag_names]
    out = [i.replace('_lag_0', '') for i in out]
    return out

def add_lagged_variables(X, y=None, n_lags=1, pandas_output=False):

    '''
    X is a 2D numpy array
    **kwargs are further arguments to be passed to modwt()
    '''
    
    assert len(X.shape) > 1
    assert X.shape[1] > 0
    assert X.shape[0] > X.shape[1]

    n_inputs = X.shape[1]

    out = np.apply_along_axis(add_lags, 0, X, n_lags, False)
    out = np.split(out, n_inputs, 2)
    out = [i.squeeze() for i in out]
    out = np.concatenate(out, axis=1)

    if pandas_output:
        out = pd.DataFrame(out, columns=make_lag_names(n_inputs, n_lags))

    if y is not None:
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        y = y[n_lags:]
        assert out.shape[0] == y.shape[0]
        if pandas_output:
            y = pd.Series(y, name="y")
        return out, y
    else:
        return out