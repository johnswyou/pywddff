import pickle
import numpy as np
import pandas as pd

def load_pickle(filepath):
    """
    Load a pickle file.

    Args:

        filepath (str): A string that indicates the path to the pickle file (including the pickle file itself).

    Returns:
        Object that was stored in filepath.
    """
    with open(filepath, 'rb') as f:
        file = pickle.load(f)

    return file

def insert_zeros_between(x, j):
    """
    Inserts a specified number of zeros between each element in a 1D numpy array.
    The first set of zeros are inserted between the first and second elements in x.
    No zeros are inserted after the last element in x.

    Args:

        x (np.ndarray): A 1D numpy array.
        j (int): Number of zeros to insert between elements of x.

    Returns:
        np.ndarray: A 1D numpy array.
    """
    assert len(x.shape) == 1
    new_x = np.zeros(len(x) + (len(x)-1)*(j))
    new_x[::j+1] = x
    return new_x

def circ_conv(signal, ker):
    """
    Perform circular convolution. Note that signal and ker must have same shape.
    Reference: https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution

    Args:

        signal (np.ndarray): A 1D numpy array.
        ker (np.ndarray): A 1D numpy array.

    Returns:
        np.ndarray: A 1D numpy array.
    """
    assert len(signal.shape) == len(ker.shape) == 1 # Both signal and ker are 1D numpy arrays.
    assert signal.shape[0] == ker.shape[0] # Both signal and ker have the same shape.

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

def make_lag_names_from_list(orig_input_names, n_lags):
    lag_names = ["lag_" + str(i) for i in range(n_lags+1)]
    out = [i + "_" + j for i in orig_input_names for j in lag_names]
    out = [i.replace('_lag_0', '') for i in out]
    return out

def add_lagged_variables(X, y=None, n_lags=1):

    '''
    X is a 2D numpy array
    '''
    
    assert len(X.shape) > 1
    assert X.shape[1] > 0
    assert X.shape[0] > X.shape[1]

    # If X is a pandas data frame
    pandas_output = isinstance(X, pd.DataFrame)

    if pandas_output:
        original_X_colnames = list(X)
        X = X.to_numpy()

    n_inputs = X.shape[1]

    out = np.apply_along_axis(add_lags, 0, X, n_lags, False)
    out = np.split(out, n_inputs, 2)
    out = [i.squeeze() for i in out]
    out = np.concatenate(out, axis=1)

    # If X was given as a pandas data frame by the user
    if pandas_output:
        out = pd.DataFrame(out, columns=make_lag_names_from_list(original_X_colnames, n_lags))

    if y is not None:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        y = y.squeeze()

        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1

        y = y[n_lags:]

        assert out.shape[0] == y.shape[0]
        
        # If X was given as a pandas data frame by the user
        if pandas_output:
            y = pd.Series(y, name="y")

        return out, y
    else:
        return out

def test_size(X, test_frac=0.2):

    # Check that the fraction is valid.
    assert test_frac < 1, "Invalid split fraction."

    # Get test set size
    test_size = int(test_frac * X.shape[0])

    return test_size

def val_test_sizes(X, val_frac=0.1, test_frac=0.2):

    # Check that the fractions are valid.
    assert test_frac + val_frac < 1, "Invalid split fractions."

    # Get validation and test set sizes
    val_size = int(val_frac * X.shape[0])
    test_size = int(test_frac * X.shape[0])

    return val_size, test_size

def absolute_split_2(X, y, ntest):

    assert len(X.shape) == 2
    assert len(y.shape) == 1

    assert ntest > 0

    nrows = X.shape[0]

    assert nrows > ntest

    # Test set
    # ---------

    X_test = X[(nrows-ntest):, :]
    y_test = y[(nrows-ntest):]

    assert X_test.shape[0] == ntest

    # Training set
    # -------------

    # Remove test set observations from X and y
    X = X[:(nrows-ntest), :]
    y = y[:(nrows-ntest)]

    assert X.shape[0] == (nrows-ntest)

    return X, X_test, y, y_test

def absolute_split_3(X, y, nval, ntest):

    assert len(X.shape) == 2
    assert len(y.shape) == 1

    assert nval > 0
    assert ntest > 0

    nrows = X.shape[0]

    assert nrows > nval + ntest

    # Test set
    # ---------

    X_test = X[(nrows-ntest):, :]
    y_test = y[(nrows-ntest):]

    assert X_test.shape[0] == ntest

    # Validation set
    # ---------------

    # Remove test set observations from X and y
    X = X[:(nrows-ntest), :]
    y = y[:(nrows-ntest)]

    assert X.shape[0] == (nrows-ntest)

    # nrows is now the size of training set + size of validation set
    nrows = X.shape[0]
    X_val = X[(nrows-nval):, :]
    y_val = y[(nrows-nval):, :]

    assert X_val.shape[0] == nval

    # Training set
    # -------------

    X = X[:(nrows-nval), :]
    y = y[:(nrows-nval), :]

    return X, X_val, X_test, y, y_val, y_test

def prep_forecast_data(X, y, h, auto_regress_y = False):
    """
    Prepare an input feature set X and target y for forecasting by specifying the forecast horizon h.
    The output of this function is a tuple with input features and target such that each row of input features maps to
    a future observation of the target. This setup allows cross validation to be used when evaluating machine learning models.

    Args:

        X (np.ndarray or pd.DataFrame): A 2D numpy array or pandas data frame.
        y (np.ndarray, pd.Series, or pd.DataFrame): A 1D numpy array, pandas series or pandas data frame.
        h (int): Forecast horizon.
        auto_regress_y (bool): Whether the target should be included as an auto-regressive feature (to exploit autocorrelations present in the target variable).

    Returns:
        if auto_regress = False (the default):
        
        tuple: First element is a 2D numpy array with X.shape[1] columns. If X was given as a pandas data frame, the output will be a pandas data frame.
               The number of rows will be h less than X.shape[0] of the originally provided X.

               Second element is a 1D array corresponding to the target y provided by the user. 
               If X was given as a pandas data frame, the output will be a pandas series with name "y". 
               The number of values will be h less than y.shape[0] of the originally provided y.

        if auto_regress = True:
        
        tuple: First element is a 2D numpy array with X.shape[1]+1 columns. 
               The first column will contain the auto-regressive target feature (essentially a lagged version of the target).
               If X was given as a pandas data frame, the output will be a pandas data frame.
               The number of rows will be h less than X.shape[0] of the originally provided X.

               Second element is a 1D array corresponding to the target y provided by the user. 
               If X was given as a pandas data frame, the output will be a pandas series with name "y". 
               The number of values will be h less than y.shape[0] of the originally provided y. 
    """
    assert len(X.shape) == 2

    # If X is a pandas data frame
    pandas_output = isinstance(X, pd.DataFrame)

    if pandas_output:
        original_X_colnames = list(X)
        X = X.to_numpy()

    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.to_numpy()

    y = y.squeeze()

    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1

    assert X.shape[0] == y.shape[0]

    # Add h padding rows on top of X
    X = np.pad(X, ((h, 0), (0, 0)), mode='constant', constant_values=np.nan)

    if auto_regress_y:
        y1 = np.pad(y, (h, 0), mode='constant', constant_values=np.nan)
        X = np.hstack((y1[:, None], X))
        if pandas_output:
            if "y_lagged" in original_X_colnames:
                raise ValueError('The name y_lagged cannot exist in your input feature data frame X.')
            original_X_colnames = ["y_lagged"] + original_X_colnames

    # Add h padding elements to the end of y
    y = np.pad(y, (0, h), mode='constant', constant_values=np.nan)

    yX = np.hstack((y[:, None], X))
    yX = yX[~np.isnan(yX).any(axis=1)]

    if pandas_output:
        return pd.DataFrame(yX[:, 1:], columns = original_X_colnames), pd.Series(yX[:, 0], name="y")
    else:
        return yX[:, 1:], yX[:, 0]
