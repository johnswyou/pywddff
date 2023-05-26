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

    '''
    Creates a DataFrame (or a NumPy array) where each column is a lagged version of the input series.

    Parameters
    ----------
    x : array-like
        Input sequence of data points.
    
    n_lags : int
        Number of lags to include in the output.
    
    pandas_output : bool, optional
        If True, the output will be a pandas DataFrame. 
        If False, the output will be a NumPy array. 
        Defaults to False.

    Returns
    -------
    output : pandas.DataFrame or numpy.ndarray
        DataFrame (or NumPy array) with original series and its lagged versions. 
        Each column corresponds to a lag (from 0 to n_lags). 
        The output excludes rows where lagged data is not available due to shifting (NA values).
    
    Example
    -------
    >>> add_lags([1, 2, 3, 4, 5], 2, True)
       0  1  2
    2  3  2  1
    3  4  3  2
    4  5  4  3
    '''

    x = pd.Series(x)
    
    cols = [x]
    
    for i in range(1, n_lags+1):
        cols.append(x.shift(i))

    if pandas_output:
        return pd.concat(cols, axis=1).dropna()
    else:
        return pd.concat(cols, axis=1).dropna().to_numpy()

def make_lag_names(n_inputs, n_lags):
    """
    Creates a list of string names for original and lagged inputs.

    Parameters
    ----------
    n_inputs : int
        Number of original input variables.

    n_lags : int
        Number of lags for each input variable.

    Returns
    -------
    out : list of str
        List of names for the original and lagged input variables. Each original input variable is named as 'Xn',
        where n is the input number (1-indexed). Each lagged version of an input variable is named as 'Xn_lag_m',
        where n is the input number (1-indexed) and m is the lag number. The lag number for the original (unlagged)
        variables is dropped, so they are named just 'Xn'.

    Example
    -------
    >>> make_lag_names(2, 3)
    ['X1', 'X1_lag_1', 'X1_lag_2', 'X1_lag_3', 'X2', 'X2_lag_1', 'X2_lag_2', 'X2_lag_3']
    """
    orig_input_names = ["X" + str(i) for i in range(1, n_inputs+1)]
    lag_names = ["lag_" + str(i) for i in range(n_lags+1)]
    out = [i + "_" + j for i in orig_input_names for j in lag_names]
    out = [i.replace('_lag_0', '') for i in out]
    return out

def make_lag_names_from_list(orig_input_names, n_lags):
    """
    Creates a list of string names for original and lagged inputs, based on the original input names provided.

    Parameters
    ----------
    orig_input_names : list of str
        List of original input variable names.

    n_lags : int
        Number of lags for each input variable.

    Returns
    -------
    out : list of str
        List of names for the original and lagged input variables. Each original input variable name is appended with 
        '_lag_m', where m is the lag number. The lag number for the original (unlagged) variables is dropped.

    Example
    -------
    >>> make_lag_names_from_list(['temp', 'humidity'], 3)
    ['temp', 'temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'humidity', 'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3']
    """
    lag_names = ["lag_" + str(i) for i in range(n_lags+1)]
    out = [i + "_" + j for i in orig_input_names for j in lag_names]
    out = [i.replace('_lag_0', '') for i in out]
    return out

def add_lagged_variables(X, y=None, n_lags=1):

    """
    Add lagged variables to a given input dataset X, and optionally adjust the target variable y to match the new structure.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Input dataset with shape (n_samples, n_features). Each feature is transformed to include its lags.
    
    y : numpy.ndarray or pandas.Series or pandas.DataFrame, optional
        Target variable with shape (n_samples,). If provided, it is adjusted to match the new structure of X.
        The first n_lags samples are dropped to match the size of X after adding the lagged variables.
        Defaults to None, in which case only X is processed and returned.
    
    n_lags : int, optional
        Number of lags to add for each feature in X. Defaults to 1.

    Returns
    -------
    out : numpy.ndarray or pandas.DataFrame
        Transformed input dataset with added lagged variables. 
        If X was a pandas DataFrame, out is also a pandas DataFrame, with column names adjusted to reflect the lags.
        If X was a numpy array, out is also a numpy array.
    
    y : numpy.ndarray or pandas.Series, optional
        Adjusted target variable. Only returned if y was provided as input.
        If y was a pandas Series or DataFrame, the output y is a pandas Series.
        If y was a numpy array, the output y is also a numpy array.

    Notes
    -----
    This function requires the add_lags and make_lag_names_from_list functions to work.
    
    Raises
    ------
    AssertionError
        If the shape of X is not (n_samples, n_features) with n_samples > n_features.
        If y is provided and its adjusted shape does not match the number of samples in the transformed X.

    Example
    -------
    >>> X = pd.DataFrame({'temp': [1, 2, 3, 4], 'humidity': [30, 40, 50, 60]})
    >>> y = pd.Series([0, 1, 0, 1])
    >>> add_lagged_variables(X, y, 2)
    (   temp  temp_lag_1  temp_lag_2  humidity  humidity_lag_1  humidity_lag_2
    0   3.0         2.0         1.0      50.0            40.0            30.0
    1   4.0         3.0         2.0      60.0            50.0            40.0, 0    0
    1    1
    Name: y, dtype: int64)
    """
    
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
    """
    Determines the size of the test set based on the provided fraction.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset with shape (n_samples, n_features).
    
    test_frac : float, optional
        Fraction of the total samples to be used for the test set. 
        Default is 0.2 (20% of total samples).

    Returns
    -------
    test_size : int
        Number of samples in the test set.

    Raises
    ------
    AssertionError
        If `test_frac` is greater or equal to 1, raising a ValueError.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> test_size(X, test_frac=0.3)
    1
    """
    # Check that the fraction is valid.
    assert test_frac < 1, "Invalid split fraction."

    # Get test set size
    test_size = int(test_frac * X.shape[0])

    return test_size

def val_test_sizes(X, val_frac=0.1, test_frac=0.2):
    """
    Determines the size of the validation and test sets based on the provided fractions.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset with shape (n_samples, n_features).
    
    val_frac : float, optional
        Fraction of the total samples to be used for the validation set. 
        Default is 0.1 (10% of total samples).
    
    test_frac : float, optional
        Fraction of the total samples to be used for the test set. 
        Default is 0.2 (20% of total samples).

    Returns
    -------
    val_size : int
        Number of samples in the validation set.
    
    test_size : int
        Number of samples in the test set.

    Raises
    ------
    AssertionError
        If the sum of `test_frac` and `val_frac` is greater or equal to 1, raising a ValueError.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> val_test_sizes(X, val_frac=0.2, test_frac=0.3)
    (1, 1)
    """
    # Check that the fractions are valid.
    assert test_frac + val_frac < 1, "Invalid split fractions."

    # Get validation and test set sizes
    val_size = int(val_frac * X.shape[0])
    test_size = int(test_frac * X.shape[0])

    return val_size, test_size

def absolute_split_2(X, y, ntest):
    """
    Splits the input dataset (X, y) into training and test sets based on an absolute number.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset with shape (n_samples, n_features).
    
    y : numpy.ndarray
        Target variable with shape (n_samples,).
    
    ntest : int
        Number of samples to include in the test set.

    Returns
    -------
    X : numpy.ndarray
        Training input dataset.
    
    X_test : numpy.ndarray
        Test input dataset.

    y : numpy.ndarray
        Training target variable.
    
    y_test : numpy.ndarray
        Test target variable.

    Raises
    ------
    AssertionError
        If the input dimensions do not match the requirements.
        If ntest is not a positive integer.
        If the total number of samples is less than ntest.
        If the final training or test set do not match the expected sizes.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 0, 1, 0])
    >>> absolute_split_2(X, y, 1)
    (array([[1, 2], [3, 4], [5, 6]]), 
    array([[7, 8]]), 
    array([1, 0, 1]), 
    array([0]))
    """
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
    """
    Splits the input dataset (X, y) into training, validation, and test sets based on absolute numbers.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset with shape (n_samples, n_features).
    
    y : numpy.ndarray
        Target variable with shape (n_samples,).
    
    nval : int
        Number of samples to include in the validation set.
    
    ntest : int
        Number of samples to include in the test set.

    Returns
    -------
    X : numpy.ndarray
        Training input dataset.
    
    X_val : numpy.ndarray
        Validation input dataset.
    
    X_test : numpy.ndarray
        Test input dataset.

    y : numpy.ndarray
        Training target variable.
    
    y_val : numpy.ndarray
        Validation target variable.

    y_test : numpy.ndarray
        Test target variable.

    Raises
    ------
    AssertionError
        If the input dimensions do not match the requirements.
        If nval or ntest are not positive integers.
        If the total number of samples is less than nval + ntest.
        If the final training, validation, or test sets do not match the expected sizes.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> y = np.array([1, 0, 1, 0, 1])
    >>> absolute_split_3(X, y, 2, 1)
    (array([[1, 2],
           [3, 4]]), array([[5, 6],
           [7, 8]]), array([[ 9, 10]]), array([1, 0]), array([1, 0]), array([1]))
    """
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
    y_val = y[(nrows-nval):]

    assert X_val.shape[0] == nval

    # Training set
    # -------------

    X = X[:(nrows-nval), :]
    y = y[:(nrows-nval)]

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
        tuple:
        if auto_regress = False (the default):
               First element is a 2D numpy array with X.shape[1] columns. If X was given as a pandas data frame, the output will be a pandas data frame.
               The number of rows will be h less than X.shape[0] of the originally provided X.

               Second element is a 1D array corresponding to the target y provided by the user. 
               If X was given as a pandas data frame, the output will be a pandas series with name "y". 
               The number of values will be h less than y.shape[0] of the originally provided y.

        if auto_regress = True:
               First element is a 2D numpy array with X.shape[1]+1 columns. 
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
