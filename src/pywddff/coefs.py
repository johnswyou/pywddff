import numpy as np
import pandas as pd
from pywddff.filters import equiv_scaling_filter, equiv_wavelet_filter, scaling_filter
from pywddff.utils import circ_conv

def scaling_coefs(x, filter, j, remove_bc = True, **kwargs):
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT) scaling coefficients.

    Args:

        x (np.ndarray): A 1D numpy array.
        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        j (int): Decomposition level.
        remove_bc (bool): Whether boundary coefficients should be removed. If True, boundary coefficients are
                          removed. If False, boundary coefficients are not removed.
        **kwargs: Used to specify max_L, max_J for cutting off boundary coefficients.
        max_L (int): This argument is used only when remove_bc = True. When max_L = None (and remove_bc = True), max_L is set equal to the length of the chosen
                     filter, L. When removing boundary coefficients, in the case that a user wants to use a max_L in ((2^max_J)-1)*(max_L - 1) 
                     that does not equal L, the user can specify a value for max_L that is greater than L. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.
        max_J (int): This argument is used only when remove_bc = True. When max_J = None (and remove_bc = True), max_J is set equal to J.
                     When removing boundary coefficients, in the case that a user wants to use a max_J in ((2^max_J)-1)*(max_L - 1) that does not equal J, 
                     the user can specify a value for max_J that is greater than J. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.

    Returns:
        np.ndarray: A 1D numpy array.
    """
    equivalent_scaling_filter = equiv_scaling_filter(filter, j) / (2**(j/2))
    if x.shape[0] < equivalent_scaling_filter.shape[0]:
        raise ValueError("x should be longer than the equivalent scaling filter")
    n_pads = x.shape[0] - equivalent_scaling_filter.shape[0]
    equivalent_scaling_filter = np.pad(equivalent_scaling_filter, (0, n_pads), mode='constant')

    scaling_coef = circ_conv(x, equivalent_scaling_filter)

    # Handling boundary coefficients
    if remove_bc:
        num_bc = n_boundary_coefs(filter, j, **kwargs)
        scaling_coef = scaling_coef[num_bc:]

    return scaling_coef

def wavelet_coefs(x, filter, j, remove_bc = True, **kwargs):
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT) wavelet coefficients.

    Args:

        x (np.ndarray): A 1D numpy array.
        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        j (int): Decomposition level.
        remove_bc (bool): Whether boundary coefficients should be removed. If True, boundary coefficients are
                          removed. If False, boundary coefficients are not removed.
        **kwargs: Used to specify max_L, max_J for cutting off boundary coefficients.
        max_L (int): This argument is used only when remove_bc = True. When max_L = None (and remove_bc = True), max_L is set equal to the length of the chosen
                     filter, L. When removing boundary coefficients, in the case that a user wants to use a max_L in ((2^max_J)-1)*(max_L - 1) 
                     that does not equal L, the user can specify a value for max_L that is greater than L. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.
        max_J (int): This argument is used only when remove_bc = True. When max_J = None (and remove_bc = True), max_J is set equal to J.
                     When removing boundary coefficients, in the case that a user wants to use a max_J in ((2^max_J)-1)*(max_L - 1) that does not equal J, 
                     the user can specify a value for max_J that is greater than J. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.

    Returns:
        np.ndarray: A 1D numpy array.
    """
    equivalent_wavelet_filter = equiv_wavelet_filter(filter, j) / (2**(j/2))
    if x.shape[0] < equivalent_wavelet_filter.shape[0]:
        raise ValueError("x should be longer than the equivalent wavelet filter")
    n_pads = x.shape[0] - equivalent_wavelet_filter.shape[0]
    equivalent_wavelet_filter = np.pad(equivalent_wavelet_filter, (0, n_pads), mode='constant')

    wavelet_coef = circ_conv(x, equivalent_wavelet_filter)

    # Handling boundary coefficients
    if remove_bc:
        num_bc = n_boundary_coefs(filter, j, **kwargs)
        wavelet_coef = wavelet_coef[num_bc:]

    return wavelet_coef

def n_boundary_coefs(filter, j, max_L=None, max_J=None):
    """
    Compute the number of boundary influenced MODWT/A Trous DWT wavelet/scaling coefficients.
    This value is one less than the length of the equivalent wavelet/scaling filter for level j.
    Note that the length of the equivalent wavelet/scaling filter for level j is (2^j - 1)*(L - 1) + 1.
    The number of boundary influenced wavelet/scaling coefficients is obtained from the fact that we circularly 
    convolve the equivalent wavelet/scaling filter for level j about a time series x, which means that we 
    require the last (2^j - 1)*(L - 1) values from the time series x to compute the first (2^j - 1)*(L - 1) wavelet/scaling coefficients.
    L is the length of the level 1 wavelet/scaling filter.

    Args:

        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        j (int): Decomposition level.
        max_L (int): When max_L = None, max_L is set equal to the length of the chosen
                     filter, L. When removing boundary coefficients, in the case that a user wants to use a max_L in ((2^max_J)-1)*(max_L - 1) 
                     that does not equal L, the user can specify a value for max_L that is greater than L. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.
        max_J (int): When max_J = None, max_J is set equal to J.
                     When removing boundary coefficients, in the case that a user wants to use a max_J in ((2^max_J)-1)*(max_L - 1) that does not equal J, 
                     the user can specify a value for max_J that is greater than J. Doing this is useful when doing comparison studies
                     to compare different filter and decomposition level (J) combinations while controlling for the number boundary coefficients
                     that are removed across the different configurations. It is unlikely that this argument will be needed by most users.
    
    Returns:
        int: Number of boundary coefficients to be removed.
    """
    if max_L is None:
        g = scaling_filter(filter)
        L = g.shape[0]
        max_L = L

    if max_J is None:
        max_J = j

    assert max_L >= L
    assert max_J >= j

    num_bc = (2**max_J - 1) * (max_L - 1)

    return num_bc

def make_output_names(n_inputs, j):
    """
    Helper function for multi_stationary_dwt to make column names in the
    case that pandas_output = True.

    Args:

        n_inputs (int): Number of input features in original input feature set.
        j (int): Decomposition level.

    Returns:
        list: A list containing column names for the post wavelet transform coefficients matrix.
              It is a list of strings.
    """
    orig_input_names = ["X" + str(i) for i in range(1, n_inputs+1)]
    wavelet_names = ["W" + str(i) for i in range(1, j+1)]
    scaling_name = "V" + str(j)
    wavelet_names.append(scaling_name)
    return [i + "_" + j for i in orig_input_names for j in wavelet_names]

def make_output_names_from_df(X, j):
    """
    Helper function for multi_stationary_dwt to make column names.

    Args:

        X (pd.DataFrame): Original input feature set.
        j (int): Decomposition level.

    Returns:
        list: A list containing column names for the post wavelet transform coefficients matrix.
              It is a list of strings.
    """
    assert isinstance(X, pd.DataFrame)

    orig_input_names = list(X)
    wavelet_names = ["W" + str(i) for i in range(1, j+1)]
    scaling_name = "V" + str(j)
    wavelet_names.append(scaling_name)
    return [i + "_" + j for i in orig_input_names for j in wavelet_names]

def make_output_names_from_list(orig_input_names, j):
    """
    Helper function for multi_stationary_dwt to make column names.

    Args:

        orig_input_names (list): Original input feature set.
        j (int): Decomposition level.

    Returns:
        list: A list containing column names for the post wavelet transform coefficients matrix.
              It is a list of strings.
    """
    assert isinstance(orig_input_names, list)
    
    wavelet_names = ["W" + str(i) for i in range(1, j+1)]
    scaling_name = "V" + str(j)
    wavelet_names.append(scaling_name)
    return [i + "_" + j for i in orig_input_names for j in wavelet_names]