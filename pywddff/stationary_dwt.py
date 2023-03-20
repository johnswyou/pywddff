# import pywt # OLD
from filters import wavelet_filter, scaling_filter
from coefs import scaling_coefs, n_boundary_coefs, make_output_names
import numpy as np
import pandas as pd
import math

def modwt(x, filter, J, remove_bc = True, **kwargs):
    """
    Perform Maximal Overlap Discrete Wavelet Transform (MODWT)
    on a 1D numpy array x.

    Args:
        x (np.ndarray): A 1D numpy array.
        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        J (int): Decomposition level.
        remove_bc (bool): If True, the first ((2^max_J)-1)*(max_L - 1) rows of the output are removed.
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
        np.ndarray: A 2D array with (J+1) columns, the first J columns being the J wavelet coefficients and the last
                    column being the level J scaling coefficient.
    """

    # -----------------
    # Input validation
    # -----------------

    if not isinstance(x, np.ndarray):
        raise TypeError('x should be a 1D Numpy array')
    
    assert len(x.shape) == 1

    # Get MODWT wavelet and scaling filters (OLD)
    # wavelet = pywt.Wavelet(filter)
    # h = wavelet.dec_hi # wavelet filter
    # g = wavelet.dec_lo # scaling filter
    # h_t = np.array(h) / np.sqrt(2) # MODWT re-scaled h
    # g_t = np.array(g) / np.sqrt(2) # MODWT re-scaled g

    # -------------------------------------
    # Get MODWT wavelet and scaling filters
    # -------------------------------------

    h_t = wavelet_filter(filter, modwt=True)
    g_t = scaling_filter(filter, modwt=True)

    # Sanity check
    assert h_t.shape[0] == g_t.shape[0]

    L = h_t.shape[0]
    N = x.shape[0]

    # ----------------------
    # More input validation
    # ----------------------

    if J >= math.log((N/(L-1))+1, 2):
        raise ValueError('J must be less than base 2 logarithm of (N/(L-1)) + 1')

    # -------------------------------------------------
    # Computing MODWT wavelet and scaling coefficients
    # -------------------------------------------------

    # Reference: Page 177, Wavelet Methods for Time Series Analysis, 
    #            Percival & Walden (Comments and Extensions to Section 5.5)

    W = np.empty((N, J))
    V_last = x.copy() # shape (N,)
    V = np.empty(N)   # shape (N,)

    for j in range(1, J+1):
        
        for t in range(N):
            k = t
            W[t, j-1] = h_t[0]*V_last[k]
            V[t] = g_t[0]*V_last[k]

            for n in range(1, L):
                k -= 2**(j-1)
                if k < 0:
                    k = k % N
                W[t, j-1] += h_t[n]*V_last[k]
                V[t] += g_t[n]*V_last[k]

        V_last = V.copy()

    WV_J = np.column_stack([W, V])

    # ------------------------------
    # Handling boundary coefficients
    # ------------------------------

    if remove_bc:
        num_bc = n_boundary_coefs(filter, J, **kwargs)
        WV_J = WV_J[num_bc:, :]

    return WV_J

def atrousdwt(x, filter, J, remove_bc = True, **kwargs):
    """
    Perform A Trous Discrete Wavelet Transform
    on a 1D numpy array x.

    Args:
        x (np.ndarray): A 1D numpy array.
        filter (str): A string indicating the desired filter. There are 128 options, see the README on
                      this package's github page to see the list of filters available.
        J (int): Decomposition level.
        remove_bc (bool): If True, the first ((2^max_J)-1)*(max_L - 1) rows of the output are removed.
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
        np.ndarray: A 2D array with (J+1) columns, the first J columns being the J wavelet coefficients and the last
                    column being the level J scaling coefficient.
    """

    # ----------------
    # Input validation
    # ----------------

    if not isinstance(x, np.ndarray):
        raise TypeError('x should be a 1D Numpy array')
    
    assert len(x.shape) == 1

    h_t = wavelet_filter(filter, modwt=True)
    g_t = scaling_filter(filter, modwt=True)

    # Sanity check
    assert h_t.shape[0] == g_t.shape[0]

    L = h_t.shape[0]
    N = x.shape[0]

    if J >= math.log((N/(L-1))+1, 2):
        raise ValueError('J must be less than base 2 logarithm of (N/(L-1)) + 1')
    
    # Remove variables that won't be used
    del h_t
    del g_t
    del L
    del N

    # -----------------------------------------
    # Compute wavelet and scaling coefficients
    # -----------------------------------------

    # step 1. produce J scaling coefficients
    scaling_coefficients = [scaling_coefs(x, filter, j, False) for j in range(1, J+1)]
    scaling_coefficients = np.column_stack(scaling_coefficients)

    # step 1.5. remove boundary coefficients
    if remove_bc:
        nbc = n_boundary_coefs(filter, J, **kwargs)
        scaling_coefficients = scaling_coefficients[nbc:, :]
        x = x[nbc:]

    # step 2. produce J wavelet coefficient
    wavelet_coefficients = [scaling_coefficients[:, j-1] - scaling_coefficients[:, j] for j in range(1, J)]
    wavelet_coefficients = [x - scaling_coefficients[:, 0]] + wavelet_coefficients
    wavelet_coefficients = np.column_stack(wavelet_coefficients)

    # step 3. column concatenate wavelet coefficients and the Jth level scaling coefficient
    return np.hstack((wavelet_coefficients, scaling_coefficients[:, J-1][:, None]))

def multi_stationary_dwt(X, y=None, transform = "modwt", filter="db1", J=1, pandas_output = False, remove_bc=True, **kwargs):

    '''
    X is a 2D numpy array
    filter is a string
    J is an integer
    **kwargs: Used to specify max_L, max_J for cutting off boundary coefficients.
    '''
    
    assert len(X.shape) > 1
    assert X.shape[1] > 0
    assert X.shape[0] > X.shape[1]

    n_inputs = X.shape[1]

    if transform == "modwt":
        out = np.apply_along_axis(modwt, 0, X, filter, J, remove_bc, **kwargs)
    elif transform == "atrousdwt":
        out = np.apply_along_axis(atrousdwt, 0, X, filter, J, remove_bc, **kwargs)
    else:
        raise ValueError('Currently, only modwt and atrousdwt are supported for the argument transform')
    
    out = np.split(out, n_inputs, 2)
    out = [i.squeeze() for i in out]
    out = np.concatenate(out, axis=1)

    if pandas_output:
        out = pd.DataFrame(out, columns=make_output_names(n_inputs, J))

    if y is not None:
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1

        if remove_bc:
            num_bc = n_boundary_coefs(filter, J, **kwargs)
            y = y[num_bc:]
        
        assert out.shape[0] == y.shape[0]

        if pandas_output:
            y = pd.Series(y, name="y")
        
        return out, y
    else:
        return out
