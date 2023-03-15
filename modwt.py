# import pywt # OLD
from filters import wavelet_filter, scaling_filter
import numpy as np
import math

# *****************
# Author: John You
# *****************

def modwt(x, filter, J, remove_bc = True, max_L = None, max_J = None):

    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError('x should be a 1D Numpy array')

    # Get MODWT wavelet and scaling filters (OLD)
    # wavelet = pywt.Wavelet(filter)
    # h = wavelet.dec_hi # wavelet filter
    # g = wavelet.dec_lo # scaling filter
    # h_t = np.array(h) / np.sqrt(2) # MODWT re-scaled h
    # g_t = np.array(g) / np.sqrt(2) # MODWT re-scaled g

    # Get MODWT wavelet and scaling filters
    h_t = wavelet_filter(filter, modwt=True)
    g_t = scaling_filter(filter, modwt=True)

    # Sanity check
    assert h_t.shape[0] == g_t.shape[0]

    L = h_t.shape[0]
    N = x.shape[0]

    # More input validation
    if J >= math.log((N/(L-1))+1, 2):
        raise ValueError('J must be less than base 2 logarithm of (N/(L-1)) + 1')

    if max_J is not None and max_J < J:
        raise ValueError('max_J must be greater than or equal to J')

    if max_L is not None and max_L < L:
        raise ValueError('max_L must be greater than or equal to L')

    # Computing MODWT wavelet and scaling coefficients
    # Reference: Page 177, Wavelet Methods for Time Series Analysis, Percival & Walden (Comments and Extensions to Section 5.5)

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

    # Handling boundary coefficients
    if remove_bc:

        if max_L is None:
            max_L = L

        if max_J is None:
            max_J = J

        num_bc = (2**max_J - 1) * (max_L - 1)
        WV_J = WV_J[num_bc:, :]

    return WV_J
