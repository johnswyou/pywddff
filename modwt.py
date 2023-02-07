import pywt
import numpy as np

# Reference: Page 177, Wavelet Methods for Time Series Analysis (Comments and Extensions to Section 5.5)
#            Percival & Walden

def modwt(x, filter, J):

    if not isinstance(x, np.ndarray):
        raise TypeError('x should be a 1D Numpy array')

    wavelet = pywt.Wavelet(filter)
    h = wavelet.dec_hi # wavelet filter
    g = wavelet.dec_lo # scaling filter
    h_t = np.array(h) / np.sqrt(2) # MODWT re-scaled h
    g_t = np.array(g) / np.sqrt(2) # MODWT re-scaled g

    assert h_t.shape[0] == g_t.shape[0]

    L = h_t.shape[0]
    N = x.shape[0]

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

    return np.column_stack([W, V])
