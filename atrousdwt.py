from coefs import scaling_coefs, n_boundary_coefs
import numpy as np

def atrousdwt(x, filter, J, remove_bc = True, **kwargs):

    '''
    **kwargs used to specify max_L, max_J for cutting off boundary coefficients
    '''

    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError('x should be a 1D Numpy array')
    
    assert len(x.shape) == 1

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