from modwt import modwt
from atrousdwt import atrousdwt
import numpy as np
from coefs import n_boundary_coefs, make_output_names
import pandas as pd

def multi_stationary_dwt(X, y=None, transform = "modwt", filter="db1", J=1, pandas_output = False, remove_bc=True, **kwargs):

    '''
    X is a 2D numpy array
    filter is a string
    J is an integer
    **kwargs are further arguments to be passed to modwt()
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