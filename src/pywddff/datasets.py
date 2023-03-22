from importlib import resources
from pywddff.utils import load_pickle

def get_camels_subset():
    """Get dictionary containing a subset of the CAMELS data set.

    Returns
    -------
    dictionary

    References
    ----------
    A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett, 2014. 
    A large-sample watershed-scale hydrometeorological dataset for the contiguous USA. 
    Boulder, CO: UCAR/NCAR. https://dx.doi.org/10.5065/D6MW2F4D
    """
    with resources.path("pywddff.data", "camels_subset.pkl") as f:
        data_file_path = f
    
    return load_pickle(data_file_path)

def get_scaling_dict():
    """Get dictionary of scaling filters.

    Returns
    -------
    dictionary
    """
    with resources.path("pywddff.data", "scaling_dict.pkl") as f:
        data_file_path = f
    
    return load_pickle(data_file_path)

def get_wavelet_dict():
    """Get dictionary of wavelet filters.

    Returns
    -------
    dictionary
    """
    with resources.path("pywddff.data", "wavelet_dict.pkl") as f:
        data_file_path = f
    
    return load_pickle(data_file_path)