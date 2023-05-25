from pywddff.pywddff import *
import numpy as np

np.random.seed(42)  # set the random seed to 42 for reproducibility
N = 10000
p = 6
J = 6
X = np.random.rand(N,p)  # create a random numpy array of shape (10000,6) with values between 0 and 1
x = np.random.random(N)
# filters = ['bl7', 'bl9', 'bl10',
# 'beyl',
# 'coif1', 'coif2', 'coif3', 'coif4', 'coif5',
# 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12',
# 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23',
# 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33',
# 'db34', 'db35', 'db36', 'db37', 'db38', 'db39', 'db40', 'db41', 'db42', 'db43', 'db44', 'db45',
# 'fk4', 'fk6', 'fk8', 'fk14', 'fk18', 'fk22',
# 'han2_3', 'han3_3', 'han4_5', 'han5_5',
# 'dmey',
# 'mb4_2', 'mb8_2', 'mb8_3', 'mb8_4', 'mb10_3', 'mb12_3', 'mb14_3', 'mb16_3', 'mb18_3', 'mb24_3', 'mb32_3',
# 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14',
# 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'sym21', 'sym22', 'sym23', 'sym24', 'sym25', 'sym26', 'sym27',
# 'sym28', 'sym29', 'sym30', 'sym31', 'sym32', 'sym33', 'sym34', 'sym35', 'sym36', 'sym37', 'sym38', 'sym39', 'sym40',
# 'sym41', 'sym42', 'sym43', 'sym44', 'sym45',
# 'vaid',
# 'la8', 'la10', 'la12', 'la14', 'la16', 'la18', 'la20']

filters = ['bl7', 'beyl'] # 2 random filters

def test_modwt():

    # Test 1: remove_bc = True, max_L = None, max_J = None
    for filter in filters:
        for j in [J]:
            x_modwt = modwt(x, filter, j, remove_bc = True)
            nbc = n_boundary_coefs(filter, j)

            assert isinstance(x_modwt, np.ndarray), "Output of modwt is not a numpy ndarray!"
            assert x_modwt.shape[0] == N-nbc, "Output of modwt does not have the correct number of rows!"
            assert x_modwt.shape[1] == j+1, "Output of modwt does not have the correct number of columns!"

    # Test 2: remove_bc = False, max_L = None, max_J = None
    for filter in filters:
        for j in [J]:
            x_modwt = modwt(x, filter, j, remove_bc = False)

            assert isinstance(x_modwt, np.ndarray), "Output of modwt is not a numpy ndarray!"
            assert x_modwt.shape[0] == N, "Output of modwt does not have the correct number of rows!"
            assert x_modwt.shape[1] == j+1, "Output of modwt does not have the correct number of columns!"

def test_atrousdwt():

    # Test 1: remove_bc = True, max_L = None, max_J = None
    for filter in filters:
        for j in [J]:
            x_atrousdwt = atrousdwt(x, filter, j, remove_bc = True)
            nbc = n_boundary_coefs(filter, j)

            assert isinstance(x_atrousdwt, np.ndarray), "Output of atrousdwt is not a numpy ndarray!"
            assert x_atrousdwt.shape[0] == N-nbc, "Output of atrousdwt does not have the correct number of rows!"
            assert x_atrousdwt.shape[1] == j+1, "Output of atrousdwt does not have the correct number of columns!"

    # Test 2: remove_bc = False, max_L = None, max_J = None
    for filter in filters:
        for j in [J]:
            x_atrousdwt = atrousdwt(x, filter, j, remove_bc = False)

            assert isinstance(x_atrousdwt, np.ndarray), "Output of atrousdwt is not a numpy ndarray!"
            assert x_atrousdwt.shape[0] == N, "Output of atrousdwt does not have the correct number of rows!"
            assert x_atrousdwt.shape[1] == j+1, "Output of atrousdwt does not have the correct number of columns!"

def test_multi_stationary_dwt():
    FILTER = "db4"
    j = 6

    # Test 1: remove_bc = True, , max_L = None, max_J = None
    X_modwt = multi_stationary_dwt(X, y=None, transform = "modwt", filter=FILTER, J=j, remove_bc=True, approach = "single")
    nbc = n_boundary_coefs(FILTER, j)

    assert X_modwt.shape[0] == N - nbc
    assert X_modwt.shape[1] == p*(j+1)


