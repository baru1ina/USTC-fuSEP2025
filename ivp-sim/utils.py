import numpy as np
from numba import njit


@njit('float64[:](float64[:])')
def safe_log(x):
    x = np.asarray(x)
    x_safe = np.where(x <= 0, 1e-20, x)
    return np.log(x_safe)


@njit
def compute_Fp(epsn, kapt):
    kapn = 1 / epsn
    Fp = kapn / (kapn + kapt)
    return Fp

