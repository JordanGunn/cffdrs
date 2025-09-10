from typing import Union

import numpy as np
from numba import float64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants (match your module)
# ---------------------------------------------------------------------
MIN: float = 0.0
COEF: float = 0.0272
EXP: float = 1.77


# ---------------------------------------------------------------------
# Scalar core (Numba-compiled)
# ---------------------------------------------------------------------
@njit(cache=True)
def _scalar(fwi: float) -> float:
    """
    Scalar Daily Severity Rating from FWI.
    DSR = COEF * (FWI ** EXP), clamped at MIN.
    """
    val = COEF * (fwi**EXP)
    return val if val > MIN else MIN


# ---------------------------------------------------------------------
# Numba ufunc (elementwise; broadcasts like NumPy)
# ---------------------------------------------------------------------
@vectorize([float64(float64)], target="parallel", cache=True)
def _ufunc(fwi):
    return _scalar(fwi)


# ---------------------------------------------------------------------
# Public API (accepts precomputed FWI)
# ---------------------------------------------------------------------
def dsr(fwi: Param) -> np.ndarray:
    """
    Daily Severity Rating (DSR) from precomputed FWI.
    Accepts scalars or arrays; broadcasts per NumPy rules.
    Returns an ndarray (0-D when scalar input).
    """
    _fwi = np.asarray(fwi, dtype=np.float64)
    return np.asarray(_ufunc(_fwi), dtype=np.float64)
