from typing import Union

import numpy as np
from numba import float64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
# Theoretical minimum BUI value
MIN = 0.0

# Eq. 27a coefficients
INI_SCALE = 0.8
INI_DC_WEIGHT = 0.4

# Eq. 27b correction coefficients
CORR_BASE = 0.92
CORR_COEFF = 0.0114
CORR_EXP = 1.7


# ---------------------------------------------------------------------
# Public API: from precomputed codes (preserves your style)
# ---------------------------------------------------------------------
def bui(dc: Param, dmc: Param) -> np.ndarray:
    """
    Buildup Index (BUI) from precomputed DC and DMC.
    Accepts scalars or arrays; broadcasts per NumPy rules.
    """
    dc_arr = np.asarray(dc, dtype=np.float64)
    dmc_arr = np.asarray(dmc, dtype=np.float64)
    out = _ufunc(dc_arr, dmc_arr)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------
# Scalar core (Numba-compiled)
# ---------------------------------------------------------------------
@njit(cache=True)
def _iszero(val: float, tol: float = 1e-9) -> bool:  # Numba-compatible zero check
    """Return True if *val* is approximately zero within *tol* absolute tolerance.

    Numba nopython mode does **not** support ``math.isclose`` (or ``numpy.isclose``)
    for scalars, so we provide a tiny helper that mimics the essential behaviour
    we need here (absolute tolerance only).
    """
    return abs(val) <= tol


@njit(cache=True)
def _scaler(dc: float, dmc: float) -> float:
    """
    Scalar BUI from precomputed DC and DMC (Van Wagner & Pickett Eq. 27a–27b).
    """
    # Eq. 27a: initial BUI (with zero-guard)
    if _iszero(dmc) and _iszero(dc):
        bui1 = 0.0
    else:
        denom = dmc + INI_DC_WEIGHT * dc
        # denom can't be zero here unless both dc and dmc are zero (handled above)
        bui1 = INI_SCALE * dc * dmc / denom

    # Eq. 27b: correction
    # p = 0 when dmc == 0, else (dmc - bui1)/dmc
    if _iszero(dmc):
        p = 0.0
    else:
        p = (dmc - bui1) / dmc

    cc = CORR_BASE + (CORR_COEFF * dmc) ** CORR_EXP
    bui0 = dmc - cc * p
    if bui0 < 0.0:
        bui0 = 0.0

    # Choose corrected vs initial
    bui_out = bui0 if (bui1 < dmc) else bui1

    # Clamp to MIN
    return bui_out if bui_out > MIN else MIN


# ---------------------------------------------------------------------
# Numba ufunc (elementwise; broadcasts like NumPy)
# ---------------------------------------------------------------------
@vectorize([float64(float64, float64)], target="parallel", cache=True)
def _ufunc(dc, dmc):
    return _scaler(dc, dmc)
