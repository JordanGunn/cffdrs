from typing import Union

import numpy as np
from numba import float64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants (flattened)
# ---------------------------------------------------------------------
MIN = 0.0

# Log curve (Eqs. 30a/30b)
COEF_BASE_E = 2.72  # base "e" coefficient in exponent
LOG_THRESH = 1.0  # threshold for piecewise on bb
LOG_MULT = 0.434  # multiplier on ln(bb); 0.434 ≈ 1/ln(10)
LOG_EXP = 0.647  # exponent on (log(bb) * LOG_MULT)

# BUI piecewise (Eqs. 28a, 28b)
BUI_HI_EXP = 0.023
BUI_HI_THRESH = 80.0
BUI_HI_NUM = 1000.0
BUI_HI_DEN_B = 25.0
BUI_HI_DEN_F = 108.64

BUI_LOW_EXP = 0.809
BUI_LOW_COEF = 0.626
BUI_LOW_CONST = 2.0

# ISI/BUI scale factor (Eq. 29/26 multiplier) - FIXED to match official CFFDRS
# Official CFFDRS uses 0.1, not 0.208 (was causing systematically high FWI values)
SCALE_FACTOR = 0.1  # ← CRITICAL BUG FIX: Changed from 0.208 to 0.1


# ---------------------------------------------------------------------
# Scalar core
# ---------------------------------------------------------------------
@njit(cache=True)
def _isclose(a: float, b: float, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> bool:
    """Numba-compatible replacement for math.isclose."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@njit(cache=True)
def _scalar(bui: float, isi: float) -> float:
    """
    Scalar Fire Weather Index from precomputed BUI and ISI.
    Uses a Numba-compatible isclose implementation for the bb == 1.0 boundary
    (Eq. 30a vs 30b).
    """
    # Eqs. 28a/28b: piecewise bb
    if bui > BUI_HI_THRESH:
        denom = BUI_HI_DEN_B + BUI_HI_DEN_F / np.exp(BUI_HI_EXP * bui)
        bb = SCALE_FACTOR * isi * (BUI_HI_NUM / denom)
    else:
        bb = SCALE_FACTOR * isi * (BUI_LOW_COEF * (bui**BUI_LOW_EXP) + BUI_LOW_CONST)

    # Eqs. 30a/30b: piecewise FWI with isclose for equality to 1.0
    if (bb < LOG_THRESH) or _isclose(bb, LOG_THRESH):
        fwi_val = bb
    else:
        fwi_val = np.exp(COEF_BASE_E * ((LOG_MULT * np.log(bb)) ** LOG_EXP))

    return fwi_val if fwi_val > MIN else MIN


# ---------------------------------------------------------------------
# Numba ufunc (elementwise; NumPy broadcasts)
# ---------------------------------------------------------------------
@vectorize([float64(float64, float64)], target="parallel", cache=True)
def _ufunc(bui, isi):
    return _scalar(bui, isi)


# ---------------------------------------------------------------------
# Public API: now accepts precomputed BUI and ISI
# ---------------------------------------------------------------------
def fwi(bui: Param, isi: Param) -> np.ndarray:
    """
    Fire Weather Index (FWI) from precomputed BUI and ISI.
    Accepts scalars or arrays; broadcasts per NumPy rules.
    Returns an ndarray (0-D for scalar inputs).
    """
    bui_arr = np.asarray(bui, dtype=np.float64)
    isi_arr = np.asarray(isi, dtype=np.float64)

    return np.asarray(_ufunc(bui_arr, isi_arr), dtype=np.float64)
