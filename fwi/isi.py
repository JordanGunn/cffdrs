import math
from typing import Union

import numpy as np
from numba import float64, njit, vectorize

# If you already import this in your project, keep your import instead:
# from .constants import SCALE_FACTOR
SCALE_FACTOR: float = 0.208  # <-- replace with your .constants.SCALE_FACTOR

Param = Union[int, float, np.ndarray]

# -------------------------
# Constants (flattened)
# -------------------------
MIN = 0.0

# Wind effect (Eq. 24)
WIND_EFFECT_EXP_NORMAL = 0.05039
WIND_EFFECT_THRESH = 40.0
WIND_EFFECT_COEF = 12.0
WIND_EFFECT_EXP_FBP = -0.0818
WIND_EFFECT_OFFSET = 28.0
WIND_EFFECT_EXP_NORM = 0.05039

# Fine fuel moisture factor (Eq. 25)
FFM_COEFF = 91.9
FFM_EXP_COEF = -0.1386
FFM_POWER = 5.31
FFM_DENOM = 49_300_000.0

# FFMC -> moisture (Eq. 1), matches your Conversion.* constants
CONV_A = 147.27723
CONV_C = 59.5
BASE = 101.0


@njit(cache=True)
def _moisture(ffmc_val: float) -> float:
    """Eq. 1: convert FFMC → fuel moisture content (%)"""
    return CONV_A * (BASE - ffmc_val) / (CONV_C + ffmc_val)


@njit(cache=True)
def _high_winds(ws: float) -> float:
    """FBP ‘modified wind function’ for ws ≥ 40 km h⁻¹ (Eq. 24 alt.)"""
    return WIND_EFFECT_COEF * (1.0 - math.exp(WIND_EFFECT_EXP_FBP * (ws - WIND_EFFECT_OFFSET)))


@njit(cache=True)
def _we(ws: float) -> float:
    """
    Wind effect factor.
    """
    if ws >= WIND_EFFECT_THRESH:
        return _high_winds(ws)
    else:
        return math.exp(WIND_EFFECT_EXP_NORM * ws)


@njit(cache=True)
def _ffmf(mc: float) -> float:
    """Eq. 25: Fine-fuel moisture factor."""
    return FFM_COEFF * math.exp(FFM_EXP_COEF * mc) * (1.0 + (mc**FFM_POWER) / FFM_DENOM)


@njit(cache=True)
def _scalar(ffmc: float, ws: float) -> float:
    """Eq. 26: ISI = 0.208 × WE × FFMF."""
    mc = _moisture(ffmc)
    we = _we(ws)
    ffm = _ffmf(mc)
    _isi = SCALE_FACTOR * we * ffm
    return _isi if _isi > MIN else MIN


# ------------------------------------------------------------
# Numba ufunc (element-wise; broadcasts with NumPy)
# ------------------------------------------------------------
@vectorize(
    [float64(float64, float64)],
    target="parallel",
    cache=True,
    nopython=True,
)
def _isi_vec(ffmc_val, ws_val):  # <- 2-arg signature
    return _scalar(ffmc_val, ws_val)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def isi(ffmc: Param, ws: Param) -> np.ndarray:
    """
    Vectorised Initial Spread Index (ISI).

    Parameters
    ----------
    ffmc : scalar or array-like
    ws   : wind speed (km/h), scalar or array-like

    Returns
    -------
    np.ndarray broadcast to the shape of `ffmc` & `ws`
    """
    ffmc_arr = np.asarray(ffmc, dtype=np.float64)
    ws_arr = np.asarray(ws, dtype=np.float64)

    return np.array(_isi_vec(ffmc_arr, ws_arr))
