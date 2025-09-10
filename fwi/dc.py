from typing import Optional, Union

import numpy as np
from numba import float64, int64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants (float64 by default; month uses int64 in the ufunc)
# ---------------------------------------------------------------------
MIN = 0.0
MIN_TEMP = -2.8

# Day length tables
NORTH = np.array(
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6], dtype=np.float64
)
SOUTH = np.array(
    [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8], dtype=np.float64
)
EQUATOR = 1.4

# Potential Evapotranspiration
TEMP_COEF = 0.36
TEMP_OFFSET = 2.8
DIVISOR = 2.0

# Rain threshold & effective rainfall
THRESHOLD_MM = 2.8
EFFECTIVE_COEFF = 0.83
EFFECTIVE_OFFSET = 1.27

# Moisture Index
MOISTURE_BASE = 800.0
DECAY_COEF = 400.0
RAIN_COEF = 3.937


# ---------------------------------------------------------------------
# Public API: same name/signature as your original, with optional lat.
# Handles dtype normalization & default lat, then calls the ufunc.
# ---------------------------------------------------------------------
def dc(
    temp: Param, prec: Param, dc0: Param = 15.0, month: Param = 7, lat: Optional[Param] = 46
) -> np.ndarray:
    """
    Calculate Drought Code (DC) with a Numba-compiled ufunc backend.

    Accepts scalars or arrays; broadcasts per NumPy rules.
    `lat` defaults to 46.0 when not provided.
    """
    temp_arr = np.asarray(temp, dtype=np.float64)
    prec_arr = np.asarray(prec, dtype=np.float64)
    dc0_arr = np.asarray(dc0, dtype=np.float64)
    month_arr = np.asarray(month, dtype=np.int64)
    lat_arr = np.asarray(lat, dtype=np.float64)

    # NumPy broadcasting applies; _dc_ufunc executes elementwise at native speed
    return np.array(_ufunc(temp_arr, prec_arr, dc0_arr, month_arr, lat_arr))


# ---------------------------------------------------------------------
# Numba-compiled scalar helpers
# ---------------------------------------------------------------------
@njit(cache=True)
def _dlf(lat: float, month: int) -> float:
    """
    Day length factor at (lat, month) for DC.
    month: 1..12
    """
    m = month - 1  # 0-based index
    # default: north of 20N
    dlf = NORTH[m]
    if lat <= -20.0:
        dlf = SOUTH[m]
    elif -20.0 < lat <= 20.0:
        dlf = EQUATOR
    return dlf


@njit(cache=True)
def _pe(dc0: float, prec: float) -> float:
    """
    DC after rain (Eq. 18-21 variant), clamped at 0.
    """
    rw = EFFECTIVE_COEFF * prec - EFFECTIVE_OFFSET  # Eq. 18
    smi = MOISTURE_BASE * np.exp(-dc0 / DECAY_COEF)  # Eq. 19
    dr0 = dc0 - DECAY_COEF * np.log(1.0 + RAIN_COEF * rw / smi)  # ~Eq. 21 variation
    return dr0 if dr0 > 0.0 else 0.0


@njit(cache=True)
def _scalar(temp: float, prec: float, dc0: float, month: int, lat: float) -> float:
    """
    Pure scalar DC calculation; calls the scalar helpers.
    """
    # Clamp temperature
    if temp < MIN_TEMP:
        temp = MIN_TEMP

    # Day length factor
    dlf = _dlf(lat, month)

    # Potential evapotranspiration (Eq. 22)
    pe = (TEMP_COEF * (temp + TEMP_OFFSET) + dlf) / DIVISOR
    if pe < 0.0:
        pe = 0.0

    # Rain effect and final DC (Eq. 23 variation)
    if prec <= THRESHOLD_MM:
        dr = dc0
    else:
        dr = _pe(dc0, prec)

    out = dr + pe
    return out if out > MIN else MIN


# ---------------------------------------------------------------------
# Numba ufunc (elementwise) built on the scalar core
# ---------------------------------------------------------------------
# Signature aligns with NumPy defaults: float64 for floats, int64 for ints.
@vectorize([float64(float64, float64, float64, int64, float64)], target="parallel", cache=True)
def _ufunc(temp, prec, dc0, month, lat):
    return _scalar(temp, prec, dc0, month, lat)
