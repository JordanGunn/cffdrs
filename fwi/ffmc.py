from typing import Union

import numpy as np
from numba import float64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants (float64)
# ---------------------------------------------------------------------
NORMAL_TEMP = 21.1
MIN = 0.0
BASE = 101.0
MAX = 250.0
DEFAULT = 150.0  # (not used below; kept for completeness)

# Rain
RAIN_A = 42.5
RAIN_B = 100.0
RAIN_C = 251.0
RAIN_D = 6.93
RAIN_HEAVY = 0.0015

RAIN_THRESH_MM = 0.5
RAIN_THRESH_MOISTURE = 150.0

# Equilibrium – drying (Eq. 4)
EQD_A = 0.942
EQD_B = 0.679
EQD_C = 11.0
EQD_D = 0.18
EQD_RH_COEF = 0.115

# Equilibrium – wetting (Eq. 5)
EQW_A = 0.618
EQW_B = 0.753
EQW_C = 10.0
EQW_D = 0.18
EQW_RH_COEF = 0.115

# Drying/Wetting rate (Eqs. 6–7)
RATE_A = 0.424
RATE_B = 0.0694
RATE_TEMP_COEF = 0.581
RATE_TEMP_EXP = 0.0365
RH_EXP_A = 1.7
RH_EXP_B = 8.0

# Conversion (Eqs. 1 & 10)
CONV_A = 147.27723
CONV_C = 59.5

# ---------------------------------------------------------------------
# Scalar helpers (Numba-compiled)
# ---------------------------------------------------------------------


@njit(cache=True)
def _ffmc_to_moisture_scalar(ffmc0: float) -> float:
    # Eq. 1: moisture from FFMC
    # moisture = CONV_A * (BASE - ffmc) / (CONV_C + ffmc)
    return CONV_A * (BASE - ffmc0) / (CONV_C + ffmc0)


@njit(cache=True)
def _equilibrium_drying_scalar(rh: float, temp: float) -> float:
    # Eq. 4
    t1 = EQD_A * (rh**EQD_B)
    t2 = EQD_C * np.exp((rh - 100.0) / 10.0)
    t3 = EQD_D * (NORMAL_TEMP - temp)
    t4 = 1.0 - 1.0 / np.exp(rh * EQD_RH_COEF)
    return t1 + t2 + t3 * t4


@njit(cache=True)
def _equilibrium_wetting_scalar(rh: float, temp: float) -> float:
    # Eq. 5
    t1 = EQW_A * (rh**EQW_B)
    t2 = EQW_C * np.exp((rh - 100.0) / 10.0)
    t3 = EQW_D * (NORMAL_TEMP - temp)
    t4 = 1.0 - 1.0 / np.exp(rh * EQW_RH_COEF)
    return t1 + t2 + t3 * t4


@njit(cache=True)
def _rain_term_scalar(ra: float, wmo: float) -> float:
    """Intermediate rain term (part of Eq. 3)."""
    if ra <= 0.0:
        return 0.0
    return RAIN_A * ra * np.exp(-RAIN_B / (RAIN_C - wmo)) * (1.0 - np.exp(-RAIN_D / ra))


@njit(cache=True)
def _apply_rain_scalar(prec: float, wmo: float) -> float:
    """Apply rain effects to initial moisture wmo (Eqs. 2–3), with heavy rain handling."""
    # Eq. 2: effective rain after canopy threshold
    if prec > RAIN_THRESH_MM:
        ra = prec - RAIN_THRESH_MM
    else:
        # no significant rain; return unchanged
        return wmo

    # Eq. 3 terms
    rain_exp_term = _rain_term_scalar(ra, wmo)

    if wmo > RAIN_THRESH_MOISTURE:
        # heavy rain curve
        # heavy term: RAIN_HEAVY * (wmo - 150)^2 * sqrt(ra)
        heavy_term = (
            RAIN_HEAVY * (wmo - RAIN_THRESH_MOISTURE) * (wmo - RAIN_THRESH_MOISTURE) * np.sqrt(ra)
        )
        return wmo + heavy_term + rain_exp_term
    else:
        # normal rain
        return wmo + rain_exp_term


@njit(cache=True)
def _dry_rate_scalar(rh: float, ws: float) -> float:
    # Eq. 6a: z_dry at normal temperature (only used when drying applies)
    # ((100 - rh)/100)^p
    one_minus = (100.0 - rh) / 100.0
    term_a = RATE_A * (1.0 - (one_minus**RH_EXP_A))
    term_b = RATE_B * np.sqrt(ws if ws > 0.0 else 0.0) * (1.0 - (one_minus**RH_EXP_B))
    return term_a + term_b


@njit(cache=True)
def _wet_rate_scalar(rh: float, ws: float) -> float:
    # Eq. 7a: z_wet at normal temperature
    # (rh/100)^p
    rh01 = rh / 100.0
    term_a = RATE_A * (1.0 - (rh01**RH_EXP_A))
    term_b = RATE_B * np.sqrt(ws if ws > 0.0 else 0.0) * (1.0 - (rh01**RH_EXP_B))
    return term_a + term_b


@njit(cache=True)
def _temperature_effect_scalar(z: float, temp: float) -> float:
    # Eqs. 6b / 7b: x = z * RATE_TEMP_COEF * exp(RATE_TEMP_EXP * temp)
    return z * RATE_TEMP_COEF * np.exp(RATE_TEMP_EXP * temp)


@njit(cache=True)
def _ffmc_scalar(temp: float, rh: float, ws: float, prec: float, ffmc0: float) -> float:
    """
    Scalar FFMC calculation faithful to your vector logic.
    """
    # Eq. 1: convert previous FFMC -> initial moisture wmo
    wmo = _ffmc_to_moisture_scalar(ffmc0)

    # Eqs. 2–3: rain effects (if any)
    wmo = _apply_rain_scalar(prec, wmo)

    # Cap moisture at MAX
    if wmo > MAX:
        wmo = MAX

    # Equilibrium moisture contents
    ed = _equilibrium_drying_scalar(rh, temp)  # Eq. 4
    ew = _equilibrium_wetting_scalar(rh, temp)  # Eq. 5

    wm = wmo

    # Drying branch: when wmo < ed and wmo < ew
    if (wmo < ed) and (wmo < ew):
        z_dry = _dry_rate_scalar(rh, ws)  # Eq. 6a
        x_dry = _temperature_effect_scalar(z_dry, temp)  # Eq. 6b
        # Eq. 8: wm = ew - (ew - wmo) / (10 ** x_dry)
        wm = ew - (ew - wmo) / (10.0**x_dry)

    # Wetting branch: when wmo > ed
    if wmo > ed:
        z_wet = _wet_rate_scalar(rh, ws)  # Eq. 7a
        x_wet = _temperature_effect_scalar(z_wet, temp)  # Eq. 7b
        # Eq. 9: wm = ed + (wmo - ed) / (10 ** x_wet)
        wm = ed + (wmo - ed) / (10.0**x_wet)

    # Eq. 10: moisture -> FFMC
    ff = (CONV_C * (MAX - wm)) / (CONV_A + wm)

    # Clamp
    if ff < MIN:
        ff = MIN
    elif ff > MAX:
        ff = MAX
    return ff


# ---------------------------------------------------------------------
# Numba ufunc (elementwise; NumPy broadcasts)
# ---------------------------------------------------------------------
@vectorize([float64(float64, float64, float64, float64, float64)], target="parallel", cache=True)
def _ffmc_ufunc(temp, rh, ws, prec, ffmc0):
    return _ffmc_scalar(temp, rh, ws, prec, ffmc0)


# ---------------------------------------------------------------------
# Public API: same name/signature as before; dtype normalization.
# ---------------------------------------------------------------------
def ffmc(temp: Param, rh: Param, ws: Param, prec: Param, ffmc0: Param = 85.0) -> np.ndarray:
    """
    Fine Fuel Moisture Code (FFMC), Numba-accelerated ufunc.
    Accepts scalars or arrays; broadcasts per NumPy rules.
    """
    temp_arr = np.asarray(temp, dtype=np.float64)
    rh_arr = np.asarray(rh, dtype=np.float64)
    ws_arr = np.asarray(ws, dtype=np.float64)
    prec_arr = np.asarray(prec, dtype=np.float64)
    ffmc0_arr = np.asarray(ffmc0, dtype=np.float64)

    return np.array(_ffmc_ufunc(temp_arr, rh_arr, ws_arr, prec_arr, ffmc0_arr))
