from typing import Union

import numpy
import numpy as np
from numba import float64, int64, njit, vectorize

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Param = Union[int, float, np.ndarray]

# ---------------------------------------------------------------------
# Constants (float64; month handled as int64)
# ---------------------------------------------------------------------
# Daylength tables (by month 1..12)
ELL_46N = np.array(
    [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0], dtype=np.float64
)
ELL_20N = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8], dtype=np.float64)
ELL_20S = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2], dtype=np.float64)
ELL_40S = np.array(
    [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8], dtype=np.float64
)
ELL_EQUATOR: float = 9.0  # equatorial band constant in original vector code

# Other constants
MIN_TEMP: float = -1.1

# Duff Moisture Code eqs. constants (mirroring your code)
# Eq. 16 dry rate rk = 1.894e-4*(T+1.1)*(100-RH)*ELL
RK_COEF: float = 1.894e-4

# Rain branch constants (Eqs. 11–15 variants)
RW_A: float = 0.92
RW_B: float = 1.27  # Eq. 11: rw = 0.92*prec - 1.27
WMI_A: float = 20.0
WMI_B: float = 280.0  # altered Eq. 12: wmi = 20 + 280/exp(0.023*dmc0)
WMI_EXP: float = 0.023
WM_DEN_A: float = 48.77  # Eq. 14 denominator offset
PR_A: float = 43.43  # altered Eq. 15: 43.43*(5.6348 - log(wmr-20))
PR_B: float = 5.6348

RAIN_THRESHOLD: float = 1.5  # your vector code uses 1.5 mm threshold


# ---------------------------------------------------------------------
# Public API: same name/signature; dtype normalization + default lat.
# ---------------------------------------------------------------------
def dmc(
    temp: Param, prec: Param, rh: Param, dmc0: Param = 6.0, month: Param = 7, lat: Param = 46.0
) -> np.ndarray:
    """
    Duff Moisture Code (DMC), Numba-accelerated.
    Accepts scalars or arrays; broadcasts per NumPy rules.
    `lat` defaults to 46.0 when not provided.
    """
    # Validate months once (avoid exception from inside ufunc)
    month_arr = np.asarray(month, dtype=np.int64)
    if month_arr.min() < 1 or month_arr.max() > 12:
        bad = np.unique(month_arr[(month_arr < 1) | (month_arr > 12)])
        raise ValueError(f"Month value must be in 1..12; invalid: {bad.tolist()}")

    temp_arr = np.asarray(temp, dtype=np.float64)
    prec_arr = np.asarray(prec, dtype=np.float64)
    rh_arr = np.asarray(rh, dtype=np.float64)
    dmc0_arr = np.asarray(dmc0, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)

    # NumPy broadcasting happens here
    return numpy.array(_func(temp_arr, prec_arr, rh_arr, dmc0_arr, month_arr, lat_arr))


# ---------------------------------------------------------------------
# Numba-compiled scalar helpers
# ---------------------------------------------------------------------
@njit(cache=True)
def _month(month: int) -> int:
    # Validate 1..12 (do it scalar to avoid overhead in wrapper if you like)
    if month < 1 or month > 12:
        # In a ufunc context, throwing is tricky; we validate in wrapper too.
        # Keep this as a guard for scalar use.
        raise ValueError("Month must be in 1..12")
    return month - 1  # 0-based


@njit(cache=True)
def _ell(lat: float, month_idx0: int) -> float:
    """
    Latitude band switch (scalar), month_idx0 is 0-based.
    Bands:
      - base = 46N
      - 20N:  10 < lat <= 30
      - 20S: -30 < lat <= -10
      - 40S: -90 <= lat <= -30
      - EQU: -10 < lat <=  10  -> constant 9.0
    """
    # default base band (46N)
    ell = ELL_46N[month_idx0]

    # order mirrors your vectorized conditions
    if (lat <= 30.0) and (lat > 10.0):  # 20N
        ell = ELL_20N[month_idx0]
    elif (lat <= -10.0) and (lat > -30.0):  # 20S
        ell = ELL_20S[month_idx0]
    elif (lat <= -30.0) and (lat >= -90.0):  # 40S
        ell = ELL_40S[month_idx0]
    elif (lat <= 10.0) and (lat > -10.0):  # Equatorial
        ell = ELL_EQUATOR

    return ell


@njit(cache=True)
def _b(dmc0: float) -> float:
    """Piecewise b (Eqs. 13a/b/c)."""
    if dmc0 <= 33.0:
        return 100.0 / (0.5 + 0.3 * dmc0)
    elif dmc0 <= 65.0:
        return 14.0 - 1.3 * np.log(dmc0)
    else:
        return 6.2 * np.log(dmc0) - 17.2


@njit(cache=True)
def _scalar(temp: float, prec: float, rh: float, dmc0: float, month: int, lat: float) -> float:
    """
    Scalar Duff Moisture Code (DMC) calculation, faithful to your vector logic.
    """
    # 1) Month index (1-based -> 0-based)
    m_idx = _month(month)

    # 2) Constrain temperature
    if temp < MIN_TEMP:
        temp = MIN_TEMP

    # 3) Eq. 16: rk with latitude-daylength factor
    ell = _ell(lat, m_idx)
    rk = RK_COEF * (temp + 1.1) * (100.0 - rh) * ell

    # 4) Rain branch
    if prec > RAIN_THRESHOLD:
        rw = RW_A * prec - RW_B  # Eq. 11
        wmi = WMI_A + WMI_B / np.exp(WMI_EXP * dmc0)  # altered Eq. 12
        b = _b(dmc0)  # Eqs. 13a/b/c
        wmr = wmi + 1000.0 * rw / (WM_DEN_A + b * rw)  # Eq. 14
        pr = PR_A * (PR_B - np.log(wmr - 20.0))  # altered Eq. 15
        if pr < 0.0:
            pr = 0.0
    else:
        pr = dmc0
        if pr < 0.0:
            pr = 0.0

    # 5) Final DMC
    out = pr + rk
    if out < 0.0:
        out = 0.0
    return out


# ---------------------------------------------------------------------
# Numba ufunc (elementwise) built on the scalar core
# Signature aligns with NumPy defaults: float64, int64 for month.
# ---------------------------------------------------------------------
@vectorize(
    [float64(float64, float64, float64, float64, int64, float64)], target="parallel", cache=True
)
def _func(temp, prec, rh, dmc0, month, lat):
    return _scalar(temp, prec, rh, dmc0, month, lat)
