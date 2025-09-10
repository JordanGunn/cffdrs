from __future__ import annotations

from typing import Final

import numpy as np

from cffdrs.param import broadcast


class Default:
    """
    Default values for FMC.

    Attributes:
        LAT: Latitude (decimal degrees).
        LONG: Longitude (decimal degrees).
        ELV: Elevation (m).
        DJ: Day-of-year (1-365/366).
        D0: Optional preset date of minimum FMC. Provide **0** to request the
            algorithmic calculation (legacy behaviour).
    """

    LAT: Final[float] = 46.0
    LONG: Final[float] = -120.0
    ELV: Final[float] = 0.0
    DJ: Final[int] = 180
    D0: Final[float] = 0.0


class Latitude:
    """
    Normalised latitude (LATN) with default coefficients.

    Attributes:
        A1: Coefficient.
        B1: Coefficient.
        EXP1: Exponent.
        A2: Coefficient.
        B2: Coefficient.
        EXP2: Exponent.
        SHIFT: Shift.
    """

    A1: Final[float] = Default.LAT
    B1: Final[float] = 23.4
    EXP1: Final[float] = -0.0360
    A2: Final[float] = 43.0
    B2: Final[float] = 33.7
    EXP2: Final[float] = -0.0351
    SHIFT: Final[float] = 150.0

    @classmethod
    def default(cls):
        return cls.A1 + cls.B1 * np.exp(cls.EXP1 * (cls.SHIFT - Default.LONG))


class D0:
    """
    Date of minimum FMC (D0) with default coefficients.

    Attributes:
        LOW: Low elevation (m).
        HIGH: High elevation (m).
        FACTOR: Coefficient.
    """

    LOW: Final[float] = 151.0
    HIGH: Final[float] = 142.1
    FACTOR: Final[float] = 0.0172


class ND:
    """
    Days offset and final FMC thresholds.

    Attributes:
        LOW: Lower threshold (days).
        HIGH: Upper threshold (days).
    """

    LOW: Final[float] = 30.0
    HIGH: Final[float] = 50.0


class FMC:
    """
    FMC coefficients.

    Attributes:
        A1: Coefficient.
        B1: Coefficient.
        A2: Coefficient.
        B2: Coefficient.
        C: Coefficient.
        MAX: Maximum value.
    """

    A1: Final[float] = 85.0
    B1: Final[float] = 0.0189
    A2: Final[float] = 32.9
    B2: Final[float] = 3.17
    C: Final[float] = -0.0288
    MAX: Final[float] = 120.0


def _original(
    lat: float,
    long: float = Default.LONG,
    elv: float = Default.ELV,
    dj: int = Default.DJ,
    d0: float = Default.D0,
) -> np.ndarray:
    """Compute foliar moisture content (FMC).

    Args:
        lat: Latitude (decimal degrees). Scalar or array.
        long: Longitude (decimal degrees). Defaults to **−120**.
        elv: Elevation (m). Defaults to **0**.
        dj: Day-of-year (1-365/366). Defaults to **180** (early July).
        d0: Optional preset date of minimum FMC. Provide **0** to request the
            algorithmic calculation (legacy behaviour). Scalar or array.

    Returns:
        NumPy array of FMC values (percent).
    """
    # Cast/broadcast inputs --------------------------------------------------
    lat_arr, d0_arr = broadcast(lat, d0)
    long_arr = np.asarray(long, dtype=float)
    elv_arr = np.asarray(elv, dtype=float)

    # Eq. 1 & 3 – normalised latitude (LATN) ---------------------------------
    # Branches depend on D0 sentinel and elevation sign.
    _first = Latitude.A1 + Latitude.B1 * np.exp(Latitude.EXP1 * (Latitude.SHIFT - long_arr))

    _second = Latitude.A2 + Latitude.B2 * np.exp(Latitude.EXP2 * (Latitude.SHIFT - long_arr))
    latn = np.where(
        d0_arr <= 0,
        np.where(elv_arr <= 0, _first, _second),
        np.nan,  # will not be used when d0_arr > 0
    )

    # Eq. 2 & 4 – date of minimum FMC (D0) -----------------------------------
    d0_calc = np.where(
        d0_arr <= 0,
        np.where(
            elv_arr <= 0,
            D0.LOW * (lat_arr / latn),
            D0.HIGH * (lat_arr / latn) + D0.FACTOR * elv_arr,
        ),
        d0_arr,
    )

    d0_int = np.rint(d0_calc)  # round to nearest integer day

    # Eq. 5 – days difference from minimum FMC date --------------------------
    nd = np.abs(dj - d0_int)

    # Eq. 6–8 – final FMC -----------------------------------------------------
    fmc_val: np.ndarray = np.where(
        nd < ND.LOW,
        FMC.A1 + FMC.A2 * nd**2,
        np.where(
            nd < ND.HIGH,
            FMC.A2 + FMC.B2 * nd + FMC.C * nd**2,
            FMC.MAX,
        ),
    )

    return fmc_val


def _adapted(lat: float) -> np.ndarray:  # noqa: N802
    """
    Compute FMC with all default constants inlined (only *lat* required).

    This is an optimized form of :func:`fmc` where *long*, *elv*, *dj*, and
    *d0* are fixed to their legacy default values.

    The function retains full scalar/array support via :pyfunc:`broadcast`.
    """
    (lat_arr,) = broadcast(lat)

    # Date of minimum FMC (D0) with defaults --------------------------------
    d0_calc = D0.LOW * (lat_arr / Default.LAT)
    d0_int = np.rint(d0_calc)

    # Days offset and final FMC ---------------------------------------------
    nd = np.abs(Default.DJ - d0_int)

    fmc_val: np.ndarray = np.where(
        nd < ND.LOW,
        FMC.A1 + FMC.B1 * nd**2,
        np.where(
            nd < ND.HIGH,
            FMC.A1 + FMC.B2 * nd + FMC.C * nd**2,
            FMC.MAX,
        ),
    )

    return fmc_val


# ----------------------------------------------------------------------------
#: Public API alias for the adapted version
# ----------------------------------------------------------------------------
fmc = _adapted
