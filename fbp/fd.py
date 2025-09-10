"""
Fire Description (FD) calculation for Fire Behaviour Prediction.

This module provides the main FD calculation function that determines
the fire description category based on head fire intensity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cffdrs import fwi
from cffdrs.param import Param
from .fuel import Code, factory


def fd(
    code: Code,
    bui: Param,
    isi: Param,
    ffmc: Optional[Param] = None,
    lat: Param = 46.0,
    modifier: Optional[float] = None,
) -> np.ndarray:
    """
    Vectorized Fire Description (FD) facade.

    Computes model-specific CFB internally (handling C6’s RSS/RSO/RSC path),
    then classifies FD via the model’s `fd(cfb, numeric)`.

    Returns "S"/"I"/"C" (default) or 1/2/3 if `numeric=True`.

    Parameters
    ----------
        code: Fuel code (single).
        bui: Buildup Index.
        isi: Initial Spread Index.
        ffmc: Fine Fuel Moisture Code (required for C1 and C7).
        lat: Latitude (default 46.0).
        modifier: Optional modifier to apply to the model.

    Returns
    -------
        Fire Description (FD).

    Raises
    ------
        ValueError if ffmc is required for C1 and C7 but not provided.
    """
    deps = code in (Code.C1, Code.C7)
    if deps and ffmc is None:
        raise ValueError("ffmc is required for fuel models C1 and C7.")

    fmc = fwi.fmc(lat)
    bui = np.asarray(bui, dtype=float)
    isi = np.asarray(isi, dtype=float)
    fmc = np.asarray(fmc, dtype=float)
    bui, isi = np.broadcast_arrays(bui, isi)
    ffmc = None if ffmc is None else np.broadcast_to(np.asarray(ffmc, float), bui.shape)

    m = factory.model(code, modifier)
    sfc = m.sfc(bui=bui, ffmc=ffmc)
    rss = m.rss(bui, isi)
    csi = m.csi(fmc)
    rso = m.rso(csi, sfc)
    if code == Code.C6:
        rsc = m.rsc(isi, fmc)
        cfb = m.cfb(rss, rso, rsc=rsc)
    else:
        cfb = m.cfb(rss, rso)

    cfl = np.broadcast_to(np.asarray(m.cfl(), float), cfb.shape)

    cfb = np.where(cfl > 0.0, cfb, 0.0)
    _fd_result = m.fd(cfb)

    # Ensure result is always a proper numpy array, not a scalar
    return np.asarray(_fd_result)
