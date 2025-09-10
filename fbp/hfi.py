"""
Head Fire Intensity (HFI) calculation for Fire Behaviour Prediction.

This module provides the main HFI calculation function that determines
the head fire intensity based on fuel type and weather conditions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cffdrs import fwi
from cffdrs.param import Param, broadcast
from .fuel import Code, Model
from .fuel import model as factory


def hfi(
    code: Code,
    ros: Param,
    ffmc: Param,
    bui: Param,
    isi: Param,
    lat: Param = 46.0,
    modifier: Optional[float] = None,
) -> np.ndarray:
    """
    Head Fire Intensity (HFI) for a single fuel code.

    HFI = 300 * TFC * ROS

    This function computes TFC per the selected model:
      - SFC from model.sfc(...), passing ffmc/bui (models ignore unused args)
      - RSO = model.rso( model.csi(FMC(lat)), SFC )
      - Base CFB (all models except C6): model.cfb(ROS, RSO)
      - C6 CFB gate: model.cfb(RSS, RSO, rsc=RSC) where
            RSS = model.rss(bui, isi)
            RSC = model.rsc(isi, FMC(lat))
      - TFC = model.tfc(CFB, SFC)

    Args
    ----
    code: Fuel code (single).
    ros : Final rate of spread (m·min⁻¹).
    ffmc: Fine Fuel Moisture Code.
    bui : Buildup Index.
    isi : Initial Spread Index (used for C6’s RSS/RSC).
    lat : Latitude (deg N) for FMC calculation. Scalar or array. Default 46.0.
    modifier : Optional model modifier (pc/pdf/cc depending on code).

    Returns
    -------
    np.ndarray – Head Fire Intensity (kW·m⁻¹).
    - Shape follows broadcast of inputs.
    """
    _fmc = fwi.fmc(lat)

    # Broadcast common inputs up-front (keeps shapes consistent).
    _ros, _ffmc, _bui, _isi, _fmc = broadcast(ros, ffmc, bui, isi, _fmc)

    # Model instance
    mdl: Model = factory(code, modifier)

    # Surface fuel consumption (model-specific; args are optional per model)
    # Conifers:
    #   C1.sfc(ffmc), C7.sfc(bui, ffmc), others use generic bui-only.
    # Other fuel types should also accept these kwargs or ignore extras.
    sfc = mdl.sfc(ffmc=_ffmc, bui=_bui)

    cfb_val = _cfb(mdl, _bui, _fmc, _isi, _ros, sfc)

    # Total fuel consumption and final HFI
    tfc = mdl.tfc(cfb_val, sfc)
    return mdl.hfi(_ros, tfc)


def _cfb(
    mdl: Model,
    bui: np.ndarray,
    fmc: np.ndarray,
    isi: np.ndarray,
    ros: np.ndarray,
    sfc: np.ndarray,
) -> np.ndarray:
    """Calculate the crown-fraction burned burned for the input model."""
    csi = mdl.csi(fmc)
    rso = mdl.rso(csi, sfc)

    if mdl.CODE == Code.C6:
        rss = mdl.rss(bui, isi)  # Eq. 63
        rsc = mdl.rsc(isi, fmc)  # Eq. 64 with FME
        cfb = mdl.cfb(rss, rso, rsc=rsc)
    else:
        cfb = mdl.cfb(ros, rso)
    return cfb
