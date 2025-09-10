"""
Rate of Spread (ROS) Calculation Module

This module implements the Rate of Spread (ROS) function as defined in the
Canadian Forest Fire Danger Rating System (CFFDRS).

Functions in this module are based on the following publications:

- Forestry Canada Fire Danger Group (FCFDG) (1992). "Development and Structure
  of the Canadian Forest Fire Behavior Prediction System." Technical Report
  ST-X-3, Forestry Canada, Ottawa, Ontario.

- Wotton, B.M., Alexander, M.E., Taylor, S.W. (2009). "Updates and revisions
  to the 1992 Canadian forest fire behavior prediction system." Information
  Report GLC-X-10, Natural Resources Canada, Sault Ste. Marie, Ontario.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import fwi
from ..param import Param, broadcast
from .fuel import Code, Model, model


def ros(
    code: Code,
    bui: Param,
    isi: Param,
    lat: Param = 46.0,
    modifier: Optional[float] = None,
) -> np.ndarray:
    """
    Vectorized surface Rate-of-Spread (m · min⁻¹) for any fuel-type model.

    *Conifer, deciduous, mixed, slash, grass — all handled here.*
    Crown overrides (C6) are honored automatically via model.ros() if present.

    If no modifier is provided, the appropriate default value for the model will be used.
        - M1, M2:   Percentage of conifer in the mixedwood (pc=80.0).
        - M3, M4:   Percentage of dead balsam fir in the mixedwood (pdf=35.0).
        - O1A, O1B: Percentage of crown closure (cc=50.0).

    Args:
        code: Fuel type code.
        bui:  Build-up index.
        isi:  Initial spread index.
        lat:  Latitude in degrees (default=46.0).
        modifier: Optional percent modifier.

    Examples:
        # ----------------------------------------------------------------
        # Passing the pc modifier to M1 and M2 models:
        # ----------------------------------------------------------------
        >>> pc: float = 50.0
        >>> m1 = ros(Code.M1, bui, isi, modifier=pc)
        >>> m2 = ros(Code.M2, bui, isi, modifier=pc)
        # ----------------------------------------------------------------
        # Passing the modifier with a random name to M1 and M2 models:
        # ----------------------------------------------------------------
        >>> # The name of the modifier does not matter.
        >>> # It is correctly interpreted as the percentage of dead balsam fir (pdf).
        >>> dead_fir_percentage: float = 35.0
        >>> m3 = ros(Code.M3, bui, isi, modifier=dead_fir_percentage)
        >>> m4 = ros(Code.M4, bui, isi, modifier=dead_fir_percentage)
        # ----------------------------------------------------------------
        # Passing the modifier to a model that does not use it (e.g., C1):
        # ----------------------------------------------------------------
        >>> # The modifier is ignored for C1, no effect on results.
        >>> pc: float = 50.0
        >>> c1 = ros(Code.C1, bui, isi, modifier=pc)
        # ----------------------------------------------------------------
        # Passing the modifier directly to the keyword argument:
        # ----------------------------------------------------------------
        >>> # The modifier is correctly interpreted as the percentage of conifer (pc).
        >>> m1 = ros(Code.M1, bui, isi, modifier=50.0)
        >>> m2 = ros(Code.M2, bui, isi, modifier=50.0)
        # ----------------------------------------------------------------
        # Passing the modifier with a random name to C1:
        # ----------------------------------------------------------------
        >>> # The modifier is ignored for C1, no effect on results.
        >>> dead_fir_percentage: float = 35.0
        >>> c1 = ros(Code.C1, bui, isi, modifier=dead_fir_percentage)
        # ----------------------------------------------------------------
        # Passing no modifier to a model that uses it (e.g., M1):
        # ----------------------------------------------------------------
        >>> # The default modifier is used (pc=80.0).
        >>> m1 = ros(Code.M1, bui, isi)
    """
    fmc = None
    args = [bui, isi]

    if code == Code.C6:
        fmc = fwi.fmc(lat)

    if fmc is None:
        bui, isi = broadcast(*args)
    else:
        args.append(fmc)
        bui, isi, fmc = broadcast(*args)

    mdl: Model = model(code, modifier)
    return mdl.ros(bui, isi, fmc=fmc)
