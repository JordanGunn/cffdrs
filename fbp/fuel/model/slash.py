from __future__ import annotations

from typing import Final

import numpy as np

from ..code import Code
from ..description import Description
from .abstract import Model


# --------------------------------------------------------------------------- #
#  Slash fuel model implementations                                           #
# --------------------------------------------------------------------------- #
class _Slash(Model):
    """Shared helpers for slash fuels."""

    CBH: Final[float] = 0.0  # slash fuels have no aerial crown
    CFL: Final[float] = 0.0

    def sfc(self, bui: np.ndarray, **__) -> np.ndarray:  # type: ignore[override]
        """Default slash SFC — overridden in each subclass."""
        raise NotImplementedError


class S1(_Slash):
    """Jack or Lodgepole Pine Slash fuel model."""

    CODE = Code.S1
    DESCRIPTION = Description.S1
    SFC_BUI_FACTOR_I = 4.0
    SFC_BUI_DECAY_I = 0.025
    SFC_BUI_FACTOR_II = 4.0
    SFC_BUI_DECAY_II = 0.034
    # RSI (Eq. 26)
    A, B, C = 75.0, 0.0297, 1.3

    # Build‑Up Effect (Eq. 54)
    Q = 0.75
    BUI0 = 38.0

    # SFC coefficients (Eq. 19, 20, 25)
    def sfc(self, bui: np.ndarray, **__) -> np.ndarray:  # type: ignore[override]
        """Surface fuel consumption for S1 model:"""
        _part1 = self.SFC_BUI_FACTOR_I * (1.0 - np.exp(-self.SFC_BUI_DECAY_I * bui))
        _part2 = self.SFC_BUI_FACTOR_II * (1.0 - np.exp(-self.SFC_BUI_DECAY_II * bui))
        _result = np.maximum(_part1 + _part2, Model.EPSILON)
        return np.asarray(_result, dtype=float)


class S2(S1):
    """White Spruce - Balsam Slash fuel model."""

    CODE: Final[Code] = Code.S2
    DESCRIPTION: Final[Description] = Description.S2
    A, B, C = 40.0, 0.0438, 1.7
    Q, BUI0 = 0.75, 63.0
    SFC_BUI_FACTOR_I: Final = 10.0
    SFC_BUI_DECAY_I: Final = 0.013
    SFC_BUI_FACTOR_II: Final = 6.0
    SFC_BUI_DECAY_II: Final = 0.060


class S3(S1):
    """Coastal Cedar - Hemlock - Douglas Fir Slash fuel model."""

    CODE: Final[Code] = Code.S3
    DESCRIPTION: Final[Description] = Description.S3
    A, B, C = 55.0, 0.0829, 3.2
    Q, BUI0 = 0.75, 31.0
    SFC_BUI_FACTOR_I: Final[float] = 12.0
    SFC_BUI_DECAY_I: Final[float] = 0.0166
    SFC_BUI_FACTOR_II: Final[float] = 20.0
    SFC_BUI_DECAY_II: Final[float] = 0.0210
