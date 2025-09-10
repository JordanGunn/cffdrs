from __future__ import annotations

from typing import Final

import numpy as np

from loki.api.cffdrs.param import Param

from ..code import Code
from ..description import Description
from .abstract import Model


class _Deciduous(Model):
    SFC_FACTOR: float = 1.5
    SFC_DECAY: float = 0.0183
    SFC_POWER: float = 1.0

    def sfc(self, bui: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Surface fuel consumption for deciduous types."""
        _consumption = self.SFC_FACTOR * (1.0 - np.exp(-self.SFC_DECAY * bui))
        _result = np.maximum(_consumption, self.EPSILON)
        return np.asarray(_result, dtype=float)

    def cbh(self, *args) -> Param:
        """Delegate to base Model.cbh for consistent array return and validation."""
        return super().cbh(*args)


class D1(_Deciduous):
    """Leafless Aspen fuel model."""

    CODE: Final = Code.D1
    DESCRIPTION: Final = Description.D1
    A, B, C = 0, 0, 0  # D-type fuels don’t use Eq. 26
    Q: Final = 0.90
    BUI0: Final = 32.0
