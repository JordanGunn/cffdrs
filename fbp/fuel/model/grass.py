"""Grass fuel models (O‑series) – O1A, O1B

Implements:
  * RSI with curing factor (Eqs. 35b & 36)
  * Constant SFC (Eq. 18)
  * Build‑Up Effect disabled (Q = 1, BUI0 = 1)
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ..code import Code
from ..description import Description
from .abstract import Model


# -----------------------------------------------------------------------------
#  Base: common curing + SFC logic
# -----------------------------------------------------------------------------
class _Grass(Model):
    """
    Base class for O‑series grass fuels.

    Attributes:
        Q (float): Build‑Up disabled (Eq.54 → 1).
        BUI0 (float): Build‑Up disabled (Eq.54 → 1).
        GFL (float): Constant surface‑fuel consumption (kgm‑2).
    """

    A: float
    B: float
    C: float
    Q: Final = 1.0
    BUI0: Final = 1.0
    GFL: Final = 0.35
    CF_FACTOR_I: Final = 0.005
    CF_THRESHOLD: Final = 58.8
    CF_FACTOR_II = 0.02
    CF_CONSTANT = 0.176
    CF_POWER = 0.061

    def __init__(self, cc: float = 80.0):
        """cc – curing / crown‑cover percent (default 80%)."""
        super().__init__(modifier=cc)
        self.cc: float = cc  # stored as 0–100

    def cf(self, cc: np.ndarray) -> np.ndarray:
        """
        Curing function CF (Eq.35b).

        RSI = a*(1-exp(-b*ISI))^c  ×  CF   (Eq.36)
        """
        return np.where(
            cc < self.CF_THRESHOLD,
            self.CF_FACTOR_I * (np.exp(self.CF_POWER * cc) - 1.0),
            self.CF_CONSTANT + self.CF_FACTOR_II * (cc - self.CF_THRESHOLD),
        )

    def rsi(self, isi: np.ndarray, **__) -> np.ndarray:  # type: ignore[override]
        """Rate of spread index (RSI) with curing factor (Eq.36)."""
        _cf = self.cf(np.full_like(isi, self.cc))
        _core = self.A * (1.0 - np.exp(-self.B * isi)) ** self.C
        _rsi = _core * _cf
        return np.asarray(_rsi, dtype=float)

    def sfc(self, *_, **__) -> np.ndarray:  # type: ignore[override]
        """Surface‑fuel consumption (Eq.18) – constant grass fuel load"""
        return np.array(self.GFL, dtype=float)


class O1A(_Grass):
    """Matted grass fuel model (O1A)."""

    CODE: Final = Code.O1A
    DESCRIPTION: Final = Description.O1A
    A: Final = 190.0
    B: Final = 0.0310
    C: Final = 1.4


class O1B(_Grass):
    """Standing grass fuel model (O1B)."""

    CODE: Final = Code.O1B
    DESCRIPTION: Final = Description.O1B
    A: Final = 250.0
    B: Final = 0.0350
    C: Final = 1.7


# -----------------------------------------------------------------------------
#  Public exports
# -----------------------------------------------------------------------------

# Alias to expose base grass model for generic usage/imports
Grass = _Grass  # type: ignore

__all__ = [
    "O1A",
    "O1B",
    "Grass",
]
