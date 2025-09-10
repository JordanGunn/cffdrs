from __future__ import annotations

from typing import Final

import numpy as np

from loki.api.cffdrs.param import Param

from ..code import Code
from ..description import Description
from .abstract import Model
from .conifer import C2
from .deciduous import D1


class _Mixed(Model):
    """Base class for mixed fuel models with shared SFC logic."""

    CBH: Final[float] = 6.00
    CFL: Final[float] = 0.80
    BUI_THRESHOLD: Final[float] = 60.0

    # Subclasses must define these constants
    SFC_EXPONENT: float
    SFC_MULTIPLIER: float

    def __init__(self, modifier: float):
        """Initialize the M1 model with a modifier.

        Args:
            modifier: percentage of conifer or dead balsam fir in the mixedwood.
        """
        super().__init__(modifier=modifier)
        self.modifier: float = modifier

    def cfc(self, cfb: np.ndarray) -> Param:  # noqa: N802
        """Crown fuel consumption (CFC) with mixed-fuel modifier.

        Applies the *modifier* (``pc`` for M1/M2, ``pdf`` for M3/M4) as a
        fractional multiplier to the base crown-fuel consumption from the
        abstract :py:class:`~loki.api.cffdrs.fuel.model.abstract.Model`.
        """

        _cfc = super().cfc(cfb)
        _modifier = self.modifier / 100.0
        _modifiers = np.asarray(_modifier, dtype=float)

        return _cfc * _modifiers


class M1(_Mixed):
    """Boreal Mixedwood - Leafless fuel model."""

    CODE: Code = Code.M1
    DESCRIPTION: Description = Description.M1
    Q: Final = 0.80
    BUI0: Final = 50.0

    def __init__(self, pc: float = 50.0):
        """Initialize the M1 model with a modifier.

        Args:
            pc: percentage of coniferous fuel in the mixedwood.
        """
        super().__init__(modifier=pc)
        self.pc: float = pc / 100.0
        self.pdf: float = 1 - self.pc

    def rsi(self, isi: np.ndarray, *__) -> np.ndarray:
        _con = C2().rsi(isi)
        _dec = D1().rsi(isi)
        _rsi = self.pc * _con + self.pdf * _dec  # M1 (no 0.2)
        return np.asarray(_rsi, dtype=float)

    # SFC: Eq. 17
    def sfc(self, bui: np.ndarray, *args, **kwargs) -> np.ndarray:
        _pc = np.asarray(self.pc, dtype=float)
        _con = C2().sfc(bui)
        _dec = D1().sfc(bui)
        _mix = _pc * _con + self.pdf * _dec
        _result = np.maximum(_mix, self.EPSILON)
        return np.asarray(_result, dtype=float)


class M2(M1):
    CODE, DESCRIPTION = Code.M2, Description.M2

    def __init__(self, pc: float = 80.0):
        """Initialize the M2 model with a modifier.

        Args:
            pc: percentage of coniferous fuel in the mixedwood.
        """
        super().__init__(pc=pc)

    def rsi(self, isi: np.ndarray, **__) -> np.ndarray:
        _con = C2().rsi(isi)
        _dec = D1().rsi(isi)
        _rsi = self.pc * _con + 0.2 * self.pdf * _dec
        return np.asarray(_rsi, dtype=float)

    def sfc(self, bui: np.ndarray, **__) -> np.ndarray:
        _con = C2().sfc(bui)
        _dec = D1().sfc(bui)
        _mix = self.pc * _con + 0.2 * (1 - self.pc) * _dec
        _result = np.maximum(_mix, self.EPSILON)
        return np.asarray(_result, dtype=float)


class M3(_Mixed):
    """Leafless Mixedwood with high dead balsam component (PDF weighting)."""

    CODE, DESCRIPTION = Code.M3, Description.M3
    A, B, C = 120.0, 0.0572, 1.4  # RSI coefficients from table
    Q, BUI0 = 0.80, 50.0

    # SFC: same as C2 (Eq. 10) for all M3
    SFC_FACTOR, SFC_DECAY, SFC_POWER = 5.0, 0.0115, 1.0

    def __init__(self, pdf: float = 35.0):
        """pdf – % dead balsam fir."""
        super().__init__(modifier=pdf)
        self.pdf: float = pdf / 100.0
        self.lf: float = 1 - self.pdf  # living fraction

    def rsi(self, isi: np.ndarray, **__) -> np.ndarray:
        """RSI: Eq. 29 + 30 (Wotton 2009)"""
        _dead = self.A * (1.0 - np.exp(-self.B * isi)) ** self.C  # dead-fir part
        _base = D1().rsi(isi)  # background D1
        _rsi = self.pdf * _dead + self.lf * _base
        return np.asarray(_rsi, dtype=float)

    def sfc(self, bui: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Surface fuel consumption for M3 model using C2-style formula."""
        _consumption = 1.0 - np.exp(-self.SFC_DECAY * bui)
        _sfc_val = (self.SFC_FACTOR * _consumption) ** self.SFC_POWER
        _result = np.maximum(_sfc_val, self.EPSILON)
        return np.asarray(_result, dtype=float)


class M4(_Mixed):
    """Leafed Mixedwood (late season) with dead balsam weighting."""

    CODE, DESCRIPTION = Code.M4, Description.M4
    A, B, C = 100.0, 0.0404, 1.48
    Q, BUI0 = 0.80, 50.0
    SFC_FACTOR, SFC_DECAY, SFC_POWER = 5.0, 0.0115, 1.0

    def __init__(self, pdf: float = 35.0):
        super().__init__(modifier=pdf)
        self.pdf: float = pdf / 100.0
        self.lf: float = 1 - self.pdf

    def rsi(self, isi: np.ndarray, **__) -> np.ndarray:
        _fir = self.A * (1.0 - np.exp(-self.B * isi)) ** self.C
        _base = D1().rsi(isi)
        _rsi = self.pdf * _fir + 0.2 * self.lf * _base  # Eq. 33 (0.2 factor)
        return np.asarray(_rsi, dtype=float)

    def sfc(self, bui: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Surface fuel consumption for M4 model using C2-style formula."""
        _consumption = 1.0 - np.exp(-self.SFC_DECAY * bui)
        _sfc_val = (self.SFC_FACTOR * _consumption) ** self.SFC_POWER
        _result = np.maximum(_sfc_val, self.EPSILON)
        return np.asarray(_result, dtype=float)
