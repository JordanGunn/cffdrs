from __future__ import annotations

from typing import Final, Optional

import numpy as np

from ..code import Code
from ..description import Description
from .abstract import Model


class _Conifer(Model):
    """
    Abstract base for conifer fuel models (C1–C7).

    • A, B, C        → RSI coefficients  (Eq. 26)
    • Q, BUI0        → Build-Up Effect   (Eq. 54)
    • SFC_FACTOR,
      SFC_DECAY,
      SFC_POWER      → Surface-fuel-consumption curve (Eqs. 10–12)
    """

    SFC_FACTOR: float = 5.0
    SFC_DECAY: float = 0.015
    SFC_POWER: float = 1.0

    def sfc(self, *args, **kwargs) -> np.ndarray:
        """Generic BUI-driven surface fuel consumption."""
        bui = kwargs.get("bui", None)
        if bui is None:
            if not args:
                raise TypeError("Conifer sfc() requires BUI: pass positionally or as bui=...")
            bui = args[0]

        _consumption = 1.0 - np.exp(-self.SFC_DECAY * bui)
        _sfc_val = (self.SFC_FACTOR * _consumption) ** self.SFC_POWER
        _result = np.maximum(_sfc_val, self.EPSILON)
        return np.asarray(_result, dtype=float)


class C1(_Conifer):
    """Spruce-Lichen Woodland fuel model."""

    CODE: Final = Code.C1
    DESCRIPTION: Final = Description.C1
    A: Final = 90.0
    B: Final = 0.0649
    C: Final = 4.5
    Q: Final = 0.90
    BUI0: Final = 72.0
    CBH: Final = 2.0
    CFL: Final = 0.75
    SFC_FFMC_WEIGHT: Final = 0.75
    SFC_FFMC_DECAY: Final = 0.23
    SFC_FFMC_DELTA: Final = 84.0

    def sfc(self, ffmc: Optional[np.ndarray], *args, **kwargs) -> np.ndarray:
        """C1 has specialized SFC calculation (Eq. 12 from FCFDG 1992)."""
        if ffmc is None:
            raise TypeError("Conifer sfc() requires FFMC: pass positionally or as ffmc=...")

        _delta = ffmc - self.SFC_FFMC_DELTA
        _consumption = np.sqrt(1.0 - np.exp(-self.SFC_FFMC_DECAY * np.abs(_delta)))
        _sfc = self.CFL + self.SFC_FFMC_WEIGHT * np.sign(_delta) * _consumption
        _result = np.maximum(_sfc, self.EPSILON)
        return np.asarray(_result, dtype=float)


class C2(_Conifer):
    CODE: Final = Code.C2
    DESCRIPTION: Final = Description.C2
    A: Final = 110.0
    B: Final = 0.0282
    C: Final = 1.5
    Q: Final = 0.70
    BUI0: Final = 64.0
    CBH: Final = 3.0
    CFL: Final = 0.80
    SFC_FACTOR: Final = 5.0
    SFC_DECAY: Final = 0.0115
    SFC_POWER: Final = 1.0


class C3(_Conifer):
    """Mature Jack or Lodgepole Pine fuel model."""

    CODE: Final = Code.C3
    DESCRIPTION: Final = Description.C3
    A: Final = 110.0
    B: Final = 0.0444
    C: Final = 3.0
    Q: Final = 0.75
    BUI0: Final = 62.0
    CBH: Final = 8.0
    CFL: Final = 1.15
    SFC_FACTOR: Final = 5.0
    SFC_DECAY: Final = 0.0164
    SFC_POWER: Final = 2.24


class C4(_Conifer):
    """Immature Jack or Lodgepole Pine fuel model."""

    CODE: Final = Code.C4
    DESCRIPTION: Final = Description.C4
    A: Final = 110.0
    B: Final = 0.0293
    C: Final = 1.5
    Q: Final = 0.80
    BUI0: Final = 66.0
    CBH: Final = 4.0
    CFL: Final = 1.20
    SFC_FACTOR: Final = 5.0
    SFC_DECAY: Final = 0.0164
    SFC_POWER: Final = 2.24


class C5(_Conifer):
    """Red and White Pine fuel model."""

    CODE: Final = Code.C5
    DESCRIPTION: Final = Description.C5
    A: Final = 30.0
    B: Final = 0.0697
    C: Final = 4.0
    Q: Final = 0.80
    BUI0: Final = 56.0
    CBH: Final = 18.0
    CFL: Final = 1.20
    SFC_FACTOR: Final = 5.0
    SFC_DECAY: Final = 0.0149
    SFC_POWER: Final = 2.48


class C6(_Conifer):
    """Conifer Plantation fuel model with specialized CBH calculation."""

    CODE: Final[Code] = Code.C6
    DESCRIPTION: Final = Description.C6
    A: Final = 30.0
    B: Final = 0.08
    C: Final = 3.0
    Q: Final = 0.80
    CBH: Final = 7.0
    CFL: Final = 1.8
    BUI0: Final = 62.0
    SFC_FACTOR: Final = 5.0
    SFC_POWER: Final = 2.48
    SFC_DECAY: Final = 0.0149

    class _CBH:
        """cbh() override constants."""

        HEIGHT: Final = 1.06
        DENSITY: Final = 0.0017
        UNIT_CONVERSION: Final = 10_000

        class Height:
            FACTOR: Final = 6.18
            THRESHOLD: Final = 0.3

    class _RSC:
        """rsc() override constants."""

        FACTOR: Final = 60.0
        DECAY: Final = 0.0497
        DENOMINATOR: Final = 0.778

    class _FME:
        """fme() override constants."""

        POWER: Final = 4
        SCALE: Final = 1000
        CONSTANT: Final = 1.5
        FACTOR: Final = 0.00275

        class Denom:
            FACTOR: Final = 25.9
            CONSTANT: Final = 460

    def cbh(
        self,
        density: np.ndarray = _CBH.DENSITY,
        height: np.ndarray = _CBH.HEIGHT,
        default: np.ndarray = CBH,
    ) -> np.ndarray:
        """Calculate Crown Base Height (CBH) for the C6 model using density and height."""
        _height = self._CBH.Height
        _conv = self._CBH.UNIT_CONVERSION

        valid = (default >= self.EPSILON) & (default <= self.CBH_MAX) & ~np.isnan(default)

        spacing = np.maximum(1.0, _height.THRESHOLD - density / _conv)
        cbh = _height.FACTOR * np.sqrt(spacing) * np.sqrt(height)

        cbh = np.where(valid, default, cbh)
        return np.clip(cbh, self.EPSILON, self.CBH_MAX)

    def cfb(self, rss: np.ndarray, rso: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        C6 Crown Fraction Burned.

        Uses C6 gate:
          if (RSC > RSS) & (RSS > RSO): CFB = 1 - exp(-0.23 * (RSS - RSO))
          else: CFB = 0

        Parameters
        ----------
        rss: surface rate of spread (Eq. 63)
        rso: critical surface rate of spread (Eq. 57)
        *args: (rsc) crown head-rate (Eq. 64 with FME/0.778) — provide either as:
              - exactly one positional extra arg, OR
              - keyword argument rsc=...

        Returns
        -------
        np.ndarray in [0, 1]
        """
        try:
            _rsc = self._cfb_parse(args, kwargs)
        except TypeError as e:
            raise TypeError(e)

        _gate = (_rsc > rss) & (rss > rso)
        _delta = np.maximum(0.0, rss - rso)
        _cfb = 1.0 - np.exp(-self.CFB_DECAY * _delta)

        _cfb = np.where(_gate, _cfb, 0.0)

        _cfb = np.nan_to_num(_cfb, posinf=1.0, neginf=0.0)
        _result = np.clip(_cfb, 0.0, 1.0)
        return np.asarray(_result, dtype=float)

    def ros(self, bui: np.ndarray, isi: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Rate of Spread for C6 model.
        NOTE:
            C6 model has a distinct ROS computation relative to other models.

        Eq.65) C6 Rate of Spread:
          if RSC > RSS:
            ROS = RSS + CFB * (RSC - RSS)
          else:
            ROS = RSS

        Args
        ----
            bui: Buildup Index (Eq. 60)
            isi: Initial Spread Index (Eq. 61)
            *args: (fmc) Fuel Moisture Content (Eq. 62) — provide either as:
                   - exactly one positional extra arg, OR
                   - keyword argument fmc=...
            **kwargs: (fmc) Fuel Moisture Content (Eq. 62) — provide either as:
                   - exactly one positional extra arg, OR
                   - keyword argument fmc=...
        Returns
        -------
            np.ndarray
        """
        if args:
            (fmc,) = args
        elif "fmc" in kwargs:
            fmc = kwargs["fmc"]
        else:
            raise TypeError("C6.ros() missing required argument: 'fmc'")

        _csi = self.csi(fmc)
        _sfc = self.sfc(bui)
        _rsc = self.rsc(isi, fmc)
        _rss = self.rss(bui, isi)
        _rso = self.rso(_csi, _sfc)
        _cfb = self.cfb(_rss, _rso, rsc=_rsc)

        _ros = _rss + _cfb * (_rsc - _rss)
        return np.where(_rsc > _rss, _ros, _rss)

    def rsc(self, isi: np.ndarray, fmc: np.ndarray) -> np.ndarray:
        """
        Rate of spread crown.
        NOTE:
            Unique requirement for C6 ROS computation.

        Eq. 64: RSC = FACTOR * (1 - exp(-DECAY * ISI)) * FME / DENOMINATOR
        """
        _rsc = self._RSC.FACTOR * (1.0 - np.exp(-self._RSC.DECAY * isi))
        return (_rsc * self._fme(fmc)) / self._RSC.DENOMINATOR

    def _fme(self, fmc: np.ndarray) -> np.ndarray:
        """
        Foliar moisture effect.

        Eq. 66: FME = SCALE * ((CONSTANT - FACTOR * FMC) ** POWER) / (Denom.CONSTANT + Denom.FACTOR * FMC)
        """
        numer = (self._FME.CONSTANT - self._FME.FACTOR * fmc) ** self._FME.POWER
        denom = self._FME.Denom.CONSTANT + self._FME.Denom.FACTOR * fmc
        return (numer / denom) * self._FME.SCALE

    @staticmethod
    def _cfb_parse(args: tuple, kwargs: dict):
        rsc = None
        # ---- fetch rsc from args/kwargs (but not both) ----
        rsc_provided_pos = len(args)
        rsc_provided_kwd = "rsc" in kwargs
        if rsc_provided_pos + int(rsc_provided_kwd) == 0:
            raise TypeError(
                "C6.cfb() requires RSC: pass as a single extra positional arg or as rsc=..."
            )
        if rsc_provided_pos + int(rsc_provided_kwd) > 1:
            raise TypeError("C6.cfb(): provide RSC either positionally or as rsc=..., not both")
        if rsc_provided_pos == 1:
            rsc = args[0]
        elif rsc_provided_kwd:
            rsc = kwargs["rsc"]  # ignore other kwargs if present
        return rsc


class C7(_Conifer):
    """Ponderosa Pine - Douglas-Fir fuel model."""

    CODE: Final = Code.C7
    DESCRIPTION: Final = Description.C7
    A: Final = 45.0
    B: Final = 0.0305
    C: Final = 2.0
    Q: Final = 0.85
    BUI0: Final = 106.0
    CBH: Final = 10.0
    CFL: Final = 0.50

    class SFC:
        class BUI:
            FACTOR: Final = 1.5
            DECAY: Final = 0.0201

        class FFMC:
            FACTOR: Final = 2.0
            DELTA: Final = 70.0
            DECAY: Final = 0.104

    def sfc(
        self, bui: Optional[np.ndarray], ffmc: Optional[np.ndarray], *args, **kwargs
    ) -> np.ndarray:
        """
        Surface fire consumption for C7 (Eq. 12 from FCFDG 1992).

        Eq. 12: SFC = SFC_FFMC + SFC_BUI

        Args:
            bui: Building index.
            ffmc: Fine fuel moisture code.

        Returns:
            Surface fuel consumption (SFC).
        """
        _delta = ffmc - self.SFC.FFMC.DELTA

        _consumption = np.maximum(0.0, 1.0 - np.exp(-self.SFC.FFMC.DECAY * _delta))
        _ffmc = self.SFC.FFMC.FACTOR * _consumption

        _consumption = 1.0 - np.exp(-self.SFC.BUI.DECAY * bui)
        _bui = self.SFC.BUI.FACTOR * _consumption

        _consumption = _ffmc + _bui
        _result = np.maximum(_consumption, self.EPSILON)
        return np.asarray(_result, dtype=float)
