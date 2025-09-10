from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np

from cffdrs.param import Param

from ..code import Code
from ..description import Description


class Model(ABC):
    """
    Fuel model type.

    This is an abstract base class for all fuel models, providing a common interface
    and shared functionality for fuel-model-dependent properties and computations.

    Constants for fuel consumption model equations. Values originate from FCFDG 1992, Eq. 56-58).
        Attributes:
            CBH: float
                Default crown base height (CBH) for the model.
            CFL: float
                Default crown fuel load (CFL) for the model.
            EPSILON: Final[float]
                Small value to prevent division by zero or negative values.
            RSO_FACTOR: Final[float]
                Constant divisor for critical surface rate of spread (RSO).
            CSI_EXP: Final[float]
                Constant exponent for critical surface intensity (CSI) calculations.
            CSI_FMC_B: Final[float]
                Coefficient B for critical surface intensity based on foliar moisture content (FMC).
            CSI_FMC_A: Final[float]
                Coefficient A for critical surface intensity based on foliar moisture content (FMC).
            CSI_FACTOR: Final[float]
                Factor for scaling critical surface intensity (CSI).
            CBH_MAX: Final[float]
                Maximum crown base height (CBH) threshold for validation.
            CFL_MAX: Final[float]
                Maximum crown fuel load (CFL) threshold for validation.


    """

    CODE: Code
    DESCRIPTION: Description
    # --
    Q: float
    BUI0: float
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    CBH: float = 0.0
    CFL: float = 0.0
    # --
    CFL_MAX: Final[float] = 2.0
    CFB_DECAY: Final[float] = 0.23
    CBH_MAX: Final[float] = 50.0
    EPSILON: Final[float] = 1e-07
    RSO_FACTOR: Final[float] = 300.0
    HFI_FACTOR: Final[float] = 300.0

    def __init__(self, *args, **kwargs):
        """
        Initialize the model with optional parameters.

        The `modifier` keyword argument is used to set a specific percentage
        modifier for mixed (M1, M2, M3, M4) fuel models and (O1A, O1B) crown
        closure models.

        Passing a `modifier` is optional, and if not provided, a default one is provided.
        Notes that there is no consequence for providing a `modifier` to to a model that
        does not use it (e.g., C1, C2, C3, C4, ...), as it is ignored in these cases.

        This is typically represented by the idiomatic names:
            - pc: Percentage of crown fuel (default: 80.0) [M1, M2]
            - pdf: Percentage of dead balsam fir (default: 35.0) [M3, M4]
            - cc: Crown closure percentage (default: 50.0) [O1A, O1B]

        Keyword Args:
            modifier: float, optional

        Examples:
            Passing the pc modifier to M1 and M2 models:
            >>> pc: float = 50.0
            >>> model = model.M1(modifier=pc)
            >>> model = model.M2(modifier=pc)
            Passing the modifier with a random name to M1 and M2 models:
            >>> dead_fir_percentage: float = 35.0
            >>> model = model.M3(modifier=dead_fir_percentage)
            >>> model = model.M4(modifier=dead_fir_percentage)
            Passing the modifier to a model that does not use it (e.g., C1):
            >>> cc: float = 50.0
            >>> model = model.C1(modifier=cc)
            >>> # Computations will still be correct, since they are not used by this model
            Initializing a model with no modifier:
            >>> c1 = model.C1()  # No modifier needed.
            >>> m1 = model.M1()  # Default modifier (pc=50.0) is used.
        """
        pass

    class CSI:
        """Critical surface intensity (CSI) parameters."""

        A: Final[float] = 460.0
        B: Final[float] = 25.9
        EXP: Final[float] = 1.5
        FACTOR: Final[float] = 0.001

    class FD:
        """
        Fire description codes, numerics, and parameters.

        Attributes:
            SURFACE: float
                Crown fraction burned (CFB) threshold for surface fire (CFB < 0.10 → Surface).
            CROWN: float
                Crown fraction burned (CFB) threshold for crown fire (CFB ≥ 0.90 → Crown).
        """

        SURFACE: Final[float] = 0.10
        CROWN: Final[float] = 0.90

        class Surface:
            NUMERIC: Final[int] = 1
            ALPHA: Final[str] = "S"

        class Intermittent:
            NUMERIC: Final[int] = 2
            ALPHA: Final[str] = "I"

        class Crown:
            NUMERIC: Final[int] = 3
            ALPHA: Final[str] = "C"

        @classmethod
        def alphas(cls) -> list[str]:
            return [cls.Surface.ALPHA, cls.Intermittent.ALPHA, cls.Crown.ALPHA]

        @classmethod
        def numerics(cls) -> list[int]:
            return [cls.Surface.NUMERIC, cls.Intermittent.NUMERIC, cls.Crown.NUMERIC]

    @abstractmethod
    def sfc(self, *args, **kwargs) -> Param:
        """
        Compute Surface Fuel Consumption (SFC) for this fuel model.

        Args:
            *args: Fuel-specific parameters (e.g., ffmc, bui).
            **kwargs: Additional fuel-specific parameters.

        Returns:
            Surface Fuel Consumption (kg/m^2).
        """
        pass

    def rsi(self, isi: np.ndarray) -> np.ndarray:
        """
        Initial Rate of Spread Index (RSI) calculation.

        Eq.35: RSI = A * (1-exp(-B*ISI))^C

        Args:
            isi: Initial Spread Index
        """
        _rsi = self.A * (1.0 - np.exp(-self.B * isi)) ** self.C
        return np.asarray(_rsi, dtype=float)

    def be(self, bui: np.ndarray) -> np.ndarray:
        """
        BUI-Effect for surface fires.

        Eq.54: BE = exp(50 * log(Q) * (1 / BUI - 1 / BUI0))

        Args:
            bui: Buildup Index
        """
        _be = np.exp(50.0 * np.log(self.Q) * (1.0 / bui - 1.0 / self.BUI0))
        _result = np.where(bui > 0, _be, 1.0)
        return np.asarray(_result, dtype=float)

    def rss(self, bui: np.ndarray, isi: np.ndarray) -> np.ndarray:
        """
        Rate of Spread for surface fires.

        Eq.: RSS = BE * RSI

        Args:
            bui: Buildup Index
            isi: Initial Spread Index
        """
        _be = self.be(bui)
        _rsi = self.rsi(isi)
        _result = _be * _rsi
        return np.asarray(_result, dtype=float)

    def cbh(self, *args) -> Param:
        """
        Return CBH for mixed fuel types, validating against thresholds.

        Eq. 55: CBH = A + B * PC + C * PC²

        Args:
            custom:
                - Optional custom CBH value.
                - If not provided, the model's default CBH value is used.

        Returns
        -------
            Crown base height (CBH) in meters.
        """
        if not args:
            return np.array(self.CBH, dtype=float)
        (_cbh,) = args

        mask = (_cbh < self.EPSILON) | (_cbh > self.CBH_MAX) | np.isnan(_cbh)

        return np.where(mask, self.CBH, _cbh)

    def cfl(self, *args) -> Param:
        """
        Validate and/or return crown-fuel-load.

        Eq. 58: CFL = A + B * CBH + C * CBH²

        Args:
            custom:
                - Optional custom CFL value.
                - If not provided, the model's default CFL value is used.

        Returns
        -------
            Crown fuel load (CFL) in kg/m².
        """
        if not args:
            return np.array(self.CFL, dtype=float)
        (_cfl,) = args

        _invalid = (_cfl < self.EPSILON) | (_cfl > self.CFL_MAX) | np.isnan(_cfl)
        return np.where(_invalid, self.CFL, _cfl)

    def csi(self, fmc: np.ndarray) -> np.ndarray:  # noqa: N802
        """Critical surface intensity (Eq. 56) for this model.

        Args:
            fmc: Foliar moisture content (percent).

        Returns
        -------
            Critical surface fireline intensity (kW·m⁻¹).
        """
        _term = self.CSI.A + self.CSI.B * fmc
        _cbh = self.cbh()
        _result = self.CSI.FACTOR * (_term**self.CSI.EXP) * (_cbh**self.CSI.EXP)
        return np.asarray(_result, dtype=float)

    def rsc(self, isi: np.ndarray, fmc: np.ndarray) -> np.ndarray:
        """Rate of spread crown."""
        pass

    def rso(self, csi: np.ndarray, sfc: np.ndarray) -> np.ndarray:  # noqa: N802
        """
        Critical surface rate of spread.

        Eq. 57: RSO = CSI / (300 * SFC)

        Args:
            csi: Critical surface intensity.
            sfc: Surface fuel consumption. If ``None``, the model's default
                :pyfunc:`sfc` value is used (computed with typical parameters).

        Returns
        -------
            Critical surface rate of spread (m·min⁻¹).
        """
        _result = csi / (self.RSO_FACTOR * sfc)
        return np.asarray(_result, dtype=float)

    def cfc(self, cfb: np.ndarray) -> np.ndarray:  # noqa: N802
        """
        Crown fuel consumption (CFC).

        Eq. 66: CFC = CFL * CFB

        Args:
            cfb: Crown fraction burned (CFB) as a percentage (0-100).

        Returns
        -------
            Crown fuel consumption (CFC) in kg/m².
        """
        _cfl = self.cfl()
        _result = _cfl * cfb
        return np.asarray(_result, dtype=float)

    def cfb(self, ros: np.ndarray, rso: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Crown Fraction Burned (CFB).

        Eq. 58: CFB = 1 - exp(-CFB_DECAY * (ROS - RSO))

        Parameters
        ----------
        ros: Rate of spread.
        rso: Surface fire rate of spread.
        **kwargs: Ignored by the base implementation but allows subclasses (e.g., C6) to accept
                  additional parameters without breaking the common interface.

        Returns
        -------
        np.ndarray in [0, 1]
        """
        mask = ros > rso
        diff = np.maximum(0.0, ros - rso)
        cfb = np.zeros_like(ros)
        cfb[mask] = 1.0 - np.exp(-self.CFB_DECAY * diff[mask])

        return cfb

    def tfc(self, cfb: np.ndarray, sfc: np.ndarray) -> np.ndarray:  # noqa: N802
        """
        Total fuel consumption (TFC).

        Eq. 67: TFC = SFC + CFC

        Args:
            cfb: Crown fraction burned (CFB) as a percentage (0-100).
            sfc: Surface fuel consumption (SFC) in kg/m².

        Returns
        -------
            Total fuel consumption (TFC) in kg/m².
        """
        _cfc = self.cfc(cfb)
        _result = sfc + _cfc
        return np.asarray(_result, dtype=float)

    def ros(self, bui: np.ndarray, isi: np.ndarray, *args, **kwargs) -> Param:
        """
        Rate of spread for the model.
            - For All models except C6, ROS is the same as RSS.
            - For C6, the conifer C6 class handles model-specific behavior.

        Args:
            bui: Buildup index.
            isi: Initial spread index.
            *args: Ignored by the base implementation but allows subclasses (e.g., C6) to accept
                   additional parameters without breaking the common interface.
            **kwargs: Ignored by the base implementation but allows subclasses (e.g., C6) to accept
                      additional parameters without breaking the common interface.

        Returns
        -------
            Rate of spread (m·min⁻¹)
        """
        return self.rss(bui, isi)

    def hfi(self, ros: np.ndarray, tfc: np.ndarray) -> np.ndarray:
        """
        Head Fire Intensity (HFI).

        Eq. 68: HFI = 300 * TFC * ROS

        Args:
            ros: Rate of spread (m·min⁻¹).
            tfc: Total fuel consumption (TFC) in kg/m².

        Returns
        -------
            Head Fire Intensity (kW·m⁻¹)
        """
        _result = self.HFI_FACTOR * tfc * ros
        return np.asarray(_result, dtype=float)

    def fd(self, cfb: Param, *, numeric: bool = False) -> np.ndarray:
        """
        Fire Description (FD).
            - By default, returns "Surface", "Intermittent", or "Crown" (S, I, C).
            - If numeric=True, returns 1, 2, or 3 (cffdrs defined codes).

        Args:
            cfb: Crown fraction burned (CFB) as a percentage (0-100).
            numeric: If True, returns numeric codes (1, 2, or 3) instead of alpha codes (S, I, C).

        Returns
        -------
            Fire description (str or int)
        """
        _low, _high = self.FD.SURFACE, self.FD.CROWN
        _nums, _alphas = self.FD.numerics(), self.FD.alphas()
        _cfb = np.asarray(cfb, dtype=float)
        np.clip(_cfb, 0.0, 1.0, out=_cfb)

        idx = np.digitize(_cfb, bins=[_low, _high])
        idx = np.where(np.isnan(_cfb), 1, idx)

        _dtype = int if numeric else "U1"
        _codes = _nums if numeric else _alphas

        labels = np.asarray(_codes, dtype=_dtype)
        return labels[idx]
