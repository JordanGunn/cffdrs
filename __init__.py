"""
Canadian Forest Fire Danger Rating System (CFFDRS) implementation.

Provides comprehensive CFFDRS calculations including FWI (Fire Weather Index),
FBP (Fire Behaviour Prediction), and raster-based spatial processing.
"""

from . import fbp, fwi, raster
from .metric import Metric
from .param import Param, broadcast, to_array

# Grouped exports
types = ["Param", "Metric"]
funcs = ["broadcast", "to_array"]
modules = ["fwi", "fbp", "raster"]

__all__ = [*types, *funcs, *modules]
