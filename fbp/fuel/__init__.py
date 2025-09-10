"""Fuel Module
===========

This module provides fuel type definitions and classifications for fire behavior modeling.

Components
---------
* Fuel codes and standardized fuel type identifiers
* Fuel models for different vegetation types (Mixed, Slash, Grass, Conifer, Deciduous)
* Utilities for working with fuel types in fire behavior calculations

The fuel types and models are essential inputs for fire behavior prediction
calculations and fire spread modeling.
"""

from . import model
from .code import Code
from .factory import model, models
from .model import Model

enums = [
    "Code",
]

modules = [
    "model",
]

funcs = [
    "model",
    "models",
]

__all__ = [
    *enums,
    *modules,
    *funcs,
]
