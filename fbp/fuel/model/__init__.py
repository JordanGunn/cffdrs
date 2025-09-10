from .abstract import Model
from .conifer import C1, C2, C3, C4, C5, C6, C7
from .deciduous import D1
from .grass import O1A, O1B
from .mixed import M1, M2, M3, M4
from .slash import S1, S2, S3

ABSTRACT = [
    "Model",
]

CONIFER = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
]

DECIDUOUS = [
    "D1",
]

SLASH = [
    "S1",
    "S2",
    "S3",
]

GRASS = [
    "O1A",
    "O1B",
]

MIXED = [
    "M1",
    "M2",
    "M3",
    "M4",
]

MODELS = [
    *MIXED,
    *SLASH,
    *GRASS,
    *CONIFER,
    *DECIDUOUS,
]

__all__ = [
    *MODELS,
    *ABSTRACT,
]
