from typing import Dict, Type

from .code import Code
from .model import C1, C2, C3, C4, C5, C6, C7, D1, M1, M2, M3, M4, O1A, O1B, S1, S2, S3, Model

#: LUT is a lookup table mapping fuel codes to their respective model classes.
LUT: Dict[Code, Type[Model]] = {
    Code.C1: C1,
    Code.C2: C2,
    Code.C3: C3,
    Code.C4: C4,
    Code.C5: C5,
    Code.C6: C6,
    Code.C7: C7,
    Code.D1: D1,
    Code.S1: S1,
    Code.S2: S2,
    Code.S3: S3,
    Code.M1: M1,
    Code.M2: M2,
    Code.M3: M3,
    Code.M4: M4,
    Code.O1A: O1A,
    Code.O1B: O1B,
}
