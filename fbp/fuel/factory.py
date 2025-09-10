from __future__ import annotations

from typing import List, Optional

from .code import Code
from .lut import LUT
from .model import Model


def model(
    code: Code,
    modifier: Optional[float] = None,
) -> Model:
    """
    Factory function to create appropriate fuel model implementation.

    Optionally accepts a `modifier` for `Mixed` and `Grass` types to adjust their properties.
    If no `modifier` is provided, the default values for the model will be used.

    Args:
        code (Code): The fuel code representing the desired fuel model.
        modifier (Optional[float]): An optional percentage modifier for `Mixed` and `Grass` types.
            - For `M1` and `M2`, this represents the percentage of conifer (pc) in the mixedwood.
            - For `M3` and `M4`, this represents the percentage of dead balsam fir (pdf) in the mixedwood.
            - For `O1A` and `O1B`, this represents the crown closure percentage (cc).

    Returns:
        Model: Instantiated fuel model object.

    Raises:
        ValueError: If the fuel code is not supported.
    """
    if not Code.exists(code):
        raise ValueError(f"Invalid fuel code: {code}")

    _model_class = LUT.get(code)

    # Handle different constructor signatures based on fuel type
    if modifier is None:
        # No modifier provided - use default constructor
        return _model_class()

    # Determine parameter name based on fuel code
    if code in [Code.M1, Code.M2]:
        # Mixed models M1/M2 expect 'pc' parameter
        return _model_class(pc=modifier)
    elif code in [Code.M3, Code.M4]:
        # Mixed models M3/M4 expect 'pdf' parameter
        return _model_class(pdf=modifier)
    elif code in [Code.O1A, Code.O1B]:
        # Grass models expect 'cc' parameter
        return _model_class(cc=modifier)
    else:
        # Other fuel types (Conifer, Deciduous, Slash) don't accept modifiers
        # Ignore the modifier and use default constructor
        return _model_class()


def models(
    codes: List[Code],
    pc: Optional[float] = None,
    pdf: Optional[float] = None,
    cc: Optional[float] = None,
) -> List[Model]:
    """
    Generate multiple computational fuel models from a list of provided fuel codes.

    Args:
        codes (List[Code]): List of fuel codes representing the desired fuel models.
        pc (Optional[float]): Optional percentage modifier for mixed fuel types.
        pdf (Optional[float]): Optional percentage modifier for dead balsam fir mixed fuel types.
        cc (Optional[float]): Optional percentage modifier for grass fuel types.

    Returns:
        List[Model]: List of instantiated fuel model objects.
    """
    _grass = _grasses(codes, cc)
    _mixed = _mixeds(codes, pc, pdf)

    _exclude = [Code.mixed(), Code.grass()]
    _models = [model(code) for code in codes if code not in _exclude]

    return [*_models, *_grass, *_mixed]


def _grasses(codes: List[Code], cc: Optional[float] = None):
    """Build grass fuel types based on provided codes."""
    _ccs = Code.grass()
    _grass = [model(code, modifier=cc) for code in codes if code in _ccs]
    return _grass


def _mixeds(
    codes: List[Code], pc: Optional[float] = None, pdf: Optional[float] = None
) -> List[Model]:
    """Build mixed fuel types based on provided codes."""
    _pcs = [Code.M1, Code.M2]
    _mixed1 = [model(code, modifier=pc) for code in codes if code in _pcs]

    _pdfs = [Code.M3, Code.M4]
    _mixed2 = [model(code, modifier=pdf) for code in codes if code in _pdfs]

    _mixed = _mixed1 + _mixed2
    return _mixed
