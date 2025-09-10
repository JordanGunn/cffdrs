from enum import StrEnum
from typing import Union

from .description import Description


class Code(StrEnum):
    """
    Enumeration of all fuel type codes used in fire behavior modeling.

    This enum defines the standardized codes for all fuel types across
    different vegetation categories. Each code represents a specific fuel
    type with associated fire behavior characteristics.
    """

    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    C6 = "C6"
    C7 = "C7"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    D1 = "D1"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    O1A = "O1A"
    O1B = "O1B"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return tuple(cls)

    @classmethod
    def sort(cls, _codes: list[Union[str, "Code"]]) -> list[str]:
        return sorted(_codes, key=lambda c: c.name)

    @classmethod
    def exists(cls, code: str):
        return code in cls

    @classmethod
    def cast(cls, code: str) -> "Code":
        sep = code.split("-")
        if len(sep) > 1:
            code = f"{sep[0]}" f"{sep[1]}"

        code = code.upper()
        if code not in cls:
            raise ValueError(f"Unknown Fuel code: {code}")
        return cls(code)

    @classmethod
    def casts(cls, _codes: list[str]) -> list["Code"]:
        """
        Cast a list of string codes to Code enum.

        Parameters
        ----------
        _codes : list[str]
            List of string codes to cast.

        Returns
        -------
        list[Code]
            List of Code enum instances.
        """
        return [cls.cast(code) for code in _codes]

    @classmethod
    def conifer(cls) -> tuple[str, ...]:
        """
        Get all coniferous fuel attributes.

        Returns
        -------
        tuple[T]
            Tuple of coniferous fuel attributes (C1-C7)
        """
        return (cls.C1, cls.C2, cls.C3, cls.C4, cls.C5, cls.C6, cls.C7)

    @classmethod
    def mixed(cls) -> tuple[str, ...]:
        """
        Get all mixed fuel attributes.

        Returns
        -------
        tuple[T]
            Tuple of mixed fuel attributes (M1-M4)
        """
        return (cls.M1, cls.M2, cls.M3, cls.M4)

    @classmethod
    def deciduous(cls) -> tuple[str, ...]:
        """
        Get all deciduous fuel attributes.

        Returns
        -------
        tuple[T]
            Tuple of deciduous fuel attributes (D1)
        """
        return (cls.D1,)

    @classmethod
    def slash(cls) -> tuple[str, ...]:
        """
        Get all slash fuel attributes.

        Returns
        -------
        tuple[T]
            Tuple of slash fuel attributes (S1-S3)
        """
        return (cls.S1, cls.S2, cls.S3)

    @classmethod
    def grass(cls) -> tuple[str, ...]:
        """
        Get all grass fuel attributes.

        Returns
        -------
        tuple[T]
            Tuple of grass fuel attributes (O1A-O1B)
        """
        return (cls.O1A, cls.O1B)

    @classmethod
    def description(cls, code: str) -> str:
        _code = cls.cast(code)
        _desc = Description.dict()
        return _desc[_code]

    @classmethod
    def pc(cls) -> tuple[str, ...]:
        return cls.M1, cls.M2

    @classmethod
    def pdf(cls) -> tuple[str, ...]:
        return cls.M3, cls.M4

    @classmethod
    def cc(cls) -> tuple[str, ...]:
        return cls.grass()
