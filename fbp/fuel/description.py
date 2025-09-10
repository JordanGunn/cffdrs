from enum import StrEnum


class Description(StrEnum):
    """
    Enumeration of fuel model descriptions used in fire behavior modeling.

    This enum defines the standardized descriptions for all fuel types,
    providing human-readable names that describe the specific vegetation
    characteristics of each fuel model. Enum attributes correspond directly
    to fuel codes for consistency.
    """

    C1 = "Spruce-Lichen Woodland"
    C2 = "Boreal Spruce"
    C3 = "Mature Jack or Lodgepole Pine"
    C4 = "Immature Jack or Lodgepole Pine"
    C5 = "Red and White Pine"
    C6 = "Conifer Plantation"
    C7 = "Ponderosa Pine / Douglas Fir"
    M1 = "Boreal Mixedwood - Leafless"
    M2 = "Boreal Mixedwood - Green"
    M3 = "Dead Balsam Fir Mixedwood - Leafless"
    M4 = "Dead Balsam Fir Mixedwood - Green"
    D1 = "Leafless Aspen"
    S1 = "Jack or Lodgepole Pine Slash"
    S2 = "White Spruce / Balsam Slash"
    S3 = "Coastal Cedar / Hemlock / Douglas-Fir Slash"
    O1A = "Matted Grass"
    O1B = "Standing Grass"

    @classmethod
    def dict(cls) -> dict[str, str]:
        return {code.name: code for code in cls}
