from cffdrs.metric import Metric as Base


class Metric(Base):
    """
    FWI metric codes.

    Attributes
    ----------
    DC   : str
        Drought Code
    DMC  : str
        Duff Moisture Code
    FWI  : str
        Fire Weather Index
    ISI  : str
        Initial Spread Index
    BUI  : str
        Buildup Index
    DSR  : str
        Daily Severity Index
    FFMC : str
        Fine Fuel Moisture Code
    """

    DC = "dc"
    DMC = "dmc"
    FWI = "fwi"
    ISI = "isi"
    BUI = "bui"
    DSR = "dsr"
    FFMC = "ffmc"
