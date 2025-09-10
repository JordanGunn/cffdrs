from loki.api.cffdrs.metric import Metric as Base


class Metric(Base):
    """
    Fire Behavior Prediction (FBP) metric codes.

    Attributes
    ----------
    FD  : str
        Fire Description
    ROS : str
        Rate of Spread
    HFI : str
        Head Fire Intensity
    """

    FD = "fd"
    ROS = "ros"
    HFI = "hfi"
