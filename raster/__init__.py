try:
    from .fbp import FBP
    from .fwi import FWI
    __all__ = ["FBP", "FWI"]
except ImportError:
    # Raster modules require additional dependencies (xarray, rasterio)
    # Skip if not available
    __all__ = []
