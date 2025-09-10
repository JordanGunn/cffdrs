from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from rasterio.profiles import Profile

from loki.api.cffdrs import fwi as cffdrs

#: Shorthand type-ref
Array = np.ndarray


def _broadcast(x: Optional[Union[float, Array]], shape: tuple[int, int], default: float) -> Array:
    if x is None:
        x = default

    _64 = np.float64
    return (
        np.full(shape, float(x), dtype=_64)
            if np.isscalar(x)
            else np.asarray(x, _64)
    )


class FWI:
    """
    MODERNIZED: Raster-native FWI engine with xarray interface (lazy computations, clean I/O).

    BREAKING CHANGES:
    - Now accepts xarray.DataArray inputs for geospatial correctness
    - Extracts numpy arrays for CFFDRS computation, wraps results back to DataArrays
    - Preserves coordinate systems and spatial metadata
    - Maintains Dask compatibility through chunking
    - Single-day FWI calculations (multi-day looping handled at Pipeline level)

    Usage:
    - Initialize with weather DataArrays (temp, rh, wind, precip) + seed values
    - Properties compute lazily and return xarray.DataArrays with proper coordinates
    - compute_single_day() returns 7-band FWI DataArray with all components
    - Designed for Pipeline integration and future FBP extension
    """

    BAND_ORDER: Tuple[str, ...] = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR")

    def __init__(
        self,
        temp: Union[Array, xr.DataArray],
        rh: Union[Array, xr.DataArray],
        ws: Union[Array, xr.DataArray],
        prec: Union[Array, xr.DataArray],
        month: int = datetime.now().month,
        ffmc0: Optional[Union[float, Array, xr.DataArray]] = None,
        dmc0: Optional[Union[float, Array, xr.DataArray]] = None,
        dc0: Optional[Union[float, Array, xr.DataArray]] = None,
    ):
        # Find DataArray with spatial metadata - fail fast if none found
        self._template_da = None
        for _param_name, input_data in [("temp", temp), ("rh", rh), ("ws", ws), ("prec", prec)]:
            if isinstance(input_data, xr.DataArray) and hasattr(input_data, "rio"):
                try:
                    # Verify it has proper spatial metadata
                    _ = input_data.rio.crs
                    _ = input_data.rio.transform()
                    self._template_da = input_data
                    break
                except Exception:
                    # Failed to extract spatial metadata; try next parameter
                    continue  # Try next parameter

        if self._template_da is None:
            raise ValueError(
                "At least one weather parameter must be a rioxarray DataArray with spatial metadata (CRS and transform). "
                "Ensure your input arrays have been loaded with rioxarray.open_rasterio() or similar."
            )

        # Extract numpy arrays for CFFDRS computation
        rh_arr = self._extract_numpy(rh)
        ws_arr = self._extract_numpy(ws)
        temp_arr = self._extract_numpy(temp)
        prec_arr = self._extract_numpy(prec)

        self._validate_shapes(temp_arr, rh_arr, ws_arr, prec_arr)

        # Store numpy arrays for CFFDRS functions
        self.temp = temp_arr.astype(np.float64)
        self.rh = np.clip(rh_arr.astype(np.float64), 0.0, 99.9999)  # R-compatible clamp
        self.wind = ws_arr.astype(np.float64)
        self.precip = prec_arr.astype(np.float64)

        self.month = int(month)

        # Auto-derive latitude array from spatial coordinates
        h, w = temp_arr.shape
        self.lat = self._lats((h, w))

        # Store seed values (must be provided - no defaults)
        # Seeds are required for operational use and must come from station data interpolation
        self._ffmc0 = ffmc0  # Required: FFMC initialization values
        self._dmc0 = dmc0  # Required: DMC initialization values
        self._dc0 = dc0  # Required: DC initialization values

        # lazy caches for xarray results
        self._ffmc: Optional[xr.DataArray] = None
        self._dmc: Optional[xr.DataArray] = None
        self._dc: Optional[xr.DataArray] = None
        self._isi: Optional[xr.DataArray] = None
        self._bui: Optional[xr.DataArray] = None
        self._fwi: Optional[xr.DataArray] = None
        self._dsr: Optional[xr.DataArray] = None

    @property
    def ffmc(self) -> xr.DataArray:
        if self._ffmc is None:
            if self._ffmc0 is None:
                raise ValueError("FFMC seeds (ffmc0) are required; no defaults allowed")
            h, w = self.temp.shape
            ffmc0 = _broadcast(self._extract_numpy(self._ffmc0), (h, w), 0.0)
            # CFFDRS computation with numpy arrays
            ffmc_result = cffdrs.ffmc(self.temp, self.rh, self.wind, self.precip, ffmc0)

            # Wrap result back to DataArray
            self._ffmc = self._darr(ffmc_result, "FFMC")
        return self._ffmc

    @property
    def dmc(self) -> xr.DataArray:
        if self._dmc is None:
            if self._dmc0 is None:
                raise ValueError("DMC seeds (dmc0) are required; no defaults allowed")
            h, w = self.temp.shape
            dmc0 = _broadcast(self._extract_numpy(self._dmc0), (h, w), 0.0)

            # CFFDRS computation with numpy arrays
            dmc_result = cffdrs.dmc(self.temp, self.precip, self.rh, dmc0, self.month, self.lat)

            # Wrap result back to DataArray
            self._dmc = self._darr(dmc_result, "DMC")
        return self._dmc

    @property
    def dc(self) -> xr.DataArray:
        if self._dc is None:
            if self._dc0 is None:
                raise ValueError("DC seeds (dc0) are required; no defaults allowed")
            h, w = self.temp.shape
            dc0 = _broadcast(self._extract_numpy(self._dc0), (h, w), 0.0)
            # CFFDRS computation with numpy arrays
            dc_result = cffdrs.dc(self.temp, self.precip, dc0, self.month, self.lat)
            # Wrap result back to DataArray
            self._dc = self._darr(dc_result, "DC")
        return self._dc

    @property
    def isi(self) -> xr.DataArray:
        if self._isi is None:
            # ISI depends on FFMC - extract numpy from DataArray result
            ffmc_values = self.ffmc.values  # Get numpy array from DataArray
            # CFFDRS computation with numpy arrays
            isi_result = cffdrs.isi(ffmc_values, self.wind)
            # Wrap result back to DataArray
            self._isi = self._darr(isi_result, "ISI")
        return self._isi

    @property
    def bui(self) -> xr.DataArray:
        if self._bui is None:
            # BUI depends on DMC and DC - extract numpy from DataArray results
            dmc_values = self.dmc.values  # Get numpy array from DataArray
            dc_values = self.dc.values  # Get numpy array from DataArray
            # CFFDRS computation with numpy arrays
            bui_result = cffdrs.bui(dc_values, dmc_values)
            # Wrap result back to DataArray
            self._bui = self._darr(bui_result, "BUI")
        return self._bui

    @property
    def fwi(self) -> xr.DataArray:
        if self._fwi is None:
            # FWI depends on ISI and BUI - extract numpy from DataArray results
            isi_values = self.isi.values  # Get numpy array from DataArray
            bui_values = self.bui.values  # Get numpy array from DataArray
            # CFFDRS computation with numpy arrays
            fwi_result = cffdrs.fwi(bui_values, isi_values)
            # Wrap result back to DataArray
            self._fwi = self._darr(fwi_result, "FWI")
        return self._fwi

    @property
    def dsr(self) -> xr.DataArray:
        if self._dsr is None:
            # DSR depends on FWI - extract numpy from DataArray result
            fwi_values = self.fwi.values  # Get numpy array from DataArray
            # CFFDRS computation with numpy arrays
            dsr_result = cffdrs.dsr(fwi_values)
            # Wrap result back to DataArray
            self._dsr = self._darr(dsr_result, "DSR")
        return self._dsr

    @classmethod
    def from_weather_dataarray(
        cls,
        weather_dataset: xr.DataArray,
        seeds: Optional[xr.DataArray] = None,
        month: int = datetime.now().month,
    ) -> "FWI":
        """
        Create FWI engine from weather DataArray with optional seed raster.

        Args:
            weather_dataset: 4-band weather data (temp, rh, wind, precip)
            seeds: Optional 3-band seed raster (FFMC0, DMC0, DC0). If None, triggers weather-derived seeding
            month: Current month for seasonal calculations

        Returns:
            FWI engine instance with weather-derived or station-interpolated seeds
        """
        # Extract weather bands as xr.DataArrays to preserve spatial metadata
        if len(weather_dataset.band) != 4:
            raise ValueError(
                f"Expected 4-band weather data (temp, RH, wind, precip), got {len(weather_dataset.band)} bands"
            )

        # Extract weather bands preserving rioxarray spatial metadata
        # CRITICAL: isel operations can strip rioxarray metadata, preserve explicitly
        temp_da = weather_dataset.isel(band=0)  # Keep as DataArray
        if hasattr(weather_dataset, "rio") and weather_dataset.rio.crs is not None:
            temp_da = temp_da.rio.write_crs(weather_dataset.rio.crs)
            temp_da = temp_da.rio.write_transform(weather_dataset.rio.transform())

        rh_da = weather_dataset.isel(band=1)  # Keep as DataArray
        if hasattr(weather_dataset, "rio") and weather_dataset.rio.crs is not None:
            rh_da = rh_da.rio.write_crs(weather_dataset.rio.crs)
            rh_da = rh_da.rio.write_transform(weather_dataset.rio.transform())

        wind_da = weather_dataset.isel(band=2)  # Keep as DataArray
        if hasattr(weather_dataset, "rio") and weather_dataset.rio.crs is not None:
            wind_da = wind_da.rio.write_crs(weather_dataset.rio.crs)
            wind_da = wind_da.rio.write_transform(weather_dataset.rio.transform())

        precip_da = weather_dataset.isel(band=3)  # Keep as DataArray
        if hasattr(weather_dataset, "rio") and weather_dataset.rio.crs is not None:
            precip_da = precip_da.rio.write_crs(weather_dataset.rio.crs)
            precip_da = precip_da.rio.write_transform(weather_dataset.rio.transform())

        # Apply necessary unit conversions while preserving rioxarray spatial metadata
        wind_kmh_da = wind_da * 3.6  # Convert m/s to km/h

        # Restore rioxarray spatial metadata after arithmetic operation
        if hasattr(wind_da, "rio") and wind_da.rio.crs is not None:
            wind_kmh_da = wind_kmh_da.rio.write_crs(wind_da.rio.crs)
            wind_kmh_da = wind_kmh_da.rio.write_transform(wind_da.rio.transform())

        # Clean negative precipitation
        precip_clean_da = precip_da.where(precip_da >= 0, 0)

        # Restore rioxarray spatial metadata after where operation
        if hasattr(precip_da, "rio") and precip_da.rio.crs is not None:
            precip_clean_da = precip_clean_da.rio.write_crs(precip_da.rio.crs)
            precip_clean_da = precip_clean_da.rio.write_transform(precip_da.rio.transform())

        # Use raw (unit-converted/cleaned) weather data for FWI calculations
        temp_da, rh_da, wind_kmh_da, precip_clean_da = temp_da, rh_da, wind_kmh_da, precip_clean_da

        # Extract seed values if provided, otherwise None (triggers weather-derived)
        if seeds is not None:
            # Extract seed values from 3-band DataArray
            ffmc0 = seeds.isel(band=0).values  # FFMC0
            dmc0 = seeds.isel(band=1).values  # DMC0
            dc0 = seeds.isel(band=2).values  # DC0
        else:
            # No seeds provided → not allowed (no defaults)
            raise ValueError(
                "FWI requires station-interpolated seeds (FFMC0, DMC0, DC0); no defaults allowed"
            )

        # Create FWI engine with DataArrays that have spatial metadata
        fwi_engine = cls(
            temp=temp_da,  # Pass DataArray with rioxarray metadata
            rh=rh_da,  # Pass DataArray with rioxarray metadata
            ws=wind_kmh_da,  # Pass DataArray with rioxarray metadata
            prec=precip_clean_da,  # Pass DataArray with rioxarray metadata
            month=month,
            ffmc0=ffmc0,  # Station seeds or None (weather-derived)
            dmc0=dmc0,  # Station seeds or None (weather-derived)
            dc0=dc0,  # Station seeds or None (weather-derived)
        )

        # Store template DataArray for coordinate preservation
        fwi_engine._template_da = weather_dataset

        return fwi_engine

    def compute(self) -> xr.DataArray:
        """
        MODERNIZED: Compute all FWI components for a single day and return as 7-band DataArray.

        This is the main method for single-day FWI calculations, replacing the old compute() method.
        Returns a multi-band DataArray with proper coordinates and CRS information.

        Returns:
            xr.DataArray with 7 bands: [FFMC, DMC, DC, ISI, BUI, FWI, DSR]
        """
        # Compute all components (triggers lazy evaluation)
        components = {
            "FFMC": self.ffmc,
            "DMC": self.dmc,
            "DC": self.dc,
            "ISI": self.isi,
            "BUI": self.bui,
            "FWI": self.fwi,
            "DSR": self.dsr,
        }

        # 🔧 ROBUSTNESS FIX: Use explicit BAND_ORDER to ensure consistent band ordering
        # Stack into multi-band array using defined band order
        fwi_stack = np.stack([components[band].values for band in self.BAND_ORDER]).astype(
            np.float32
        )

        # Create multi-band DataArray with proper coordinates
        if self._template_da is not None:
            fwi_da = xr.DataArray(
                fwi_stack,
                dims=["band", "y", "x"],
                coords={
                    "x": self._template_da.x,
                    "y": self._template_da.y,
                    "band": list(self.BAND_ORDER),
                },
                attrs={
                    "description": "Fire Weather Index components (7-band)",
                    "crs": (
                        str(self._template_da.rio.crs)
                        if hasattr(self._template_da, "rio")
                        else None
                    ),
                    "transform": (
                        self._template_da.rio.transform()
                        if hasattr(self._template_da, "rio")
                        else None
                    ),
                },
            )

            # Set CRS using rioxarray if available
            fwi_da = fwi_da.rio.write_crs(self._template_da.rio.crs)
            fwi_da = fwi_da.rio.write_transform(self._template_da.rio.transform())

            # Apply chunking for distributed processing
            return fwi_da.chunk({"x": 1024, "y": 1024})
        else:
            # Fallback without coordinates
            return xr.DataArray(
                fwi_stack,
                dims=["band", "y", "x"],
                coords={"band": list(components.keys())},
                attrs={"description": "Fire Weather Index components (7-band)"},
            )

    @staticmethod
    def _validate_shapes(*arrays: Array) -> None:
        """Ensure all arrays have the same shape."""
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1:
            raise ValueError(f"All weather arrays must share the same HxW; got {shapes}")

    @staticmethod
    def _extract_numpy(data: Union[Array, xr.DataArray]) -> Array:
        """Extract numpy array from DataArray or return array as-is."""
        if isinstance(data, xr.DataArray):
            return data.values
        return np.asarray(data)

    def _darr(self, data: Array, name: str) -> xr.DataArray:
        """Wrap numpy array result back to DataArray with proper coordinates."""
        if self._template_da is None:
            # Fallback: create basic DataArray without coordinates
            return xr.DataArray(data, dims=["y", "x"])

        # Use template DataArray coordinates and attributes

        attrs = {
            "description": f"Fire Weather Index component: {name}",
            "crs": str(self._template_da.rio.crs),
            "transform": self._template_da.rio.transform(),
        }
        coords = {"x": self._template_da.x, "y": self._template_da.y}
        darr = xr.DataArray(data, coords=coords, dims=["y", "x"], attrs=attrs)

        return darr.rio.write_crs(self._template_da.rio.crs)

    def _lats(self, shape: tuple[int, int]) -> np.ndarray:
        """Auto-derive latitude array from spatial coordinates in template DataArray."""
        if self._template_da is None:
            raise ValueError("No spatial metadata available to derive latitude array.")

        height, width = shape
        transform = self._template_da.rio.transform()

        # CRITICAL FIX: Proper latitude calculation using rasterio transform
        # For geographic data, transform[4] is usually negative (y decreases down rows)
        # transform[5] is the y-coordinate (latitude) of the upper-left corner
        # Need to properly convert row indices to latitude coordinates

        # Create row indices for all pixels (0 to height-1)
        row_indices = np.arange(height)

        # Calculate latitude for each row using proper transform
        # For each row: lat = upper_left_lat + (row + 0.5) * pixel_height
        # Note: transform[4] is typically negative for geographic data
        row_latitudes = transform[5] + (row_indices + 0.5) * transform[4]

        # Broadcast to full array shape (height, width)
        lats = np.broadcast_to(row_latitudes[:, np.newaxis], (height, width))

        return lats.astype(np.float64)

    def _extract_profile_from_template(self, shape: tuple[int, int]) -> Profile:
        """Extract rasterio profile from template DataArray with proper shape."""
        height, width = shape

        profile = {
            "driver": "GTiff",
            "dtype": str(self._template_da.dtype),
            "nodata": self._template_da.rio.nodata,
            "width": width,
            "height": height,
            "count": 1,  # Single band output
            "crs": self._template_da.rio.crs,
            "transform": self._template_da.rio.transform(),
            "compress": "deflate",
            "predictor": 3,
        }
        return Profile(profile)

