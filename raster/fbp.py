"""
MODERNIZED Fire Behavior Prediction (FBP) Raster Engine

This module implements a modern xarray-based FBP raster processing tool following
the successful patterns established in the FWI refactoring. It provides clean
separation of concerns, coordinate preservation, and seamless integration with
the FWI raster pipeline.

Key Features:
- xarray interface with coordinate preservation
- Lazy computation with Dask compatibility
- Vectorized FBP calculations (ROS, HFI, FD)
- String-to-numeric FD conversion for raster export
- Factory method integration with FWI outputs
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from loki.api.cffdrs.fbp import fd as fbp_fd
from loki.api.cffdrs.fbp import hfi as fbp_hfi

# Import vectorized FBP functions
from loki.api.cffdrs.fbp import ros as fbp_ros
from loki.api.cffdrs.fbp.fuel.code import Code

Array = np.ndarray


class FBP:
    """
    MODERNIZED: Raster-native FBP engine with xarray interface (following FWI patterns).

    This class provides Fire Behavior Prediction calculations for raster data using
    the same successful architecture patterns established in the FWI refactoring:
    - Accepts xarray.DataArray inputs for geospatial correctness
    - Extracts numpy arrays for CFFDRS computation, wraps results back to DataArrays
    - Preserves coordinate systems and spatial metadata
    - Maintains Dask compatibility through chunking
    - Single-day FBP calculations (multi-day looping handled at Pipeline level)

    Usage:
    - Initialize with FWI components + fuel code + optional modifier
    - Properties compute lazily and return xarray.DataArrays with proper coordinates
    - compute_single_day() returns 3-band FBP DataArray (ROS, HFI, FD_numeric)
    - Designed for Pipeline integration as natural successor to FWI processing
    """

    # Base band types for each fuel code
    BASE_BANDS: Tuple[str, ...] = ("ROS", "HFI", "FD")

    # FD string-to-numeric mapping for raster classification
    FD_MAP: Dict[str, int] = {
        "S": 1,  # Surface fire
        "I": 2,  # Intermittent crown fire
        "C": 3,  # Crown fire
    }

    def __init__(
        self,
        fwi_components: Dict[str, Array],
        fuel_codes: Union[Code, List[Code]],
        lat_array: Array,
        modifier: Optional[float] = None,
        template_da: Optional[xr.DataArray] = None,
    ):
        """
        Initialize FBP engine with FWI components and fuel parameters.

        Args:
            fwi_components: Dict of numpy arrays {"ffmc", "dmc", "dc", "isi", "bui"}
            fuel_codes: Single fuel code or list of fuel codes for fire behavior prediction
            lat_array: Latitude array for calculations
            modifier: Optional fuel modifier (defaults handled by API)
            template_da: Template DataArray for coordinate preservation
        """
        self.fwi_components = fwi_components

        # Normalize fuel codes to list for consistent internal handling
        if isinstance(fuel_codes, Code):
            self.fuel_codes = [fuel_codes]
            self.single_fuel = True  # Track for backward compatibility
        else:
            self.fuel_codes = fuel_codes
            self.single_fuel = False

        # Validate fuel codes
        if not self.fuel_codes:
            raise ValueError("At least one fuel code must be provided")
        if not all(isinstance(fc, Code) for fc in self.fuel_codes):
            raise ValueError("All fuel codes must be Code instances")

        self.lat_array = lat_array
        self.modifier = modifier
        self._template_da = template_da

        # Lazy computation flags
        self._fbp_computed = False

        # Cached results
        self._fbp_results = None

    @property
    def band_names(self) -> List[str]:
        """
        Generate dynamic band names based on fuel codes.

        Returns:
            List of band names for the output DataArray
        """
        if self.single_fuel:
            # Backward compatibility: single fuel code uses original band names
            return list(self.BASE_BANDS)
        else:
            # Multi-fuel: generate fuel-specific band names
            bands = []
            for fuel_code in self.fuel_codes:
                for band_type in self.BASE_BANDS:
                    bands.append(f"{band_type}_{fuel_code.name}")
            return bands

    @property
    def fuel_code(self) -> Code:
        """
        Backward compatibility property for single fuel code access.

        Returns:
            First fuel code (for backward compatibility with existing tests)
        """
        return self.fuel_codes[0]

    @classmethod
    def _extract_fwi_components(cls, fwi_dataset: xr.DataArray) -> Dict[str, np.ndarray]:
        """
        Extract FWI components from 7-band FWI DataArray.

        Args:
            fwi_dataset: 7-band FWI DataArray (FFMC, DMC, DC, ISI, BUI, FWI, DSR)

        Returns:
            Dict of numpy arrays with FWI components needed for FBP
        """
        if len(fwi_dataset.band) != 7:
            raise ValueError(
                f"Expected 7-band FWI data (FFMC, DMC, DC, ISI, BUI, FWI, DSR), got {len(fwi_dataset.band)} bands"
            )

        # Extract required components for FBP calculations
        components = {
            "ffmc": fwi_dataset.sel(band="FFMC").values.astype(np.float64),
            "dmc": fwi_dataset.sel(band="DMC").values.astype(np.float64),
            "dc": fwi_dataset.sel(band="DC").values.astype(np.float64),
            "isi": fwi_dataset.sel(band="ISI").values.astype(np.float64),
            "bui": fwi_dataset.sel(band="BUI").values.astype(np.float64),
        }

        return components

    @staticmethod
    def _create_latitude_array(shape: Tuple[int, int], dataset: xr.DataArray) -> np.ndarray:
        """
        Create latitude array from DataArray coordinates.

        Args:
            shape: Array shape (height, width)
            dataset: DataArray with coordinate information

        Returns:
            2D latitude array matching input shape
        """
        if "y" in dataset.coords:
            # Extract latitude from y coordinates (assuming geographic CRS)
            y_coords = dataset.y.values
            lat_array = np.broadcast_to(y_coords[:, np.newaxis], shape)
            return lat_array.astype(np.float64)
        else:
            # Fallback to default latitude if no coordinates
            return np.full(shape, 46.0, dtype=np.float64)

    @classmethod
    def from_fwi_dataarray(
        cls,
        fwi_dataset: xr.DataArray,
        fuel_codes: Union[Code, List[Code]],
        modifier: Optional[float] = None,
    ) -> "FBP":
        """
        Create FBP engine from FWI DataArray output and fuel parameters.

        This factory method follows the proven FWI refactoring patterns,
        extracting FWI components and setting up the FBP computation engine.

        Args:
            fwi_dataset: 7-band FWI DataArray from FWI raster tool
            fuel_codes: Single fuel code or list of fuel codes for fire behavior prediction
            modifier: Optional fuel modifier (let API handle defaults)

        Returns:
            Configured FBP engine ready for multi-fuel computation
        """
        # Extract FWI components using proven patterns
        fwi_components = cls._extract_fwi_components(fwi_dataset)

        # Create latitude array from coordinates
        shape = fwi_components["ffmc"].shape
        lat_array = cls._create_latitude_array(shape, fwi_dataset)

        # Create FBP engine with extracted components
        fbp_engine = cls(
            fwi_components=fwi_components,
            fuel_codes=fuel_codes,
            lat_array=lat_array,
            modifier=modifier,
            template_da=fwi_dataset,  # Store for coordinate preservation
        )

        return fbp_engine

    def _convert_fd_to_numeric(self, fd_strings: np.ndarray) -> np.ndarray:
        """
        Convert FD strings to integers for raster classification.

        Args:
            fd_strings: Array of FD strings ("S", "I", "C")

        Returns:
            Array of integers (1, 2, 3) for raster export
        """
        # Vectorized string-to-numeric conversion
        fd_numeric = np.zeros_like(fd_strings, dtype=np.int32)

        for string_val, numeric_val in self.FD_MAP.items():
            mask = fd_strings == string_val
            fd_numeric[mask] = numeric_val

        return fd_numeric

    def compute_single_day(self) -> xr.DataArray:
        """
        Compute single-day FBP components and return as N*3 band DataArray.

        This method follows the proven FWI patterns, computing all FBP components
        for each fuel code and wrapping them in a properly coordinated DataArray.

        Returns:
            N*3 band DataArray with fuel-specific ROS, HFI, FD_numeric bands and preserved coordinates.
            For single fuel: ["ROS", "HFI", "FD"] (backward compatible)
            For multi-fuel: ["ROS_C1", "HFI_C1", "FD_C1", "ROS_C2", "HFI_C2", "FD_C2", ...]
        """
        # Compute FBP components for all fuel codes
        components = self._compute_fbp_components()

        # Create individual DataArrays to preserve specific dtypes
        if self._template_da is not None:
            # Create base coordinates
            coords_2d = {"x": self._template_da.x, "y": self._template_da.y}

            # Create list of individual DataArrays for each component
            component_arrays = []
            for band_name in self.band_names:
                # Determine dtype based on band type
                if band_name.startswith("FD"):
                    dtype = np.int32
                else:
                    dtype = np.float32

                component_da = xr.DataArray(
                    components[band_name].astype(dtype), dims=["y", "x"], coords=coords_2d
                )
                component_arrays.append(component_da)

            # Combine into multi-band DataArray using concat
            fbp_da = xr.concat(component_arrays, dim="band")
            fbp_da = fbp_da.assign_coords(band=self.band_names)

            # Prepare fuel code description for metadata
            if self.single_fuel:
                fuel_desc = str(self.fuel_code)
                description = f"Fire Behavior Prediction components ({len(self.band_names)}-band)"
            else:
                fuel_desc = ", ".join(str(fc) for fc in self.fuel_codes)
                description = f"Multi-fuel Fire Behavior Prediction ({len(self.fuel_codes)} fuels, {len(self.band_names)} bands)"

            # Add attributes
            fbp_da.attrs.update(
                {
                    "description": description,
                    "fuel_codes": fuel_desc,
                    "modifier": self.modifier,
                    "num_fuel_codes": len(self.fuel_codes),
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
                }
            )

            # Set CRS using rioxarray if available
            if hasattr(self._template_da, "rio") and self._template_da.rio.crs is not None:
                fbp_da = fbp_da.rio.write_crs(self._template_da.rio.crs)
                fbp_da = fbp_da.rio.write_transform(self._template_da.rio.transform())

            # Apply chunking for distributed processing
            return fbp_da.chunk({"x": 1024, "y": 1024})
        else:
            # Fallback without coordinates - create array stack from components
            band_arrays = [components[band_name] for band_name in self.band_names]
            fbp_stack = np.stack(band_arrays, axis=0)

            return xr.DataArray(
                fbp_stack,
                dims=["band", "y", "x"],
                coords={"band": self.band_names},
                attrs={
                    "description": f"Fire Behavior Prediction components ({len(self.band_names)}-band)",
                    "fuel_codes": ", ".join(str(fc) for fc in self.fuel_codes),
                },
            )

    def _compute_fbp_components(self) -> Dict[str, np.ndarray]:
        """
        Compute FBP components using vectorized CFFDRS functions for all fuel codes.

        Uses the modernized vectorized FBP API to compute for each fuel code:
        - ROS: Rate of Spread (m/min)
        - HFI: Head Fire Intensity (kW/m)
        - FD: Fire Description ("S", "I", "C" → 1, 2, 3)

        Returns:
            Dict with fuel-specific keys: {"ROS_C1": array, "HFI_C1": array, "FD_C1": array, ...}
        """
        # Extract arrays for computation
        ffmc = self.fwi_components["ffmc"]
        bui = self.fwi_components["bui"]
        isi = self.fwi_components["isi"]

        # Initialize results dictionary
        results = {}

        # Compute FBP components for each fuel code
        for fuel_code in self.fuel_codes:
            # Determine band suffixes (backward compatibility)
            if self.single_fuel:
                ros_key = "ROS"
                hfi_key = "HFI"
                fd_key = "FD"
            else:
                ros_key = f"ROS_{fuel_code.name}"
                hfi_key = f"HFI_{fuel_code.name}"
                fd_key = f"FD_{fuel_code.name}"

            # Compute Rate of Spread using vectorized FBP function
            ros_result = fbp_ros(
                code=fuel_code, isi=isi, bui=bui, modifier=self.modifier, lat=self.lat_array
            )

            # Compute Head Fire Intensity using vectorized FBP function
            hfi_result = fbp_hfi(
                code=fuel_code,
                ros=ros_result,
                ffmc=ffmc,
                isi=isi,
                bui=bui,
                modifier=self.modifier,
                lat=self.lat_array,
            )

            # Compute Fire Description using vectorized FBP function
            fd_result_strings = fbp_fd(
                code=fuel_code,
                isi=isi,
                bui=bui,
                ffmc=ffmc,
                modifier=self.modifier,
                lat=self.lat_array,
            )

            # Convert FD strings to numeric for raster classification
            fd_result_numeric = self._convert_fd_to_numeric(fd_result_strings)

            # Store results with fuel-specific keys
            results[ros_key] = ros_result.astype(np.float32)
            results[hfi_key] = hfi_result.astype(np.float32)
            results[fd_key] = fd_result_numeric.astype(np.int32)

        return results
