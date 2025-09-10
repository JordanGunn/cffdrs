# CFFDRS

The Canadian Forest Fire Danger Rating System (CFFDRS) API provides comprehensive fire weather and fire behavior prediction capabilities. This package implements the complete CFFDRS calculation suite including the Fire Weather Index (FWI) System and Fire Behavior Prediction (FBP) System.

## Installation

This package can be installed using Poetry:

```bash
# Clone the repository
git clone https://github.com/JordanGunn/cffdrs.git
cd cffdrs

# Install with Poetry
poetry install

# Or install with pip (in development mode)
pip install -e .
```

### Dependencies

- Python 3.8.1+
- NumPy 1.24.0+
- Numba 0.58.0+

## Overview

The CFFDRS API is organized into several key components:

- **FWI System** (`fwi/`): Fire Weather Index calculations including moisture codes and fire behavior indices
- **FBP System** (`fbp/`): Fire Behavior Prediction calculations for rate of spread, fire intensity, and fire description
- **Raster Operations** (`raster/`): Spatial grid-based calculations for landscape-scale fire modeling (requires additional dependencies)
- **Parameters** (`param.py`): Common parameter definitions and validation

## Quick Start

```python
import cffdrs

# FWI System calculations
ffmc_result = cffdrs.fwi.ffmc(temp=20.0, rh=45.0, ws=15.0, prec=0.0, ffmc0=85.0)
dmc_result = cffdrs.fwi.dmc(temp=20.0, prec=0.0, rh=45.0, dmc0=25.0, month=7)
dc_result = cffdrs.fwi.dc(temp=20.0, prec=0.0, dc_prev=150.0, month=7)

# Calculate fire behavior indices
isi_result = cffdrs.fwi.isi(ffmc=ffmc_result, ws=15.0)
bui_result = cffdrs.fwi.bui(dmc=dmc_result, dc=dc_result)
fwi_result = cffdrs.fwi.fwi(isi=isi_result, bui=bui_result)

print(f"FFMC: {ffmc_result:.2f}")
print(f"FWI: {fwi_result:.2f}")

# FBP System calculations (requires fuel type)
from cffdrs.fbp.fuel import Code

ros_result = cffdrs.fbp.ros(code=Code.C2, bui=bui_result, isi=isi_result)
hfi_result = cffdrs.fbp.hfi(code=Code.C2, ros=ros_result, ffmc=ffmc_result, 
                            bui=bui_result, isi=isi_result)

print(f"Rate of Spread: {ros_result:.2f} m/min")
print(f"Head Fire Intensity: {hfi_result:.2f} kW/m")
```

## Architecture

The CFFDRS API follows a modular design with clear separation between:

1. **Core Calculations**: Individual calculation modules for each CFFDRS component
2. **Data Types**: Structured data containers for inputs, outputs, and intermediate results
3. **Vectorized Operations**: NumPy-based implementations for efficient batch processing
4. **Parameter Management**: Centralized parameter definitions and validation

## Key Features

- **Complete CFFDRS Implementation**: All standard FWI and FBP calculations
- **Vectorized Processing**: Efficient NumPy-based operations for large datasets
- **Type Safety**: Comprehensive type annotations and data validation
- **Modular Design**: Clean separation of concerns following Single Responsibility Principle
- **Performance Optimized**: Cached properties and lazy evaluation for complex calculations

## Modules

### FWI System (`cffdrs.fwi`)
Fire Weather Index System calculations including:
- Fine Fuel Moisture Code (FFMC)
- Duff Moisture Code (DMC) 
- Drought Code (DC)
- Initial Spread Index (ISI)
- Buildup Index (BUI)
- Fire Weather Index (FWI)
- Daily Severity Rating (DSR)

### FBP System (`cffdrs.fbp`)
Fire Behavior Prediction System calculations including:
- Rate of Spread (ROS)
- Head Fire Intensity (HFI)
- Fire Description (FD)
- Fuel-specific behavior modeling

### Fuel Types (`cffdrs.fbp.fuel`)
Canadian Forest Service fuel type definitions:
- Conifer fuels (C1-C7)
- Deciduous fuels (D1-D2)
- Mixed fuels (M1-M4)
- Grass fuels (O1A, O1B)
- Slash fuels (S1-S3)

### Raster Operations (`cffdrs.raster`)
Spatial grid-based processing for:
- Landscape-scale fire modeling
- Spatial interpolation
- Grid-based calculations
- Map generation support

#### Note: About Raster operations 
This was preserved as part of a larger project, and as such, is not recommended for use yet. While they do work, additional type-safety constraints should be added to prevent the passing of malformatted data or incorrect band ordering. 

The functions were designed with full `dask` compatibility in mind, and as such, require additional dependencies (`xarray`, `rasterio`, `rioxarray`). This was implemented to enable distributed computing
for rapid computation of fire-weather and fuel-behaviour rasters over very large geographic areas (and works very well).

In summary, these functions work, but I would like to add some additional type-safety constraints to make them more user-friendly.

## Standards Compliance

This implementation follows the official CFFDRS standards as defined by:
- Natural Resources Canada
- Canadian Forest Service
- Provincial fire management agencies

All calculations are validated against reference implementations and standard test cases.

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black .
poetry run isort .
```

### Type Checking

```bash
poetry run mypy .
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.
