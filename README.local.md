# CFFDRS API

The Canadian Forest Fire Danger Rating System (CFFDRS) API provides comprehensive fire weather and fire behavior prediction capabilities. This module implements the complete CFFDRS calculation suite including the Fire Weather Index (FWI) System and Fire Behavior Prediction (FBP) System.

## Overview

The CFFDRS API is organized into several key components:

- **FWI System** (`fwi/`): Fire Weather Index calculations including moisture codes and fire behavior indices
- **FBP System** (`fbp/`): Fire Behavior Prediction calculations for rate of spread, fire intensity, and fire description
- **Temporal Processing** (`temporal/`): Time-series processing and daily weather sequence handling
- **Raster Operations** (`raster/`): Spatial grid-based calculations for landscape-scale fire modeling
- **Parameters** (`param.py`): Common parameter definitions and validation

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

## Usage

```python
from loki.api.cffdrs import fwi, fbp

# FWI System calculations
fwi_result = fwi.calculate(
    temp=20.0, rh=45.0, wind=15.0, prec=0.0,
    ffmc_prev=85.0, dmc_prev=25.0, dc_prev=150.0
)

# FBP System calculations
fbp_result = fbp.calculate(
    fuel_codes=['C1', 'C2'], 
    weather_data=weather_inputs,
    modifiers={'pc': 50, 'pdf': 35, 'cc': 80}
)
```

## Modules

### FWI System (`fwi/`)
Fire Weather Index System calculations including:
- Fine Fuel Moisture Code (FFMC)
- Duff Moisture Code (DMC) 
- Drought Code (DC)
- Initial Spread Index (ISI)
- Buildup Index (BUI)
- Fire Weather Index (FWI)
- Daily Severity Rating (DSR)

### FBP System (`fbp/`)
Fire Behavior Prediction System calculations including:
- Rate of Spread (ROS)
- Head Fire Intensity (HFI)
- Fire Description (FD)
- Fuel-specific behavior modeling

### Temporal Processing (`temporal/`)
Time-series processing capabilities for:
- Daily weather sequence processing
- Historical data integration
- Seasonal adjustments
- Multi-day fire weather calculations

### Raster Operations (`raster/`)
Spatial grid-based processing for:
- Landscape-scale fire modeling
- Spatial interpolation
- Grid-based calculations
- Map generation support

## Dependencies

- NumPy: Vectorized numerical operations
- Numba: Just-in-time compilation for performance
- Pydantic: Data validation and serialization

## Standards Compliance

This implementation follows the official CFFDRS standards as defined by:
- Natural Resources Canada
- Canadian Forest Service
- Provincial fire management agencies

All calculations are validated against reference implementations and standard test cases.
