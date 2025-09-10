# FWI Module

Canadian Fire Weather Index System calculations for fire danger assessment.

## Functions

### Moisture Codes
```python
def ffmc(temp: Param, rh: Param, ws: Param, prec: Param, ffmc0: Param = 85.0) -> np.ndarray
def dmc(temp: Param, prec: Param, rh: Param, dmc0: Param = 6.0, month: Param = 7, lat: Param = 46.0) -> np.ndarray  
def dc(temp: Param, prec: Param, dc_prev: Param, month: Param, lat: Param = 46.0) -> np.ndarray
```

### Fire Behavior Indices
```python
def isi(ffmc: Param, ws: Param) -> np.ndarray
def bui(dmc: Param, dc: Param) -> np.ndarray
def fwi(isi: Param, bui: Param) -> np.ndarray
```

### Additional Components
```python
def dsr(fwi: Param) -> np.ndarray
def fmc(dc: Param, month: Param, lat: Param = 46.0) -> np.ndarray
```

## Classes

### `Metric` (metric.py)
FWI component enumeration:
```python
class Metric(StrEnum):
    FFMC = "ffmc"
    DMC = "dmc" 
    DC = "dc"
    ISI = "isi"
    BUI = "bui"
    FWI = "fwi"
    DSR = "dsr"
    FMC = "fmc"
```

## Usage Examples

### Basic Calculations
```python
import cffdrs

# Daily weather inputs
temp = 20.0      # Temperature (°C)
rh = 45.0        # Relative humidity (%)
ws = 15.0        # Wind speed (km/h)
prec = 0.0       # Precipitation (mm)

# Previous day values
ffmc0 = 85.0
dmc0 = 25.0
dc_prev = 200.0

# Calculate moisture codes
ffmc_today = cffdrs.fwi.ffmc(temp, rh, ws, prec, ffmc0)
dmc_today = cffdrs.fwi.dmc(temp, prec, rh, dmc0, month=7)
dc_today = cffdrs.fwi.dc(temp, prec, dc_prev, month=7)

# Calculate fire behavior indices
isi_today = cffdrs.fwi.isi(ffmc_today, ws)
bui_today = cffdrs.fwi.bui(dmc_today, dc_today)
fwi_today = cffdrs.fwi.fwi(isi_today, bui_today)
```

### Vectorized Processing
```python
import numpy as np
import cffdrs

# Multiple weather observations
temps = np.array([18.0, 22.0, 25.0])
rh_values = np.array([50.0, 40.0, 35.0])
winds = np.array([12.0, 18.0, 20.0])
precip = np.array([0.0, 2.5, 0.0])

# Previous values
ffmc0 = np.array([82.0, 85.0, 88.0])
dmc0 = np.array([20.0, 25.0, 30.0])
dc_prev = np.array([180.0, 200.0, 220.0])

# Batch calculations
ffmc_results = cffdrs.fwi.ffmc(temps, rh_values, winds, precip, ffmc0)
dmc_results = cffdrs.fwi.dmc(temps, precip, rh_values, dmc0, month=7)
dc_results = cffdrs.fwi.dc(temps, precip, dc_prev, month=7)
```

### Complete FWI System
```python
import cffdrs

def calculate_fwi_system(temp, rh, ws, prec, ffmc0, dmc0, dc_prev, month=7, lat=46.0):
    """Calculate complete FWI system for given weather conditions."""
    
    # Moisture codes
    ffmc_val = cffdrs.fwi.ffmc(temp, rh, ws, prec, ffmc0)
    dmc_val = cffdrs.fwi.dmc(temp, prec, rh, dmc0, month, lat)
    dc_val = cffdrs.fwi.dc(temp, prec, dc_prev, month, lat)
    
    # Fire behavior indices
    isi_val = cffdrs.fwi.isi(ffmc_val, ws)
    bui_val = cffdrs.fwi.bui(dmc_val, dc_val)
    fwi_val = cffdrs.fwi.fwi(isi_val, bui_val)
    
    # Additional metrics
    dsr_val = cffdrs.fwi.dsr(fwi_val)
    
    return {
        'ffmc': ffmc_val,
        'dmc': dmc_val,
        'dc': dc_val,
        'isi': isi_val,
        'bui': bui_val,
        'fwi': fwi_val,
        'dsr': dsr_val
    }
```

## Component Details

### FFMC - Fine Fuel Moisture Code
- **Range**: 0-101 (higher = drier)
- **Response**: Fast (hours)
- **Critical**: >85 for ignition potential
- **Inputs**: Temperature, humidity, wind, precipitation, previous FFMC

### DMC - Duff Moisture Code  
- **Range**: 0+ (higher = drier)
- **Response**: Moderate (days)
- **Purpose**: Fire intensity assessment
- **Inputs**: Temperature, humidity, precipitation, previous DMC, month, latitude

### DC - Drought Code
- **Range**: 0+ (higher = drier) 
- **Response**: Slow (weeks)
- **Purpose**: Season severity assessment
- **Inputs**: Temperature, precipitation, previous DC, month, latitude

### ISI - Initial Spread Index
- **Range**: 0+ (higher = faster spread)
- **Purpose**: Rate of spread potential
- **Inputs**: FFMC, wind speed

### BUI - Buildup Index
- **Range**: 0+ (higher = more fuel available)
- **Purpose**: Fuel availability assessment
- **Inputs**: DMC, DC

### FWI - Fire Weather Index
- **Range**: 0+ (logarithmic scale)
- **Purpose**: General fire intensity potential
- **Inputs**: ISI, BUI

### DSR - Daily Severity Rating
- **Range**: 0+ (exponential scale)
- **Purpose**: Fire suppression difficulty
- **Inputs**: FWI

### FMC - Foliar Moisture Content
- **Range**: 0-200% (lower = higher crown fire risk)
- **Purpose**: Crown fire potential assessment
- **Inputs**: DC, month, latitude

## Parameter Types

All functions accept `Param` type (Union[float, np.ndarray]) for vectorized operations:
- **Scalars**: Single weather observation
- **Arrays**: Multiple observations for batch processing

## Integration Notes

- **Dependency-free**: All functions accept primitive parameters
- **Vectorized**: NumPy array support for batch processing
- **Type-safe**: Comprehensive type annotations
- **Standards-compliant**: Validated against CFFDRS reference implementations
