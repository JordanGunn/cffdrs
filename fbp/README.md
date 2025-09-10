# Fire Behavior Prediction (FBP) Module

Canadian Forest Fire Danger Rating System (CFFDRS) Fire Behavior Prediction implementation.

## Functions

### Core Calculations
- **`fd(code, bui, isi, lat=46.0, modifier=None)`** - Fire Description classification
- **`hfi(code, ros, ffmc, bui, isi, lat=46.0, modifier=None)`** - Head Fire Intensity (kW/m)
- **`ros(code, bui, isi, lat=46.0, modifier=None)`** - Rate of Spread (m/min)

### Parameters
- **`code`**: Fuel type (`Code` enum)
- **`bui`**: Buildup Index
- **`isi`**: Initial Spread Index  
- **`ffmc`**: Fine Fuel Moisture Code (for HFI)
- **`ros`**: Rate of Spread (for HFI)
- **`lat`**: Latitude in degrees (default: 46.0)
- **`modifier`**: Optional fuel modifier (pc/pdf/cc by fuel type)

## Classes

### `Metric` (metric.py)
```python
class Metric(StrEnum):
    FD = "fd"      # Fire Description
    ROS = "ros"    # Rate of Spread  
    HFI = "hfi"    # Head Fire Intensity
    
    @classmethod
    def all() -> tuple[str, ...]
    
    @classmethod  
    def exists(code: str | Metric) -> bool
    
    @classmethod
    def cast(code: str) -> Metric
```

## Submodules

### `fuel/`
Fuel type definitions and model implementations:
- **`Code`**: Fuel type enumeration (C1-C7, D1-D2, M1-M4, O1A/O1B, S1-S3)
- **`Model`**: Abstract fuel model interface
- **`model/`**: Fuel-specific implementations (conifer, deciduous, mixed, slash, grass)

## Usage Examples

### Basic Calculations
```python
import cffdrs
from cffdrs.fbp.fuel import Code

# Rate of Spread
rate = cffdrs.fbp.ros(Code.C2, bui=50.0, isi=8.0, lat=55.0)

# Head Fire Intensity  
intensity = cffdrs.fbp.hfi(Code.C2, ros=rate, ffmc=85.0, bui=50.0, isi=8.0)

# Fire Description
description = cffdrs.fbp.fd(Code.C2, bui=50.0, isi=8.0, lat=55.0)
```

### Vectorized Processing
```python
import numpy as np
import cffdrs
from cffdrs.fbp.fuel import Code

# Multiple scenarios
bui_vals = np.array([30.0, 50.0, 80.0])
isi_vals = np.array([5.0, 8.0, 12.0])

ros_results = cffdrs.fbp.ros(Code.C3, bui=bui_vals, isi=isi_vals)
```

### Fuel Modifiers
```python
import cffdrs
from cffdrs.fbp.fuel import Code

# M1/M2: pc (percentage conifer, default 80.0)
ros_m1 = cffdrs.fbp.ros(Code.M1, bui=45.0, isi=7.0, modifier=60.0)

# M3/M4: pdf (percentage dead balsam fir, default 35.0)  
ros_m3 = cffdrs.fbp.ros(Code.M3, bui=55.0, isi=9.0, modifier=25.0)

# O1A/O1B: cc (crown closure, default 50.0)
ros_o1a = cffdrs.fbp.ros(Code.O1A, bui=40.0, isi=6.0, modifier=70.0)
```

## Special Cases

### C6 Fuel Type
Requires Foliar Moisture Content (FMC) calculation from latitude:
```python
import cffdrs
from cffdrs.fbp.fuel import Code

# C6 automatically calculates FMC from lat parameter
ros_c6 = cffdrs.fbp.ros(Code.C6, bui=70.0, isi=15.0, lat=49.0)
```

## Integration

- Uses **FWI components** (FFMC, BUI, ISI) as inputs
- Supports **NumPy vectorization** for batch processing  
- **Type-safe** with comprehensive annotations
- **Fuel model factory** pattern for extensibility
