# FBP Fuel Model Implementations

Concrete fuel-specific model implementations for Canadian fire behavior prediction.

## Abstract Base

### `Model` (__init__.py)
```python
class Model(ABC):
    CODE: Code  # Fuel type identifier
    
    @abstractmethod
    def ros(self, bui: Param, isi: Param, fmc: Param = None) -> np.ndarray
    
    @abstractmethod
    def hfi(self, ros: Param, tfc: Param) -> np.ndarray
    
    @abstractmethod
    def sfc(self, **kwargs) -> np.ndarray
    
    @abstractmethod
    def tfc(self, cfb: Param, sfc: Param) -> np.ndarray
```

## Coniferous Models (conifer.py)

### C1 - Spruce-Lichen Woodland
```python
class C1(Model):
    CODE = Code.C1
    # Light surface fire, slow spread
```

### C2 - Boreal Spruce
```python
class C2(Model):
    CODE = Code.C2
    # Moderate surface fire
```

### C3 - Mature Jack/Lodgepole Pine
```python
class C3(Model):
    CODE = Code.C3
    # Fast surface fire, high intensity
```

### C4 - Immature Jack/Lodgepole Pine
```python
class C4(Model):
    CODE = Code.C4
    # Moderate surface fire, lower intensity than C3
```

### C5 - Red and White Pine
```python
class C5(Model):
    CODE = Code.C5
    # Moderate surface fire
```

### C6 - Conifer Plantation
```python
class C6(Model):
    CODE = Code.C6
    # Crown fire capable, requires FMC parameter
    def ros(self, bui: Param, isi: Param, fmc: Param) -> np.ndarray
```

### C7 - Ponderosa Pine - Douglas-fir
```python
class C7(Model):
    CODE = Code.C7
    # Surface fire, moderate spread
```

## Deciduous Models (deciduous.py)

### D1 - Leafless Aspen
```python
class D1(Model):
    CODE = Code.D1
    # Surface fire, spring conditions
```

### D2 - Green Aspen (with understory)
```python
class D2(Model):
    CODE = Code.D2
    # Surface fire with understory effects
```

## Mixed Wood Models (mixed.py)

### M1 - Boreal Mixed Wood - Leafless
```python
class M1(Model):
    CODE = Code.M1
    # Uses percentage conifer (pc) modifier
    # Default pc = 80.0
```

### M2 - Boreal Mixed Wood - Green
```python
class M2(Model):
    CODE = Code.M2
    # Uses percentage conifer (pc) modifier
    # Default pc = 80.0
```

### M3 - Dead Balsam Fir Mixed Wood - Leafless
```python
class M3(Model):
    CODE = Code.M3
    # Uses percentage dead balsam fir (pdf) modifier
    # Default pdf = 35.0
```

### M4 - Dead Balsam Fir Mixed Wood - Green
```python
class M4(Model):
    CODE = Code.M4
    # Uses percentage dead balsam fir (pdf) modifier
    # Default pdf = 35.0
```

## Slash Models (slash.py)

### S1 - Jack/Lodgepole Pine Slash
```python
class S1(Model):
    CODE = Code.S1
    # High intensity surface fire
```

### S2 - White Spruce - Balsam Slash
```python
class S2(Model):
    CODE = Code.S2
    # Very high intensity surface fire
```

### S3 - Coastal Cedar - Hemlock - Douglas-fir Slash
```python
class S3(Model):
    CODE = Code.S3
    # High intensity surface fire
```

## Grass Models (grass.py)

### O1A - Matted Grass (standing)
```python
class O1A(Model):
    CODE = Code.O1A
    # Uses crown closure (cc) modifier
    # Default cc = 50.0
```

### O1B - Matted Grass (matted)
```python
class O1B(Model):
    CODE = Code.O1B
    # Uses crown closure (cc) modifier
    # Default cc = 50.0
```

## Model Calculations

### Rate of Spread (ROS)
All models implement fuel-specific rate of spread calculations:
- **Input**: BUI (Buildup Index), ISI (Initial Spread Index)
- **Optional**: FMC (Foliar Moisture Content) for C6
- **Output**: Rate of spread in m/min

### Head Fire Intensity (HFI)
Calculates fire line intensity:
- **Input**: ROS, TFC (Total Fuel Consumption)
- **Output**: Fire intensity in kW/m

### Surface Fuel Consumption (SFC)
Fuel-specific surface consumption:
- **Input**: Varies by fuel type
- **Output**: Surface fuel consumed in kg/m²

### Total Fuel Consumption (TFC)
Combined surface and crown consumption:
- **Input**: CFB (Crown Fraction Burned), SFC
- **Output**: Total fuel consumed in kg/m²

## Modifier Handling

### Percentage Conifer (pc) - M1, M2
```python
# Default: 80% conifer
m1_model = model(Code.M1, modifier=60.0)  # 60% conifer
```

### Percentage Dead Balsam Fir (pdf) - M3, M4
```python
# Default: 35% dead balsam fir
m3_model = model(Code.M3, modifier=25.0)  # 25% dead balsam fir
```

### Crown Closure (cc) - O1A, O1B
```python
# Default: 50% crown closure
o1a_model = model(Code.O1A, modifier=70.0)  # 70% crown closure
```

## Vectorization Support

All models support vectorized NumPy operations:
```python
import numpy as np

# Vectorized inputs
bui_array = np.array([30.0, 50.0, 80.0])
isi_array = np.array([5.0, 8.0, 12.0])

# Vectorized calculation
ros_results = c2_model.ros(bui_array, isi_array)
# Returns: array([2.1, 4.8, 9.2])  # Example values
```

## Special Model Features

### C6 Crown Fire Capability
- Only fuel type requiring FMC parameter
- Implements crown fire transition logic
- Enhanced rate of spread for active crown fire

### Mixed Wood Complexity
- M1/M2: Conifer percentage affects fire behavior
- M3/M4: Dead balsam fir percentage influences spread
- Dynamic fuel loading based on composition

### Grass Model Variations
- O1A: Standing grass conditions
- O1B: Matted grass conditions  
- Crown closure affects wind reduction

## Integration Notes

- **Factory Creation**: Use `fuel.model(code, modifier)` function
- **Type Safety**: All models validate input parameters
- **Error Handling**: Invalid parameters raise appropriate exceptions
- **Performance**: Optimized for batch processing with NumPy arrays
