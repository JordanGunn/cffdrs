# FBP Fuel Module

Canadian fuel type definitions and model implementations for fire behavior prediction.

## Classes

### `Code` (code.py)
Fuel type enumeration for all Canadian fuel types:
```python
class Code(StrEnum):
    # Coniferous
    C1 = "C1"  # Spruce-Lichen Woodland
    C2 = "C2"  # Boreal Spruce
    C3 = "C3"  # Mature Jack/Lodgepole Pine
    C4 = "C4"  # Immature Jack/Lodgepole Pine
    C5 = "C5"  # Red and White Pine
    C6 = "C6"  # Conifer Plantation
    C7 = "C7"  # Ponderosa Pine - Douglas-fir
    
    # Deciduous
    D1 = "D1"  # Leafless Aspen
    D2 = "D2"  # Green Aspen (with understory)
    
    # Mixed
    M1 = "M1"  # Boreal Mixed Wood - Leafless
    M2 = "M2"  # Boreal Mixed Wood - Green
    M3 = "M3"  # Dead Balsam Fir Mixed Wood - Leafless
    M4 = "M4"  # Dead Balsam Fir Mixed Wood - Green
    
    # Slash
    S1 = "S1"  # Jack/Lodgepole Pine Slash
    S2 = "S2"  # White Spruce - Balsam Slash
    S3 = "S3"  # Coastal Cedar - Hemlock - Douglas-fir Slash
    
    # Grass
    O1A = "O1A"  # Matted Grass (standing)
    O1B = "O1B"  # Matted Grass (matted)
    
    @classmethod
    def all() -> tuple[str, ...]
    
    @classmethod
    def exists(code: str) -> bool
    
    @classmethod
    def cast(code: str) -> Code
```

### `Model` (model/__init__.py)
Abstract base class for fuel-specific behavior models:
```python
class Model(ABC):
    CODE: Code
    
    @abstractmethod
    def ros(self, bui: Param, isi: Param, fmc: Param = None) -> np.ndarray
    
    @abstractmethod
    def hfi(self, ros: Param, tfc: Param) -> np.ndarray
    
    @abstractmethod
    def sfc(self, **kwargs) -> np.ndarray
    
    @abstractmethod
    def tfc(self, cfb: Param, sfc: Param) -> np.ndarray
```

### `Description` (description.py)
Fuel type descriptions and metadata:
```python
class Description(StrEnum):
    C1 = "Spruce-Lichen Woodland"
    C2 = "Boreal Spruce"
    # ... etc
```

## Functions

### `model(code, modifier=None)` (factory.py)
Factory function to create fuel model instances:
```python
def model(code: Code, modifier: float = None) -> Model:
    """Create fuel model instance for given code."""
```

### `models(codes, modifier=None)` (factory.py)
Create multiple fuel model instances:
```python
def models(codes: list[Code], modifier: float = None) -> list[Model]:
    """Create multiple fuel model instances."""
```

## Submodules

### `model/`
Fuel-specific model implementations:

#### `conifer.py`
- **C1**: Spruce-Lichen Woodland
- **C2**: Boreal Spruce  
- **C3**: Mature Jack/Lodgepole Pine
- **C4**: Immature Jack/Lodgepole Pine
- **C5**: Red and White Pine
- **C6**: Conifer Plantation (crown fire capable)
- **C7**: Ponderosa Pine - Douglas-fir

#### `deciduous.py`
- **D1**: Leafless Aspen
- **D2**: Green Aspen (with understory)

#### `mixed.py`
- **M1**: Boreal Mixed Wood - Leafless (pc modifier)
- **M2**: Boreal Mixed Wood - Green (pc modifier)
- **M3**: Dead Balsam Fir Mixed Wood - Leafless (pdf modifier)
- **M4**: Dead Balsam Fir Mixed Wood - Green (pdf modifier)

#### `slash.py`
- **S1**: Jack/Lodgepole Pine Slash
- **S2**: White Spruce - Balsam Slash
- **S3**: Coastal Cedar - Hemlock - Douglas-fir Slash

#### `grass.py`
- **O1A**: Matted Grass (standing) (cc modifier)
- **O1B**: Matted Grass (matted) (cc modifier)

## Usage Examples

### Basic Model Creation
```python
from loki.api.cffdrs.fbp.fuel import Code, model

# Create C2 fuel model
c2_model = model(Code.C2)

# Calculate rate of spread
ros_result = c2_model.ros(bui=50.0, isi=8.0)
```

### Models with Modifiers
```python
# M1 with custom conifer percentage
m1_model = model(Code.M1, modifier=60.0)  # 60% conifer

# O1A with custom crown closure
o1a_model = model(Code.O1A, modifier=70.0)  # 70% crown closure
```

### Multiple Models
```python
from loki.api.cffdrs.fbp.fuel import models

# Create multiple models
fuel_codes = [Code.C1, Code.C2, Code.M1]
fuel_models = models(fuel_codes)
```

## Modifier Types

### Percentage Conifer (pc)
- **Fuels**: M1, M2
- **Default**: 80.0
- **Range**: 0-100
- **Description**: Percentage of coniferous species in mixedwood

### Percentage Dead Balsam Fir (pdf)
- **Fuels**: M3, M4  
- **Default**: 35.0
- **Range**: 0-100
- **Description**: Percentage of dead balsam fir in mixedwood

### Crown Closure (cc)
- **Fuels**: O1A, O1B
- **Default**: 50.0
- **Range**: 0-100
- **Description**: Percentage crown closure in open woodland

## Special Cases

### C6 Crown Fire Model
- Requires **Foliar Moisture Content (FMC)** for crown fire calculations
- Uses enhanced rate of spread equations for crown fire behavior
- Automatically handled by model implementation

### D2 Understory Model
- Includes understory vegetation effects
- Different behavior than D1 leafless aspen

## Integration

- **Factory Pattern**: Consistent model creation via `model()` function
- **Polymorphic Interface**: All models implement same `Model` interface
- **Type Safety**: Fuel codes validated via `Code` enum
- **Vectorized**: All models support NumPy array inputs
