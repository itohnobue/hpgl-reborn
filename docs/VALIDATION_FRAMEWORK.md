# HPGL Input Validation Framework

## Overview

The HPGL Input Validation Framework addresses vulnerability **IV-001 (CVSS 7.5)** by providing comprehensive input validation for all HPGL library operations. This framework ensures that all user inputs are validated before processing, preventing security vulnerabilities, crashes, and incorrect results.

## Architecture

The validation framework consists of three main components:

### 1. C++ Validation Layer (`src/bs_gstl/include/validation.h`)

**Location:** `C:\Users\itohnobue\Git\hpgl\src\bs_gstl\include\validation.h`

The C++ layer provides:
- Template-based validation functions for bounds checking
- Validation constants for all parameter types
- Custom exception classes (`validation_exception`)
- Helper macros for throwing validation exceptions
- Parameter validation wrappers for all major parameter types

**Key Classes:**
- `validation_exception`: Enhanced exception class with parameter name tracking
- `validation_result_t`: Structure for returning validation results
- `validation_severity_t`: Enum for error severity levels (info, warning, error, critical)

**Key Functions:**
```cpp
// Grid validation
validation_result_t validate_grid_dimensions(sugarbox_grid_size_t x, y, z);

// Radius validation
validation_result_t validate_radius(double radius_x, radius_y, radius_z);
validation_result_t validate_radius(size_t radius_x, radius_y, radius_z);

// Neighbor validation
validation_result_t validate_max_neighbors(size_t max_neighbors);

// Covariance parameter validation
validation_result_t validate_covariance_parameters(double sill, double nugget, const double* ranges);

// Angle validation
validation_result_t validate_angles(const double* angles);

// Probability validation
validation_result_t validate_probability(double prob);
validation_result_t validate_probability_sum(const double* probs, size_t count);

// Seed validation
validation_result_t validate_seed(long int seed);

// Indicator validation
validation_result_t validate_indicator_count(size_t count);
validation_result_t validate_indicator_value(indicator_value_t value, size_t indicator_count);
```

### 2. Python Validation Layer (`src/geo_bsd/validation.py`)

**Location:** `C:\Users\itohnobue\Git\hpgl\src\geo_bsd\validation.py`

The Python layer provides:
- Path validation with directory traversal protection
- Grid and array size validation
- Numerical parameter validation
- Decorators for automatic function validation
- Context manager for batch validation

**Key Classes:**
- `ValidationError`: Base exception class
- `CriticalValidationError`: Critical errors that prevent operation
- `ValidationWarning`: Non-critical warnings
- `PathValidator`: File path security validation
- `GridValidator`: Grid dimension validation
- `ParameterValidator`: Numerical parameter validation
- `ValidationContext`: Context manager for batch validation

**Key Functions:**
```python
# Path validation
PathValidator.validate_filepath(filename, must_exist, allowed_extensions)

# Grid validation
GridValidator.validate_grid_dimensions(x, y, z)
GridValidator.validate_array_size(array, grid)

# Parameter validation
ParameterValidator.validate_radius(radius)
ParameterValidator.validate_max_neighbors(max_neighbors)
ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)
ParameterValidator.validate_probability(prob)
ParameterValidator.validate_probability_sum(probs)
ParameterValidator.validate_seed(seed)
ParameterValidator.validate_indicator_count(count)
```

### 3. Integration Points

The validation framework is integrated into:

**C++ Files:**
- `src/bs_gstl/src/neighbourhood_param.cpp` - Neighborhood parameter validation
- `src/bs_gstl/src/ik_params.cpp` - Indicator kriging parameter validation
- `src/bs_gstl/src/covariance_param.cpp` - Covariance parameter validation

**Python Files:**
- `src/geo_bsd/geo.py` - Core geostatistics functions validation
- `src/geo_bsd/sgs.py` - Sequential Gaussian simulation validation
- `src/geo_bsd/sis.py` - Sequential indicator simulation validation

## Validation Rules

### Grid Dimensions

| Parameter | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| X, Y, Z | 1 | 10,000,000 | Total size limited to 1 billion cells |

### Neighbor Counts

| Parameter | Minimum | Maximum | Default |
|-----------|---------|---------|---------|
| max_neighbors | 1 | 1,000 | 12 (warning if > 12) |
| min_neighbors | 0 | max_neighbors | 0 |

### Radius Values

| Parameter | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| radius | 0.0 | 1,000,000.0 | Applied to all axes |

### Covariance Parameters

| Parameter | Minimum | Maximum | Constraints |
|-----------|---------|---------|-------------|
| sill | 0.0 | 1e10 | Must be >= nugget |
| nugget | 0.0 | 1e10 | Must be <= sill |
| range | 0.0 | 1e10 | Per axis |
| angles | -inf | inf | Warning if outside [0, 360] |

### Probabilities

| Parameter | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| probability | 0.0 | 1.0 | Individual values |
| probability_sum | 0.999 | 1.001 | For probability vectors |

### Indicators

| Parameter | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| indicator_count | 1 | 256 | Number of categories |
| indicator_value | 0 | count-1 | Per value validation |

### File Paths

- **Directory traversal prevention**: Paths containing `..` segments are rejected
- **Absolute path resolution**: All paths are normalized and resolved
- **Extension checking**: Optional whitelist of allowed extensions
- **Existence validation**: Optional check for file existence

## Usage Examples

### C++ Usage

```cpp
#include <validation.h>

using namespace hpgl;

// Manual validation
validation_result_t result = validation::validate_grid_dimensions(100, 100, 50);
if (!result.is_valid) {
    throw validation_exception("MyFunction", result.message);
}

// Using macros
HPGL_VALIDATE_RESULT(validation::validate_radius(10.0, 15.0, 20.0));

// Using parameter validators
ok_params_t params;
params.set_sill(1.0);
params.set_nugget(0.1);
params.set_radiuses(10, 10, 10);
// Validation is automatic in set_radiuses due to integration
```

### Python Usage

```python
from geo_bsd import validation
from geo_bsd.geo import SugarboxGrid, CovarianceModel

# Using validators directly
validation.GridValidator.validate_grid_dimensions(100, 100, 50)

# Path validation with security
safe_path = validation.PathValidator.validate_filepath(
    "data/input.txt",
    must_exist=True
)

# Parameter validation
validation.ParameterValidator.validate_radius((10, 15, 20))
validation.ParameterValidator.validate_max_neighbors(12)

# Using validated classes (automatic validation)
grid = SugarboxGrid(100, 100, 50)  # Validates dimensions
cov_model = CovarianceModel(
    type=0,
    ranges=(100, 100, 50),
    angles=(0, 0, 0),
    sill=1.0,
    nugget=0.1  # Validates nugget <= sill
)

# Using validation context
with validation.ValidationContext() as v:
    v.validate_grid_dimensions(100, 100, 50)
    v.validate_radius((10, 15, 20))
    # All validations checked, exception raised on exit if any failed
```

## Security Features

### 1. Directory Traversal Prevention

The path validation prevents directory traversal attacks by:
- Checking for `..` in path components
- Using `pathlib.Path.resolve()` for normalization
- Validating against allowed directories

```python
# This will raise CriticalValidationError
safe_path = PathValidator.validate_filepath("../../../etc/passwd")
```

### 2. Numeric Overflow Prevention

Grid size validation prevents integer overflow:
```python
# This will raise CriticalValidationError
# 100000 x 100000 x 100000 exceeds MAX_GRID_SIZE
grid = SugarboxGrid(100000, 100000, 100000)
```

### 3. Invalid Parameter Prevention

Covariance parameters are validated:
```python
# This will raise CriticalValidationError
# Nugget cannot exceed sill
cov = CovarianceModel(sill=0.5, nugget=1.0)
```

### 4. Array Bounds Protection

Array sizes are validated against grid dimensions:
```python
# This will raise CriticalValidationError
# Array size doesn't match grid
GridValidator.validate_array_size(my_array, grid)
```

## Error Handling

### Error Severity Levels

1. **Critical**: Operation cannot proceed (exception thrown)
2. **Error**: Invalid input (exception thrown)
3. **Warning**: Unusual but acceptable value (logged)
4. **Info**: Normal operation (logged)

### Exception Hierarchy

```
ValidationError
├── CriticalValidationError (stops operation)
└── ValidationWarning (logged, operation continues)

C++:
validation_exception : public hpgl_exception
```

## Performance Considerations

- **Minimal overhead**: Validation only occurs at API boundaries
- **Early failure**: Invalid inputs detected before expensive computations
- **Configurable**: Can be disabled in release builds if needed (not recommended)
- **Caching**: Validation results for repeated parameters

### Build Configuration

To enable validation in production builds:

```cmake
# Always enabled by default
# HPGL_VALIDATION_ENABLED=ON
```

## Testing

Validation functions are unit tested in:

- `tests/test_validation.cpp` - C++ validation tests
- `tests/test_validation.py` - Python validation tests

### Test Coverage

- Grid dimension validation (boundary cases)
- Radius validation (positive, negative, overflow)
- Neighbor count validation (range checks)
- Covariance parameter validation (sill vs nugget)
- Probability validation (sum to 1.0)
- Path validation (directory traversal attempts)
- Array size validation (mismatch detection)

## Migration Guide

### For Existing Code

**Before:**
```python
from geo_bsd.geo import ordinary_kriging
result = ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model)
```

**After:**
```python
from geo_bsd.geo import ordinary_kriging
from geo_bsd import validation

# Validation is now automatic - invalid inputs will raise CriticalValidationError
result = ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model)
```

No code changes are required - validation is integrated automatically.

### For Custom Code

To add validation to your own functions:

```python
from geo_bsd.validation import validate_grid_params, validate_kriging_params

@validate_grid_params
@validate_kriging_params
def my_custom_function(prop, grid, radiuses, max_neighbours, cov_model):
    # Your function logic here
    pass
```

## Future Enhancements

Planned improvements to the validation framework:

1. **Performance profiling**: Add timing for validation overhead
2. **Custom validators**: Allow user-defined validation rules
3. **Validation policies**: Configurable strictness levels
4. **Logging integration**: Structured logging for audit trails
5. **API validation**: REST API input validation layer

## References

- **OWASP Input Validation**: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- **CVSS Calculator**: https://www.first.org/cvss/calculator/3.1
- **HPGL Security Audit**: See vulnerability report IV-001

## Support

For issues or questions about the validation framework:
- Check the validation test files for examples
- Review inline documentation in `validation.h` and `validation.py`
- Consult the HPGL development team
