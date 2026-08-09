# HPGL Reborn - High Performance Geostatistics Library (v2.0.6)

## Table of Contents

- [Description](#description)
  - [Algorithms](#algorithms)
  - [Geostatistics Glossary](#geostatistics-glossary)
  - [Covariance Models](#covariance-models)
  - [How to Use This Guide](#how-to-use-this-guide)
- [Requirements](#requirements)
- [Build Instructions](#build-instructions)
  - [Windows (MSBuild)](#windows-msbuild---recommended)
  - [Linux (CMake)](#linux-cmake)
  - [macOS Build](#macos-build)
  - [CMake Options](#cmake-options)
- [Usage](#usage)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
  - [Core Classes](#core-classes)
  - [Kriging Functions](#kriging-functions)
  - [Simulation Functions](#simulation-functions)
  - [I/O Functions](#io-functions)
  - [Utility Functions](#utility-functions)
  - [Submodules](#submodules)
- [Dataclass Property Reference](#dataclass-property-reference)
- [Common Use Cases](#common-use-cases)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Frequently Asked Questions](#frequently-asked-questions)
- [License](#license)

## Description

HPGL Reborn (High Performance Geostatistics Library) is a C++/Python library implementing geostatistical algorithms for spatial data analysis and reservoir modeling. The core computation engine is written in C++ for performance, with a Python interface for ease of use.

It is "Reborn" because it was updated, fixed and refactored to work with modern tech along with fixing dozens of bugs.

Originally developed at the Ufa Petroleum Institute, HPGL provides production-grade implementations of standard geostatistical methods as described in Deutsch & Journel's GSLIB.

### Algorithms

- **Kriging**: Simple Kriging (SK), Ordinary Kriging (OK), LVM Kriging (Locally Varying Mean)
- **Indicator Kriging**: Indicator Kriging (IK), Median Indicator Kriging
- **Cokriging**: Simple Cokriging Mark I (Markov Model 1) and Mark II
- **Simulation**: Sequential Gaussian Simulation (SGS), Sequential Indicator Simulation (SIS), General Truncated Gaussian Simulation (GTSIM)
- **Variogram Analysis**: Experimental variogram calculation, variogram search templates
- **Utilities**: CDF computation, property I/O (INC/GSLIB formats), mean calculation, Vertical Proportion Curves (VPC)

### Geostatistics Glossary

Key geostatistical terms used throughout this documentation:

| Term | Definition |
|------|-----------|
| **Kriging** | A family of geostatistical interpolation methods that predict values at unmeasured locations using weighted averages of nearby known samples, with weights derived from a spatial covariance model. Named after Danie Krige. |
| **Simple Kriging (SK)** | Kriging variant where the global mean is assumed known and constant over the entire domain. |
| **Ordinary Kriging (OK)** | Kriging variant that estimates a local mean from neighboring data; the most commonly used form. |
| **LVM Kriging** | Locally Varying Mean kriging — uses an auxiliary variable as a locally varying trend. |
| **Indicator Kriging (IK)** | Kriging applied to binary (0/1) indicator variables, used for categorical data or probability estimation. |
| **Cokriging** | Extension of kriging that uses a secondary (auxiliary) variable correlated with the primary variable to improve estimation. |
| **Sequential Gaussian Simulation (SGS)** | Simulation method that generates multiple equiprobable realizations by sequentially drawing from local Gaussian (normal) distributions conditioned on previously simulated values. |
| **Sequential Indicator Simulation (SIS)** | Simulation method for categorical variables using indicator transforms and sequential drawing from local probability distributions. |
| **GTSIM** | General Truncated Gaussian Simulation — simulation method using truncated Gaussian fields with threshold-based facies assignment. |
| **Variogram** | A function describing the spatial correlation structure of a variable: how dissimilarity increases with distance. Key tool for kriging and simulation. |
| **CDF** | Cumulative Distribution Function — maps each possible value to the probability of observing a value less than or equal to it. |
| **VPC** | Vertical Proportion Curve — the proportion of each facies (category) as a function of vertical position (depth), used in reservoir modeling. |
| **Covariance Model** | Mathematical function describing spatial continuity. HPGL supports: **Spherical** (linear behavior near origin, reaches sill at range), **Exponential** (steeper near-origin rise, asymptotically approaches sill), **Gaussian** (parabolic near origin, smooth continuity). |
| **Sill** | The value at which the variogram/covariance levels off (maximum variance). Must be positive. |
| **Nugget** | The discontinuity at zero distance (measurement error or microscale variation). Must be ≤ sill. |
| **Anisotropy** | Direction-dependent spatial continuity. The same variogram model may have different ranges in X, Y, and Z directions, controlled by `ranges` and `angles` parameters in `CovarianceModel`. |

### Covariance Models

- Spherical
- Exponential
- Gaussian

### How to Use This Guide

| You are... | Start here |
|------------|-----------|
| New to HPGL, want to try it out | [Quick Start](#quick-start) — runnable example with no external data |
| Need to build from source | [Build Instructions](#build-instructions) — Windows, Linux, macOS |
| Already using HPGL, need API details | [API Overview](#api-overview) — function signatures and usage |
| Running into problems | [Troubleshooting](#troubleshooting) — common errors and fixes |

## Requirements

### Common

- **[uv](https://docs.astral.sh/uv/)**: Package and environment manager (installs Python and dependencies automatically)
- **Python**: 3.9 or higher (tested up to 3.13) — installed by uv
- **NumPy**: 2.0 or higher, < 3.0 — installed by uv
- **SciPy**: (optional, for `routines` module) — installed by uv

### Windows Build

- Visual Studio 2022 Build Tools with C++ desktop development workload (v143 toolset)
- Intel oneAPI Math Kernel Library (MKL)

### Linux Build

- CMake 3.20+
- GCC 10+ or Clang 12+ with C++17 support
- OpenBLAS and LAPACK development libraries (or Intel MKL)
- OpenMP (optional, for parallelization)
- Python 3.9+ development headers (`python3-dev`)

## Build Instructions

### Windows (MSBuild - Recommended)

1. **Install prerequisites:**

   - Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with "Desktop development with C++" workload
   - Install [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (free)

2. **Set up the Python environment:**

   ```powershell
   uv sync --extra test
   ```

   This automatically installs the required Python version, creates a virtual environment, and installs all dependencies (NumPy, SciPy, pytest) from `pyproject.toml`.

3. **Build the native library:**

   ```powershell
   build.bat
   ```

   This compiles the C++ code using MSBuild (Release x64, v143 toolset) and produces:
   - `src\geo_bsd\hpgl.dll` (main native library, ~9.5 MB)
   - `src\geo_bsd\_cvariogram.dll` (variogram extension, ~22 KB)

4. **Verify the build:**

   ```powershell
   uv run python -c "import sys; sys.path.insert(0, 'src'); from geo_bsd import hpgl_wrap; print('Build OK')"
   ```

### Linux (CMake)

1. **Install system packages:**

   Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev \
       python3-dev libomp-dev
   ```

   Fedora/RHEL:
   ```bash
   sudo dnf install -y gcc-c++ cmake openblas-devel lapack-devel \
       python3-devel libomp-devel
   ```

2. **Set up the Python environment:**

   ```bash
   uv sync --extra test
   ```

3. **Build with CMake:**

   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DHPGL_BUILD_PYTHON=ON \
            -DHPGL_USE_OPENMP=ON \
            -DHPGL_USE_MKL=OFF
   cmake --build . --parallel $(nproc)
   ```

   To use Intel MKL instead of OpenBLAS:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DHPGL_USE_MKL=ON \
            -DMKL_ROOT=/opt/intel/oneapi/mkl/latest
   cmake --build . --parallel $(nproc)
   ```

4. **Install (optional):**

   ```bash
   sudo cmake --install .
   ```

### macOS Build

HPGL can be built on macOS using Homebrew-installed dependencies and the Clang compiler included with Xcode Command Line Tools.

1. **Install system packages:**

   ```bash
   # Install Xcode Command Line Tools (includes Apple Clang)
   xcode-select --install

   # Install dependencies via Homebrew
   brew install cmake ninja openblas libomp
   ```

   `llvm` (for OpenMP support) is also bundled with Apple Clang via `libomp`.

2. **Set up the Python environment:**

   ```bash
   uv sync --extra test
   ```

3. **Build with CMake (Ninja generator):**

   ```bash
   mkdir -p build && cd build
   cmake .. -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DHPGL_BUILD_PYTHON=ON \
            -DHPGL_USE_OPENMP=ON \
            -DHPGL_USE_MKL=OFF
   cmake --build . --parallel $(sysctl -n hw.logicalcpu)
   ```

   You may also use the `macos-clang` CMake preset:

   ```bash
   cmake --preset macos-clang
   cmake --build .
   ```

4. **Verify the build:**

   ```bash
   uv run python -c "import sys; sys.path.insert(0, 'src'); from geo_bsd import hpgl_wrap; print('Build OK')"
   ```

#### Known Issues on macOS

- **Apple Clang and the TNT library**: The bundled TNT (Template Numerical Toolkit) library uses C++98-era template patterns that may produce deprecation warnings or errors with newer Apple Clang versions. If you encounter compilation errors in `src/tnt_126/`, try:
  - Using GCC installed via Homebrew (`brew install gcc`, then set `-DCMAKE_CXX_COMPILER=g++-14`)
  - Adding `-Wno-deprecated-copy -Wno-c++11-narrowing` to `CMAKE_CXX_FLAGS`

- **OpenMP threading**: On Apple Silicon (arm64), OpenMP requires `libomp` from Homebrew. Ensure `/opt/homebrew/opt/libomp` is in your library path. CMake should detect this automatically.

- **Alternative approaches**: If native macOS build proves difficult, consider building via Docker (see the Linux instructions above in a container) or using a Linux VM (UTM, Parallels, or VMware Fusion).

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HPGL_BUILD_PYTHON` | ON | Build Python bindings |
| `HPGL_BUILD_TESTS` | OFF | Build C++ test suite |
| `HPGL_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `HPGL_USE_MKL` | OFF | Use Intel MKL (instead of OpenBLAS) |
| `HPGL_BUILD_VARIOGRAM` | ON | Build cvariogram extension module |

## Usage

### From Source (Development Mode)

After building, use `uv run` to execute scripts — it automatically manages the virtual environment:

```bash
uv run python my_script.py
```

Scripts need to add the `src/` directory to the Python path to import `geo_bsd`:

```python
import sys
sys.path.insert(0, "/path/to/hpgl/src")
import geo_bsd
```

## Quick Start

This self-contained example generates synthetic data with NumPy and runs Ordinary Kriging — no external data files required:

```python
import sys
sys.path.insert(0, "path/to/hpgl/src")

import numpy as np
import geo_bsd

# Generate synthetic data: a 3D grid with a simple trend + noise
nx, ny, nz = 50, 50, 20
np.random.seed(42)
data = np.zeros((nx, ny, nz), dtype=np.float32)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Trend surface: value grows with i and j
            data[i, j, k] = float(i + j) / 10.0 + np.random.normal(0, 0.5)

# Mask: mark all cells as "informed" (1 = valid)
mask = np.ones((nx, ny, nz), dtype=np.uint8)

# Wrap data into a contiguous property
prop = geo_bsd.ContProperty(data, mask)
grid = geo_bsd.SugarboxGrid(nx, ny, nz)

# Define a covariance model (spherical variogram)
cov = geo_bsd.CovarianceModel(
    type=geo_bsd.covariance.spherical,
    ranges=(10.0, 10.0, 5.0),
    sill=1.0,
    nugget=0.1
)

# Run Ordinary Kriging
result = geo_bsd.ordinary_kriging(
    prop, grid,
    radiuses=(10, 10, 5),
    max_neighbours=12,
    cov_model=cov
)

# Run Sequential Gaussian Simulation (SGS)
from geo_bsd.cdf import CdfData, calc_cdf
cdf = calc_cdf(prop)
sim = geo_bsd.sgs_simulation(
    prop, grid, cdf,
    radiuses=(10, 10, 5),
    max_neighbours=12,
    cov_model=cov,
    seed=42
)

print(f"Kriging result shape: {result.data.shape}, mean: {result.data.mean():.3f}")
print(f"SGS result shape:    {sim.data.shape}, mean: {sim.data.mean():.3f}")
```

Expected output (approximate values):
```
Kriging result shape: (50, 50, 20), mean: 4.950
SGS result shape:    (50, 50, 20), mean: 4.947
```

## API Overview

### Core Classes

| Class | Description |
|-------|-------------|
| `SugarboxGrid(x, y, z)` | 3D regular grid definition |
| `ContProperty(data, mask)` | Continuous property with informed/uninformed mask |
| `IndProperty(data, mask, indicator_count)` | Indicator (categorical) property |
| `CovarianceModel(type, ranges, angles, sill, nugget)` | Variogram/covariance model parameters |
| `CdfData(values, probs)` | Cumulative distribution function data |
| `covariance` | Class with model type constants: `spherical` (0), `exponential` (1), `gaussian` (2) |

### Kriging Functions

| Function | Description |
|----------|-------------|
| `ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model)` | Ordinary Kriging interpolation |
| `simple_kriging(prop, grid, radiuses, max_neighbours, cov_model, mean)` | Simple Kriging with known mean |
| `lvm_kriging(prop, grid, mean_data, radiuses, max_neighbours, cov_model)` | Kriging with Locally Varying Mean |
| `indicator_kriging(prop, grid, data, marginal_probs)` | Indicator Kriging for categorical data |
| `median_ik(prop, grid, marginal_probs, radiuses, max_neighbours, cov_model)` | Median Indicator Kriging (2 categories) |
| `simple_cokriging_markI(prop, grid, radiuses, max_neighbours, cov_model, secondary_data, primary_mean, secondary_mean, secondary_variance, correlation_coef)` | Cokriging using Markov Model I |
| `simple_cokriging_markII(grid, primary_data, secondary_data, correlation_coef, radiuses, max_neighbours)` | Cokriging using Markov Model II |

### Simulation Functions

| Function | Description |
|----------|-------------|
| `sgs_simulation(prop, grid, cdf_data, radiuses, max_neighbours, cov_model, seed, ...)` | Sequential Gaussian Simulation |
| `sis_simulation(prop, grid, data, seed, marginal_probs, ...)` | Sequential Indicator Simulation |

### I/O Functions

| Function | Description |
|----------|-------------|
| `load_cont_property(filename, undefined_value, size)` | Load continuous property from INC file |
| `load_ind_property(filename, undefined_value, indicator_values, size)` | Load indicator property from INC file |
| `read_inc_file_float(filename, undefined_value, size)` | Fast C++ reader for continuous INC data (specify grid size) |
| `read_inc_file_byte(filename, undefined_value, size, indicator_values)` | Fast C++ reader for indicator INC data |
| `write_property(prop, filename, prop_name, undefined_value, indicator_values=None)` | Write property to INC file (indicator_values for IndProperty) |
| `write_gslib_property(prop, filename, prop_name, undefined_value, indicator_values=None)` | Write property in GSLIB format (indicator_values for IndProperty) |
| `get_gslib_property(prop_dict, prop_name, undefined_value)` | Read a named property from a GSLIB-format property dictionary |

### Utility Functions

| Function | Description |
|----------|-------------|
| `calc_mean(prop)` | Calculate mean of informed values |
| `calc_cdf(prop)` | Calculate empirical CDF from property data |
| `set_thread_num(n)` | Set number of OpenMP threads |
| `get_thread_num()` | Get current OpenMP thread count |
| `simple_kriging_weights(center_point, n_x, n_y, n_z, ranges=(100000,100000,100000), sill=1, cov_type=covariance.exponential, nugget=None, angles=None)` | Compute kriging weights for a set of neighbor points |
| `set_output_handler(handler, param)` | Register a callback for C++ stdout/stderr output |
| `set_progress_handler(handler, param)` | Register a callback for C++ progress reporting |

### Submodules

Each submodule is imported under `geo_bsd` and exposes its own public API.

| Module | Description |
|--------|-------------|
| `geo_bsd.variogram` | Pure-Python variogram analysis: `TVEllipsoid`, `TVVariogramSearchTemplate`, `PointSetScanContStyle`, `PointSetScanGridStyle`, `CubeScan`, variogram/covariance/correlogram functions |
| `geo_bsd.cvariogram` | C-extension variogram (faster): `Ellipsoid`, `VariogramSearchTemplate`, `CalcVariograms`, `CalcVariogramsFromPointSet`, `CStackLayers` |
| `geo_bsd.routines` | High-level utilities: `CalcVPC`, `CalcVPCsIndicator`, `CalcMarginalProbsIndicator`, `CubeFromVPC`, `CubesFromVPCs`, `Cubes2PointSet`, `Cube2PointSet`, `PointSet2Cube`, `MeanCalc`, `CalcMean`, `SaveGSLIBPointSet`, `SaveGSLIBCubes`, `LoadGslibFile`, `MovingAverage3D`, `GetCubicalMask`, `GetEllipseMask` |
| `geo_bsd.validation` | Input validation framework: `GridValidator`, `ParameterValidator`, `PathValidator`, `ValidationContext`, `ValidationConstants` |

## Dataclass Property Reference

### ContProperty

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `numpy.ndarray` (float32, F-order) | Property values. May be 1D, 2D, or 3D. |
| `mask` | `numpy.ndarray` (uint8, F-order) | Informed/masked indicator. Non-zero = informed, 0 = masked. |

```python
prop = geo_bsd.ContProperty(data, mask)
# Access: prop.data, prop.mask
# Validate: prop.validate()
# Reshape to 3D: prop.fix_shape(grid)
```

### IndProperty

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `numpy.ndarray` (uint8, F-order) | Indicator values in `[0, indicator_count)`. |
| `mask` | `numpy.ndarray` (uint8, F-order) | Informed/masked indicator. Non-zero = informed. |
| `indicator_count` | `int` | Number of indicator categories. |

### SugarboxGrid

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Cells along X axis (1 to 10 million). |
| `y` | `int` | Cells along Y axis. |
| `z` | `int` | Cells along Z axis. |

### CovarianceModel

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `int` | `0` (spherical) | Use `geo_bsd.covariance.spherical` (0), `.exponential` (1), or `.gaussian` (2). |
| `ranges` | `tuple[float]` | `(0, 0, 0)` | Anisotropy ranges `(rx, ry, rz)`. |
| `angles` | `tuple[float]` | `(0.0, 0.0, 0.0)` | Anisotropy angles `(azimuth, dip, rotation)` in degrees. |
| `sill` | `float` | `1.0` | Covariance sill (must be positive). |
| `nugget` | `float` | `0.0` | Nugget effect (must be ≤ sill). |

### CdfData

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `numpy.ndarray` (float32) | Sorted unique property values from informed cells. |
| `probs` | `numpy.ndarray` (float32) | Cumulative probabilities corresponding to each value (monotonically non-decreasing). |

## Common Use Cases

### Load Data from File, Run Kriging, Save Result

```python
import sys
sys.path.insert(0, "path/to/hpgl/src")
import geo_bsd

# Load a continuous property from an INC file
prop = geo_bsd.load_cont_property("porosity.inc", undefined_value=-999.0, size=(50, 50, 20))

grid = geo_bsd.SugarboxGrid(50, 50, 20)

cov = geo_bsd.CovarianceModel(
    type=geo_bsd.covariance.spherical,
    ranges=(15.0, 15.0, 5.0),
    sill=1.0,
    nugget=0.1,
)

result = geo_bsd.ordinary_kriging(prop, grid, radiuses=(10, 10, 5), max_neighbours=12, cov_model=cov)

geo_bsd.write_property(result, "kriged_output.inc", "OK_Porosity", undefined_value=-999.0)
```

### Run Sequential Gaussian Simulation with a Seed

```python
from geo_bsd.cdf import calc_cdf

cdf = calc_cdf(prop)
sim = geo_bsd.sgs_simulation(
    prop, grid, cdf,
    radiuses=(10, 10, 5),
    max_neighbours=12,
    cov_model=cov,
    seed=12345,
)
```

### Run Indicator Kriging for Categorical Data

```python
import numpy as np
# 3-category indicator property (50x50x20 grid)
nx, ny, nz = 50, 50, 20
grid = geo_bsd.SugarboxGrid(nx, ny, nz)

# Create synthetic categorical data: random ints 0, 1, or 2
np.random.seed(42)
ind_data = np.random.randint(0, 3, size=(nx, ny, nz)).astype(np.uint8)
ind_mask = np.ones((nx, ny, nz), dtype=np.uint8)
ind_prop = geo_bsd.IndProperty(ind_data, ind_mask, indicator_count=3)

# Define covariance models (one per category)
cov1 = geo_bsd.CovarianceModel(
    type=geo_bsd.covariance.spherical, ranges=(8, 8, 4), sill=1.0, nugget=0.1,
)
cov2 = geo_bsd.CovarianceModel(
    type=geo_bsd.covariance.exponential, ranges=(8, 8, 4), sill=1.0, nugget=0.1,
)
cov3 = geo_bsd.CovarianceModel(
    type=geo_bsd.covariance.gaussian, ranges=(8, 8, 4), sill=1.0, nugget=0.1,
)

ik_data = [
    {"cov_model": cov1, "radiuses": (8, 8, 4), "max_neighbours": 12},
    {"cov_model": cov2, "radiuses": (8, 8, 4), "max_neighbours": 12},
    {"cov_model": cov3, "radiuses": (8, 8, 4), "max_neighbours": 12},
]

result = geo_bsd.indicator_kriging(
    ind_prop, grid, ik_data,
    marginal_probs=[0.4, 0.35, 0.25],
)
```

### Control Parallel Threads

```python
geo_bsd.set_thread_num(4)
print(f"Using {geo_bsd.get_thread_num()} threads")
```

### Compute Variogram (C Extension, Faster)

```python
from geo_bsd.cvariogram import Ellipsoid, VariogramSearchTemplate, CalcVariograms

ell = Ellipsoid(R1=20.0, R2=20.0, R3=10.0, azimuth=0, dip=0, rotation=0)
templ = VariogramSearchTemplate(
    lag_width=0.5, lag_separation=1.0, tol_distance=1.0,
    num_lags=15, first_lag_distance=0, ellipsoid=ell,
)
lags, variogram = CalcVariograms(templ, (data_array, mask_array), percent=100)
```

## Error Handling

HPGL functions raise Python exceptions for invalid input and runtime failures.
Always wrap calls in ``try``/``except`` to catch and diagnose errors.

### Input Validation Errors

```python
try:
    grid = geo_bsd.SugarboxGrid(-1, 10, 5)  # negative X dimension
except geo_bsd.validation.CriticalValidationError as e:
    print(f"Validation failed: {e}")
    print(f"  Parameter: {e.parameter_name}")
```

Common validation exceptions:

| Exception | When | Example Message |
|-----------|------|-----------------|
| `validation.CriticalValidationError` | Invalid parameter — prevents operation (e.g. negative grid dim, path traversal, `nugget > sill`) | `"Sill must be positive, got -1.0"`, `"Grid dimension must be >= 1, got -1"` |
| `validation.ValidationWarning` | Suspicious but non-fatal parameter value | `"Angle outside [0, 360] range"` |
| `TypeError` | Wrong argument type | `"expected ContProperty, IndProperty, or tuple, got str"` |
| `ValueError` | Out-of-range value, empty input, or invalid configuration | `"calc_cdf: no informed values (all cells are masked)"`, `"sill cannot be zero"` |
| `RuntimeError` | C++ computation failure (singular matrix, memory exhaustion, incompatible data) | `"ordinary_kriging failed: singular matrix"`, `"HPGL error: ..."` |
| `KeyError` | Missing property name in GSLIB property dictionary (`get_gslib_property`) | (standard Python KeyError) |
| `AttributeError` | Accessing nonexistent attribute on property or model objects | (standard Python AttributeError) |
| `OSError` / `FileNotFoundError` | File missing, permission denied, or disk I/O failure during load/write operations | `"Library directory not found"`, `"Cannot open file"` |

### Runtime Errors from C++ Backend

```python
try:
    result = geo_bsd.ordinary_kriging(prop, grid, radiuses=(10, 10, 5),
                                       max_neighbours=12, cov_model=cov)
except RuntimeError as e:
    print(f"C++ computation failed: {e}")
    # May indicate: singular matrix, memory exhaustion, incompatible data
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

### File Operation Errors

```python
try:
    prop = geo_bsd.load_cont_property("nonexistent.inc", undefined_value=-999.0,
                                       size=(50, 50, 20))
except validation.CriticalValidationError as e:
    print(f"Path validation failed: {e}")
except RuntimeError as e:
    print(f"File read failed: {e}")
```

### Progress Monitoring

```python
def my_progress(message, percent, param):
    print(f"[{percent}%] {message.decode()}")
    return 0  # return 0 to continue, non-zero to cancel

geo_bsd.set_progress_handler(my_progress, None)
result = geo_bsd.sgs_simulation(prop, grid, cdf, ...)
geo_bsd.set_progress_handler(None, None)  # unregister
```

## Testing

Run the full test suite:

```bash
uv run pytest tests/python/ -v
```

The test suite includes 622 tests covering:
- All kriging algorithms (OK, SK, LVM, IK, Median IK, Cokriging)
- All simulation algorithms (SGS, SIS)
- Edge cases and parameter validation
- NumPy 2.0+ compatibility
- Property I/O round-trip verification
- Thread management
- Memory safety

## Project Structure

```text
hpgl/
  build.bat              # Windows build script (MSBuild)
  CMakeLists.txt         # Cross-platform CMake build
  pyproject.toml         # Python project metadata
  sample-scripts/        # Example scripts demonstrating HPGL usage
  solved_problems_book/  # Practical implementations from geostatistics textbook
  legacy_documentation/  # Original documentation and manuals
  src/
    geo_bsd/             # Python package
      __init__.py        # Package entry point
      geo.py             # Core classes and kriging functions
      sgs.py             # Sequential Gaussian Simulation
      sis.py             # Sequential Indicator Simulation
      gtsim.py           # General Truncated Gaussian Simulation (GTSIM)
      cdf.py             # CDF computation
      variogram.py       # Variogram analysis (Python)
      cvariogram.py      # Variogram analysis (C extension)
      routines.py        # High-level utility routines
      validation.py      # Input validation framework
      hpgl_wrap.py       # C++ DLL interface (ctypes)
      hpgl.dll           # Built native library (Windows)
      hpgl/              # C++ source code
        api.cpp           # C API exports
        gauss_solver.cpp  # LAPACK linear system solver
        kriging_interpolation.h  # Kriging engine
        my_kriging_weights.h     # Weight calculation
        sequential_simulation.h  # Simulation framework
        ...
      _cvariogram/       # Variogram C++ extension
    msvc/                # Visual Studio project files
    tnt_126/             # TNT math library (headers)
  tests/
    python/              # Python test suite (pytest)
```

### Sample Scripts (`sample-scripts/`)

Example Python scripts demonstrating how to use the HPGL library for various geostatistical tasks: kriging, simulation (SGS/SIS), indicator kriging, cokriging, property I/O, histogram analysis, and more. These scripts serve as practical usage examples and integration tests for the `geo_bsd` Python API.

### Solved Problems (`solved_problems_book/`)

Python implementations of problems from the textbook *"Solved Problems in Geostatistics"* by Oy Leuangthong, K. Daniel Khan, and Clayton V. Deutsch (Wiley, 2008, ISBN 978-0-470-17792-1). The scripts cover topics including:

- **2.3** Standardization and probability intervals
- **3.1** Basic declustering
- **3.3** Comparison of declustering methods
- **4.1** Central limit theorem
- **4.2** Bootstrap and spatial bootstrap
- **4.3** Transfer of uncertainty (NPV analysis)
- **5.2** Variogram calculation (2D and 3D)
- **5.3** Variogram modeling and volume variance
- **7.2** Conditioning by kriging
- **7.3** Gaussian simulation
- **8.3** Indicator simulation for categorical data

The `shared/` subdirectory contains utility modules (statistics, GSLIB file I/O, variogram routines, grid classes, CDF transforms) used across all scripts.

### Legacy Documentation (`legacy_documentation/`)

Original documentation from earlier HPGL versions, including PDF manuals (English and Russian), Word source documents, and the archived sample scripts documentation.

## Troubleshooting

### General Diagnostics

Before diving into specific errors, verify your environment:

```bash
# Check Python and NumPy versions
uv run python -c "import sys, numpy; print(f'Python {sys.version}'); print(f'NumPy {numpy.__version__}')"

# Check that the native library loads
uv run python -c "import sys; sys.path.insert(0,'src'); from geo_bsd import hpgl_wrap; print('DLL loaded OK')"
```

### Build Failures

**Symptom:** `cmake` or `cmake --build` fails with linker errors.

**Cause:** Missing BLAS/LAPACK libraries (OpenBLAS or Intel MKL).

**Fix:**
1. Verify OpenBLAS is installed: `pkg-config --libs openblas` (Linux/macOS) or check `%MKLROOT%` (Windows)
2. On Linux: `sudo apt-get install -y libopenblas-dev liblapack-dev`
3. On macOS: `brew install openblas`
4. If using MKL: set `-DHPGL_USE_MKL=ON` and ensure `MKL_ROOT` points to the correct path

**Symptom:** `cmake` fails with `CMAKE_CXX_COMPILER not found`.

**Fix:** Install a C++17-capable compiler. On Linux: `sudo apt-get install build-essential`. On macOS: `xcode-select --install`. On Windows: install Visual Studio 2022 Build Tools.

**Symptom:** `fatal error: 'Python.h' file not found` during CMake build.

**Fix:** Install Python development headers. On Ubuntu/Debian: `sudo apt-get install python3-dev`. On Fedora: `sudo dnf install python3-devel`.

### DLL / Shared Library Not Found

**Symptom:** `ImportError: DLL load failed` or `ImportError: cannot open shared object file`.

**Cause:** The compiled native library (`hpgl.dll` on Windows, `hpgl.so` on Linux/macOS) is not in the Python path or is missing required system libraries.

**Fix:**
1. First re-run the build to ensure the library exists: `build.bat` (Windows) or `cmake --build .` (Linux/macOS)
2. Copy the built library to the Python package directory: `src/geo_bsd/`
3. Verify library dependencies are resolved:
   - **Linux:** `ldd src/geo_bsd/hpgl.so | grep "not found"`
   - **macOS:** `otool -L src/geo_bsd/hpgl.so`
   - **Windows:** Use Dependency Walker or `dumpbin /dependents src\geo_bsd\hpgl.dll`
4. If OpenBLAS is not found, add it to `LD_LIBRARY_PATH` (Linux) or reinstall it

### macOS-Specific Issues

**Symptom:** `error: no template named 'auto_ptr' in namespace 'std'`.

**Cause:** The TNT library uses C++98 features removed in C++17. Apple Clang is strict about this.

**Fix:** Add `-DCMAKE_CXX_FLAGS="-Wno-deprecated-copy"` to your cmake invocation, or use GCC: `brew install gcc && cmake .. -DCMAKE_CXX_COMPILER=g++-14`.

**Symptom:** `clang: error: unsupported option '-fopenmp'`.

**Cause:** Apple Clang does not support `-fopenmp`.

**Fix:** Install `libomp` via Homebrew: `brew install libomp`. CMake should detect it automatically.

**Symptom:** Build succeeds but `import geo_bsd` crashes with `Symbol not found: _omp_get_num_threads`.

**Cause:** OpenMP runtime not linked.

**Fix:** `brew install libomp` and rebuild. Ensure `/opt/homebrew/opt/libomp/lib` is in your library path.

### NumPy Version Mismatches

**Symptom:** `RuntimeError: module compiled against API version 0xf but this version of numpy is 0x10`.

**Cause:** The native library was compiled against a different NumPy version than what is installed.

**Fix:**
1. Rebuild the native library from clean state:
   ```bash
   rm -rf build/
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DHPGL_BUILD_PYTHON=ON
   cmake --build . --parallel $(nproc)
   ```
2. Ensure the NumPy version in your environment matches what CMake detected: `uv run python -c "import numpy; print(numpy.__version__)"`
3. If you upgraded NumPy after building, always rebuild the native library

## Frequently Asked Questions

### Do I need Intel MKL?
No. HPGL works with OpenBLAS (default on Linux/macOS) or Intel MKL (default on Windows). Use `-DHPGL_USE_MKL=OFF` for OpenBLAS, `-DHPGL_USE_MKL=ON` for MKL.

### Why do I get `ImportError` after building?
The native library (`hpgl.dll` / `hpgl.so`) must be built and available in `src/geo_bsd/`. Use `uv run python` to ensure the virtual environment is active.

### Can I use pip instead of uv?
The project uses `uv` for environment and dependency management. `uv sync` replaces `pip install`. Use `uv run` to execute scripts in the managed environment.

## License

HPGL Reborn is distributed under the **BSD 3-Clause License** (SPDX: `BSD-3-Clause`).
See [`license.txt`](license.txt) for the full license text.
