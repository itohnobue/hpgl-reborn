# HPGL - High Performance Geostatistics Library (v1.0.0)

## Project Site

[http://hpgl.github.io/hpgl/](http://hpgl.github.io/hpgl/)

## Description

HPGL (High Performance Geostatistics Library) is a C++/Python library implementing geostatistical algorithms for spatial data analysis and reservoir modeling. The core computation engine is written in C++ for performance, with a Python interface for ease of use.

Originally developed at the Ufa Petroleum Institute, HPGL provides production-grade implementations of standard geostatistical methods as described in Deutsch & Journel's GSLIB.

### Algorithms

- **Kriging**: Ordinary Kriging (OK), Simple Kriging (SK), LVM Kriging (Locally Varying Mean)
- **Indicator Kriging**: Indicator Kriging (IK), Median Indicator Kriging
- **Cokriging**: Simple Cokriging Mark I (Markov Model 1) and Mark II
- **Simulation**: Sequential Gaussian Simulation (SGS), Sequential Indicator Simulation (SIS)
- **Variogram Analysis**: Experimental variogram calculation, variogram search templates
- **Utilities**: CDF computation, property I/O (INC/GSLIB formats), mean calculation, Vertical Proportion Curves (VPC)

### Covariance Models

- Spherical
- Exponential
- Gaussian

## Requirements

### Common

- **Python**: 3.9 or higher (tested up to 3.14)
- **NumPy**: 1.24 or higher (compatible with NumPy 2.x)
- **SciPy**: (optional, for `routines` module)

### Windows Build

- Visual Studio 2022 Build Tools with C++ desktop development workload (v143 toolset)
- Intel oneAPI Math Kernel Library (MKL)
- Python 3.9+ with NumPy installed

### Linux Build

- CMake 3.20+
- GCC 10+ or Clang 12+ with C++17 support
- OpenBLAS and LAPACK development libraries (or Intel MKL)
- OpenMP (optional, for parallelization)
- Python 3.9+ development headers
- pybind11

## Build Instructions

### Windows (MSBuild - Recommended)

1. **Install prerequisites:**

   - Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with "Desktop development with C++" workload
   - Install [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (free)
   - Install Python 3.9+ and set up a virtual environment:

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install numpy scipy pytest
   ```

2. **Build the native library:**

   ```cmd
   build.bat
   ```

   This compiles the C++ code using MSBuild (Release x64, v143 toolset) and produces:
   - `src\geo_bsd\hpgl.dll` (main native library, ~9.5 MB)
   - `src\geo_bsd\_cvariogram.dll` (variogram extension, ~22 KB)

3. **Verify the build:**

   ```cmd
   .venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from geo_bsd import hpgl_wrap; print('Build OK')"
   ```

### Linux (CMake)

1. **Install prerequisites:**

   Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev \
       python3-dev python3-pip python3-venv libomp-dev
   pip3 install numpy scipy pytest pybind11
   ```

   Fedora/RHEL:
   ```bash
   sudo dnf install -y gcc-c++ cmake openblas-devel lapack-devel \
       python3-devel python3-pip libomp-devel
   pip3 install numpy scipy pytest pybind11
   ```

2. **Build with CMake:**

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

3. **Install (optional):**

   ```bash
   sudo cmake --install .
   ```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HPGL_BUILD_PYTHON` | ON | Build Python bindings |
| `HPGL_BUILD_TESTS` | OFF | Build C++ test suite |
| `HPGL_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `HPGL_USE_MKL` | OFF | Use Intel MKL (instead of OpenBLAS) |
| `HPGL_FETCH_EIGEN` | ON | Auto-download Eigen via FetchContent |
| `HPGL_BUILD_SHARED` | ON | Build shared libraries |
| `HPGL_BUILD_VARIOGRAM` | ON | Build cvariogram extension module |

## Installation

### From Source (Development Mode)

After building, add the `src/` directory to your Python path:

```python
import sys
sys.path.insert(0, "/path/to/hpgl/src")
import geo_bsd
```

Or set the `PYTHONPATH` environment variable:

```bash
# Linux/macOS
export PYTHONPATH="/path/to/hpgl/src:$PYTHONPATH"

# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\hpgl\src;$env:PYTHONPATH"

# Windows (CMD)
set PYTHONPATH=C:\path\to\hpgl\src;%PYTHONPATH%
```

## Quick Start

```python
import sys
sys.path.insert(0, "path/to/hpgl/src")

import numpy as np
import geo_bsd

# Create a 3D grid (100 x 100 x 50 cells)
grid = geo_bsd.SugarboxGrid(100, 100, 50)

# Load property data from INC file
prop = geo_bsd.load_cont_property("data.inc", -99, (100, 100, 50))

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

# Save results
geo_bsd.write_property(result, "kriging_result.inc", "KRIGING", -99)
geo_bsd.write_property(sim, "simulation_result.inc", "SGS_REAL1", -99)
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

### Kriging Functions

| Function | Description |
|----------|-------------|
| `ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model)` | Ordinary Kriging interpolation |
| `simple_kriging(prop, grid, radiuses, max_neighbours, cov_model, mean)` | Simple Kriging with known mean |
| `lvm_kriging(prop, grid, mean_data, radiuses, max_neighbours, cov_model)` | Kriging with Locally Varying Mean |
| `indicator_kriging(prop, grid, data, marginal_probs)` | Indicator Kriging for categorical data |
| `median_ik(prop, grid, marginal_probs, radiuses, max_neighbours, cov_model)` | Median Indicator Kriging (2 categories) |
| `simple_cokriging_markI(...)` | Cokriging using Markov Model I |
| `simple_cokriging_markII(...)` | Cokriging using Markov Model II |

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
| `write_property(prop, filename, prop_name, undefined_value)` | Write property to INC file |
| `write_gslib_property(prop, filename, prop_name, undefined_value)` | Write property in GSLIB format |

### Utility Functions

| Function | Description |
|----------|-------------|
| `calc_mean(prop)` | Calculate mean of informed values |
| `calc_cdf(prop)` | Calculate empirical CDF from property data |
| `set_thread_num(n)` | Set number of OpenMP threads |
| `get_thread_num()` | Get current OpenMP thread count |

## Testing

Run the full test suite:

```cmd
cd hpgl
.venv\Scripts\python.exe -m pytest tests/python/ -v
```

The test suite includes 344+ tests covering:
- All kriging algorithms (OK, SK, LVM, IK, Median IK, Cokriging)
- All simulation algorithms (SGS, SIS)
- Edge cases and parameter validation
- NumPy 2.0+ compatibility
- Property I/O round-trip verification
- Thread management
- Memory safety

## Project Structure

```
hpgl/
  build.bat              # Windows build script (MSBuild)
  CMakeLists.txt         # Cross-platform CMake build
  pyproject.toml         # Python project metadata
  src/
    geo_bsd/             # Python package
      __init__.py        # Package entry point
      geo.py             # Core classes and kriging functions
      sgs.py             # Sequential Gaussian Simulation
      sis.py             # Sequential Indicator Simulation
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
    hpgl.pyd/            # Boost.Python bindings (legacy)
    msvc/                # Visual Studio project files
    GsTL-1.3/            # GsTL library (included)
    tnt_126/             # TNT library (included)
  tests/
    python/              # Python test suite (pytest)
```

## Changes from v0.9.9

- **Python 3 support**: Full Python 3.9-3.14 compatibility (previously Python 2 only)
- **NumPy 2.0+ support**: Compatible with modern NumPy versions (removed `numpy.matrix`)
- **Visual Studio 2022**: Build with modern MSVC toolchain (v143, C++17)
- **Intel MKL**: Replaced CLAPACK with Intel MKL for LAPACK operations
- **Boost removed**: Replaced boost::python with ctypes-based Python bindings
- **CMake build**: Cross-platform CMake build system alongside MSBuild
- **Input validation**: Comprehensive parameter validation framework
- **Security**: Path validation, array reference management, safe library loading
- **Modern build**: MSBuild-based build.bat, pyproject.toml, CMakeLists.txt
- **Test suite**: 344+ automated tests with pytest

## License

For non-commercial use (research, education, etc.) HPGL is distributed under the BSD license.
For questions about commercial distribution, please contact the Authors.

## Authors

Management & Math:
- Savichev Vladimir
- Bezrukov Andrey

Programming (C++, Python), testing, support:
- Muharlyamov Arthur
- Barskiy Konstantin
- Nasibullina Dina
- Safin Rustam

## Acknowledgements

- The Authors wish to thank Andre Journel for his valuable support and indefatigable enthusiasm.
- The Authors also thank Iskander Shafikov for his assistance with the English translations and the User Guide cover.
