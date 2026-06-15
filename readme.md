# HPGL Reborn - High Performance Geostatistics Library (v1.5.0)

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

### Covariance Models

- Spherical
- Exponential
- Gaussian

## Requirements

### Common

- **[uv](https://docs.astral.sh/uv/)**: Package and environment manager (installs Python and dependencies automatically)
- **Python**: 3.9 or higher (tested up to 3.13) — installed by uv
- **NumPy**: 2.0 or higher — installed by uv
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

   ```cmd
   uv sync --extra test
   ```

   This automatically installs the required Python version, creates a virtual environment, and installs all dependencies (NumPy, SciPy, pytest) from `pyproject.toml`.

3. **Build the native library:**

   ```cmd
   build.bat
   ```

   This compiles the C++ code using MSBuild (Release x64, v143 toolset) and produces:
   - `src\geo_bsd\hpgl.dll` (main native library, ~9.5 MB)
   - `src\geo_bsd\_cvariogram.dll` (variogram extension, ~22 KB)

4. **Verify the build:**

   ```cmd
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

## Installation

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
| `simple_cokriging_markI(prop, grid, secondary_data, primary_mean, secondary_mean, secondary_variance, correlation_coef, radiuses, max_neighbours, cov_model)` | Cokriging using Markov Model I |
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
| `write_property(prop, filename, prop_name, undefined_value)` | Write property to INC file |
| `write_gslib_property(prop, filename, prop_name, undefined_value)` | Write property in GSLIB format |

### Utility Functions

| Function | Description |
|----------|-------------|
| `calc_mean(prop)` | Calculate mean of informed values |
| `calc_cdf(prop)` | Calculate empirical CDF from property data |
| `set_thread_num(n)` | Set number of OpenMP threads |
| `get_thread_num()` | Get current OpenMP thread count |
| `simple_kriging_weights(center_point, n_x, n_y, n_z, ...)` | Compute kriging weights for a set of neighbor points |
| `get_gslib_property(filename, name, undefined_value, size)` | Read a named property from a GSLIB-format file |

## Testing

Run the full test suite:

```bash
uv run pytest tests/python/ -v
```

The test suite includes 615 tests covering:
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
  sample-scripts/        # Example scripts demonstrating HPGL usage
  solved_problems_book/  # Practical implementations from geostatistics textbook
  legacy_documentation/  # Original documentation and manuals
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

## Changes from v0.9.9

- **Python 3 support**: Full Python 3.9+ compatibility (previously Python 2 only)
- **NumPy 2.0+ support**: Compatible with modern NumPy versions
- **Visual Studio 2022**: Windows build with newer MSVC toolchain (v143, C++17)
- **Intel MKL**: Replaced CLAPACK with Intel MKL for LAPACK operations
- **Boost removed**: Replaced legacy boost::python with ctypes-based Python bindings
- **CMake build**: Cross-platform CMake build system alongside MSBuild
- **Input validation**: Comprehensive parameter validation framework
- **Security**: Path validation, array reference management, safe library loading
- **Modern build**: MSBuild-based build.bat, pyproject.toml, CMakeLists.txt
- **Algorithm bug fixes**: Fixed 7 mathematical bugs — covariance C(0) missing nugget contribution, OK kriging variance sign error, correlogram weight adjustment inverted, Cokriging Mark II cross-covariance ratio inverted, SGS normalization coefficient, and spurious /2 in covariance and indicator correlation functions
- **Test suite**: 615 automated tests with pytest
- **Legacy cleanup**: Removed unused libraries, old Boost.Python bindings, obsolete build systems (SCons, old Makefiles), Debian packaging, old VS 2008 project files, and a bundled Win32 installer, etc

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

## License

For non-commercial use (research, education, etc.) HPGL is distributed under the BSD license.
