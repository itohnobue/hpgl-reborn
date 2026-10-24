# HPGL Reborn - High Performance Geostatistics Library (v2.1.0)

HPGL Reborn (High Performance Geostatistics Library) is a C++/Python library implementing geostatistical algorithms for spatial data analysis and reservoir modeling. The core computation engine is written in C++ for performance, with a Python interface for ease of use.

It is "Reborn" because it was updated, fixed and refactored to work with modern tech along with fixing dozens of bugs.

Originally developed at the Ufa Petroleum Institute, HPGL provides production-grade implementations of standard geostatistical methods as described in Deutsch & Journel's GSLIB.

## Features

- **Kriging**: Simple Kriging (SK), Ordinary Kriging (OK), LVM Kriging (Locally Varying Mean)
- **Indicator Kriging**: Indicator Kriging (IK), Median Indicator Kriging
- **Cokriging**: Simple Cokriging Mark I (Markov Model 1) and Mark II
- **Simulation**: Sequential Gaussian Simulation (SGS), Sequential Indicator Simulation (SIS), General Truncated Gaussian Simulation (GTSIM)
- **Variogram Analysis**: Experimental variogram calculation, variogram search templates
- **Utilities**: CDF computation, property I/O (INC/GSLIB formats), mean calculation, Vertical Proportion Curves (VPC)

## Build Instructions

**Prerequisites:** uv, Python 3.9+, NumPy 2.x, and a native toolchain — Visual Studio 2022 Build Tools with Intel oneAPI MKL on Windows; CMake 3.20+, GCC/Clang and OpenBLAS/LAPACK on Linux/macOS. See [Requirements](usage.md#requirements) for the full matrix.

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
   cmake --build --preset macos-clang
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

For the API reference, dataclass details, use cases, error handling, testing, project structure, troubleshooting, FAQ, and the geostatistics glossary, see [Usage](usage.md).

## License

HPGL Reborn is distributed under the **BSD 3-Clause License** (SPDX: `BSD-3-Clause`).
See [`license.txt`](license.txt) for the full license text.
