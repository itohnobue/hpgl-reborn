# Changelog

All notable changes to HPGL Reborn.

## [2.0.0] — 2026-07

### Added
- Parametrized test suites: 329 new tests covering all 8 kriging variants (162 tests) and SGS/SIS/GTSIM simulation (167 tests)
- Unified C++ solver entry point (`solver_entry_point.h`) with LAPACK error detection, eliminating ~440 lines of duplicated solver code across 7 kriging implementations
- Python FFI adapter layer (`ffi_adapter.py`) — 24 C API wrappers with GC pinning and error propagation, replacing direct ctypes usage from all Python modules
- Configuration dataclasses (`config.py`) — frozen `SGSConfig`, `SISConfig`, `GTSIMConfig` with `__post_init__` validation and backwards-compatible default parameters
- Math reference test suite (`test_math_reference.py`) — 69 tests validating C(0), kriging variance, correlogram weights, and cokriging formulas against textbook references
- C++ unit test suite (`test_hpgl_core.cpp` and `test_solver_entry_point.cpp`) — 646 lines covering core data structures and solver logic
- Code quality tooling: ruff linter/formatter, mypy type checker, bandit security scanner
- New Python test modules: FFI contract tests, validation coverage tests, callback lifecycle tests, kriging edge-case tests, parser consistency tests, covariance math tests

### Fixed
- **Silent C++ solver failure** — LAPACK error codes now propagate to Python as `HpglError` exceptions with user-visible error messages across all kriging and simulation functions
- **TVEllipsoid NaN/Inf validation bypass** — input range parameters (radii, angles) now validated for NaN/Inf values before use in variogram calculations
- **Division-by-zero guards** — added across all kriging weight calculations (SK, OK, LVM, IK, Median IK, Cokriging Mark I and II)
- **SIS regression** — fixed sequential indicator simulation initialization order causing incorrect indicator threshold computation
- **Kriging degenerate systems** — handled empty neighbourhood, insufficient-data, and singular-matrix fallback cases with proper error reporting
- **Numerical stability** — NaN/Inf propagation prevention in solver, covariance models, interpolation, and CDF computation
- **Cokriging Mark I/Mark II** — cross-covariance ratio handling, covariance matrix conditioning guards, degenerate system detection
- **SGS normalization** — coefficient correction, mean computation for single-sample and empty neighbourhoods
- **Covariance model** — nugget/sill enforcement (`nugget ≤ sill`), range bounds checking, parameter validation on all covariance types
- **OpenMP thread safety** — thread-local error snapshotting via `_snapshot_hpgl_error`; proper OpenMP detection in CMake
- **File I/O** — locale-independent INC file parsing; buffer overflow prevention in property writer; path traversal hardening
- **Memory safety** — use-after-free prevention and buffer bounds checking across the C++/Python bridge via array reference pinning

### Changed
- Python package metadata: updated authors, keywords, classifiers, and project URLs
- Build system: CMake presets for cross-platform configuration (`CMakePresets.json`); zero-warning compiler flags; macOS Ninja generator support
- Validation framework: activated 4 decorator-based validators across all public APIs; centralized 56 scattered `isinstance` checks into `validation.py` with `GridValidator`, `ParameterValidator`, `PathValidator`
- Kriging and simulation Python interfaces: refactored via FFI adapter and config dataclasses for cleaner API surface and reduced duplicate argument plumbing

### Removed
- CI/CD infrastructure: GitHub Actions workflows (ci.yml, release.yml) removed per minimal-project policy
- Community governance docs: CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md
- README sections: Validation Limits, Migration Guide, Changes from v0.9.9, production-ready FAQ, and Getting Help

## [1.5.0] — 2025

### Added
- Python 3.9–3.13 support (previously Python 2 only)
- NumPy 2.0+ compatibility
- Cross-platform CMake build system (Linux, macOS, Windows)
- macOS build support via Homebrew + Apple Clang
- Input validation framework (`geo_bsd.validation`)
  - `GridValidator`, `ParameterValidator`, `PathValidator` classes
  - `ValidationError`, `CriticalValidationError`, `ValidationWarning` exceptions
  - Decorator-based validation (`validate_grid_params`, `validate_kriging_params`, etc.)
  - Per-method parameter guards on all public API functions
- Security hardening
  - Path traversal prevention via `PathValidator.validate_filepath`
  - Safe native library loading with directory-containment checks
  - Array reference pinning to prevent use-after-free
  - Stale C++ error propagation fix via `_snapshot_hpgl_error` / `_check_hpgl_error`
- OpenMP thread safety: thread-aware error snapshotting
- `set_output_handler()` and `set_progress_handler()` for custom output/progress callbacks
- `variogram` module: `TVEllipsoid`, `TVVariogramSearchTemplate`, `PointSetScanContStyle`, `PointSetScanGridStyle`, `CubeScan`, CDF/covariance/correlogram functions
- `cvariogram` C-extension module: `Ellipsoid`, `VariogramSearchTemplate`, `CalcVariograms`, `CalcVariogramsFromPointSet`, `CStackLayers`
- `routines` module: `CalcVPC`, `CalcVPCsIndicator`, `CubeFromVPC`, `CubesFromVPCs`, `Cubes2PointSet`, `Cube2PointSet`, `PointSet2Cube`, `SaveGSLIBPointSet`, `SaveGSLIBCubes`, `LoadGslibFile`, `GetCubicalMask`, `GetEllipseMask`, `MovingAverage3D`, `MeanCalc`
- 622 automated tests with pytest
- Comprehensive README with build instructions for Windows, Linux, macOS

### Changed
- Windows build: migrated from VS 2008 to Visual Studio 2022 (v143 toolchain, C++17)
- BLAS backend: Replaced CLAPACK with Intel MKL (configurable; OpenBLAS also supported)
- Removed Boost dependency; replaced `boost::python` bindings with ctypes
- `sgs_simulation`: added `min_neighbours` parameter, validation on all inputs
- `sis_simulation`: added `min_neighbours`; full parameter validation
- Covariance model: `nugget ≤ sill` enforced with `CriticalValidationError`

### Fixed
- **Covariance C(0)** — nugget contribution was missing at zero-distance
- **OK kriging variance** — sign error in variance formula
- **Correlogram weights** — adjustment factor was inverted
- **Cokriging Mark II** — cross-covariance ratio was inverted
- **SGS normalization** — coefficient was incorrect
- **Covariance / indicator correlation** — spurious division by 2 removed
- **Missing parameter guard** — `cov_model` parameter made required in all kriging signatures

### Removed
- SCons and legacy Makefile build systems
- Debian packaging files
- Old VS 2008 project files
- Bundled Win32 installer
- Unused third-party libraries

## [0.9.9] — circa 2012

### Added
- Initial public release of HPGL
- C++ backend with Python bindings via Boost.Python
- Kriging algorithms: SK, OK, LVM, IK, Median IK, Cokriging (Mark I, Mark II)
- Simulation: SGS, SIS, GTSIM
- Basic property I/O (INC format)
- Experimental variogram calculation
- CDF computation
- Windows build via Visual Studio 2008
