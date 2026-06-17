# Changelog

All notable changes to HPGL Reborn.

## [1.6.0] — 2026

### Added
- CI/CD pipeline with GitHub Actions and cibuildwheel for automated multi-platform wheel builds
- Code quality tooling: ruff linter/formatter, mypy type checker, bandit security scanner
- macOS build support via Homebrew + Apple Clang with CMake Ninja generator

### Fixed
- Additional algorithm edge-case fixes (stale error propagation, thread safety)
- Build system hardening: OpenMP detection, MKL path resolution, cross-platform library linking

### Changed
- Python package metadata: updated authors, keywords, classifiers, and project URLs
- README: expanded build instructions for Windows, Linux, and macOS
- Security policy: updated supported versions and OWASP coverage documentation

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
- `sis_simulation`: added `min_neighbours`, `use_regions`, `region_size`; full parameter validation
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
