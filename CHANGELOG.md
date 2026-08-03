# Changelog

All notable changes to HPGL Reborn.

## [2.0.3] — 2026-08

### Fixed
- **Kriging / solver hardening** — NaN/Inf finiteness validation at the C API boundary (SK/LVM/SGS-LVM/SIS-LVM means, cokriging means+variance, SGS CDF internal pointers + size); SIS/indicator-kriging `marginal_prob` [0,1] gate; cokriging NaN-bypassable guards made isfinite-first + markII `primary_variance` guard; incomplete secondary-equation degradation closed (drop secondary when variance not strictly-positive-finite); per-dimension primary↔secondary shape validation; clean kriging-failure error message (no raw libstdc++ text); weight-magnitude/residual guard on the `dpotrs_` success path; range-ratio overflow guard in anisotropy transform; `select.h` abort() replaced with catchable exceptions (no more uncatchable SIGABRT)
- **Simulation** — GTSIM truncation maps thresholds through the same empirical CDF as the SGS back-transform (correct category proportions); non-default `tk_mean`/`tk_std_dev` normalized to standard-normal space; SIS 2-category and multi-category probability clamps made NaN-safe; IK pre-correction sanitization; SGS scalar stationary mean transformed through the CDF; `max_neighbours=0` unconditional simulation no longer falls back to 1-neighbour conditioning
- **Variogram** — CubeScan total-work cap (no multi-hour hang / 24GB mgrid); GridStyle work-cap formula corrected; fractional grid-spacing lag matching via continuous binning; CalcIndCorrelationFunction excludes soft-prob 0/1 pairs; cvariogram input isfinite validation + output buffer size/writeability enforcement; 64-bit seed honored (no mod-2^32 collision); `tol_distance` template validation; stack_layers zero-thickness + full-buffer initialization
- **File I/O** — Windows write path atomic (temp+rename with handle closed before replace); tokenizer reassembly fixes silent token-splitting corruption at chunk boundaries; GSLIB writers reject finite out-of-window values; LoadGslibFile line-length/token bound; Cubes2PointSet equal-shape validation; CalcMean/CalcVPC isfinite gates
- **Property / validation** — IndProperty data/mask setters re-validate the indicator-range invariant; `undefined_value` collision check; ContProperty ctor validates the stored float32 array; I/O wrappers hold the FFI lock; CovarianceModel default ranges (1,1,1) aligned with C++; Python 3.9 import fixed (PEP 604 future-import)
- **Packaging / build** — `__version__` derived from source; `gtsim_2ind` + `SGSConfig`/`SISConfig`/`GTSIMConfig` exported at top level; Linux wheel/relocatability gates corrected (GNU-sed-safe RPATH/RUNPATH extraction, ldd column fix); CTest executed in the build pipeline; preset generator collisions resolved; sdist ships build scripts; MSVC Release config + OpenBLAS default link

### Added
- 200+ regression tests for the above across C++ and Python suites

### Fixed examples/scripts
- `solved_problems_book`: gaussian_cdf sign inversion, normal_score symmetry, cdf_transform value-rank quantiles, corr_coef bias, variance-of-mean intervals, grid Z-extent, crash-chains (np.copy, cdf_data, 2D/byte props), self-pair exclusion
- `sample-scripts`: private-import fixes, marginal_probs arg, filename/data-path fixes, truncation loops, pk_prop flow, hard-data facies preservation

## [2.0.2] — 2026-08

### Fixed
- **Kriging engine** — GSLIB-compliant SGS `ndmin` (original-data-only count); SGS-only OK→SK downgrade for <4 conditioning data (GSLIB sgsim); SGS failure fallback now draws N(mean, 1.0); OK-mode SGS honors the user mean; `kriging_type` validated at the C API; stats mean denominator consistent across all consumers
- **Cancellation / OpenMP** — user cancellation effective in default builds (cooperative stop flag, no `OMP_CANCELLATION` dependency); cokriging/plain-lookup pure-nugget fallback bounded (no per-node full-box scan)
- **Validation / API hardening** — `max_neighbours=0` rejected on kriging entry points; `min_neighbours` validated at the C API (`min > max` rejected); `simple_kriging_weights` count gate + post-FFI isfinite validation; median IK marginal-probability validation; property data/mask shape invariant enforced in setters and C++ bounds; `lvm_kriging` mean_data shape validation
- **Variogram** — pure-Python point-set scans now have a total-pair-lag work cap; Python lag-binning aligned to the C++ projection metric; `GridStyle` self-pair skip (both point-set scans agree)
- **File I/O** — slow-parser fallback applies the ±1.0e21 sentinel window; streaming parser bounded memory (no full-file list materialization, truncation fails fast); slow-parser tokenizer bounded against crafted oversized lines
- **Config / validation** — `SGSConfig`/`SISConfig` type gates accept numpy scalars + reject bool; `max_neighbours` Python cap aligned with C++ (hard reject above 100000); `validate_seed` raises `CriticalValidationError`
- **Simulation wrappers** — `gtsim_2ind` never mutates caller arrays (`pk_prop` and `prop`); `sis_simulation` config path no longer mutates caller dicts
- **Version / packaging** — `__version__` now reports the installed metadata version (regression-fixed); deterministic seeded variogram sampling (`calc_variograms_seeded`); Windows export of `cvar_clear_last_error`; wheel relocatable (static libomp, no absolute build-machine paths, `macosx_11_0` tag, verification gate); wheel ships the license file; sdist includes `tests/` and `CHANGELOG.md`; relocatable CMake export package; OpenMP hard-required (no silent serial degradation)

### Added
- 100+ regression tests for the above (new test files `test_production_fixes_geo.py`, `test_production_fixes_rest.py`, expanded `tests/cpp/test_hpgl_core.cpp`)

## [2.0.1] — 2026-08

### Fixed
- **Resource-exhaustion hardening** — `max_neighbours` hard cap added to `hpgl_indicator_kriging` (the only kriging entry point still missing it); radius magnitude guards on covariance-field construction across all kriging paths; work-based caps on point-set and grid variogram computation; tightened clusterizer allocation bound
- **Kriging failure observability** — cokriging, ordinary kriging, median IK, indicator kriging, SGS, and SIS now populate kriging stats that reach the Python layer (no silent mean-fill); `_last_kriging_stats` updates serialized under the FFI lock; stale-stat exposure removed from the public `get_kriging_stats()`
- **CDF numerical robustness** — float32 tail clamp applied after downcast (no spurious monotonicity error on large grids); `CdfData` enforces contiguity before FFI
- **Variogram correctness** — Python point-set scan now skips self-pairs to match the C++ kernel; `CubeScan` guards template-extent-vs-grid; `CalcVariogramFunction` tuple-path indexing fixed; `CalcVariogramsFromPointSet` enforces `ndim==1` + contiguity
- **GTSIM** — out-of-[0,1] clamping no longer mutates the caller's array; documented and regression-tested
- **File I/O security** — `O_EXCL` + unique temp names on C++ and Python writers, `O_NOFOLLOW` on C++ fast reads, property-name validation on all write paths (C++ and Python), GSLIB ±1.0e21 missing-value trimming on all read paths
- **OpenMP robustness** — BLAS thread-count guard is now RAII/exception-safe; exceptions no longer escape parallel regions (catchable errors instead of `std::terminate`); progress-callback re-entry from worker threads prevented

### Added
- Regression tests for the fixed issues: 2-category SIS no-spurious-warning, SGS `ndmin`, GTSIM clamp, CDF multi-value tail + monotonicity, variogram tuple-path + self-pair equivalence, `CubeScan` guard, property-name injection, GSLIB trimming round-trips

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
