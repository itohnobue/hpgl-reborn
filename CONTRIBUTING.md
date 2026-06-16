# Contributing to HPGL Reborn

Thank you for your interest in contributing! This guide covers how to set up a development environment, run tests, and submit changes.

## Development Setup

### Prerequisites

- **Python 3.9+** (managed by `uv`)
- **uv** ([docs](https://docs.astral.sh/uv/)) — Python package and environment manager
- **C++ build tools**:
  - **Windows**: Visual Studio 2022 Build Tools with C++ desktop workload + Intel oneAPI MKL
  - **Linux**: GCC 10+ or Clang 12+, CMake 3.20+, OpenBLAS/LAPACK dev headers, Python dev headers
  - **macOS**: Xcode CLT, Homebrew (`cmake`, `openblas`, `libomp`)

### Environment Setup

```bash
git clone https://github.com/hpgl/hpgl.git
cd hpgl

# Install Python dependencies (creates .venv automatically)
uv sync --extra dev
```

### Building

**Windows:**
```cmd
build.bat
```

**Linux/macOS:**
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHPGL_BUILD_PYTHON=ON -DHPGL_USE_OPENMP=ON
cmake --build . --parallel $(nproc)
```

## Running Tests

```bash
uv run pytest tests/python/ -v
```

The test suite contains 615 tests covering kriging, simulation, I/O, validation, and edge cases. To run a subset:

```bash
uv run pytest tests/python/test_kriging_complete.py -v
uv run pytest tests/python/test_validation.py -v
```

## Code Quality

This project uses **ruff** for linting. Run before committing:

```bash
uv run ruff check src/
uv run ruff format src/
```

Lint configuration lives in `pyproject.toml` under `[tool.ruff]`.

## Project Structure

```
src/
  geo_bsd/           # Python package
    __init__.py       # Package entry point, re-exports all public API
    geo.py            # Core classes (ContProperty, SugarboxGrid, etc.) and kriging functions
    sgs.py            # Sequential Gaussian Simulation
    sis.py            # Sequential Indicator Simulation
    cdf.py            # CDF computation (CdfData, calc_cdf)
    variogram.py      # Pure-Python variogram analysis
    cvariogram.py     # C-extension variogram (ctypes wrapper)
    routines.py       # High-level utility routines (VPC, GSLIB I/O, moving average)
    validation.py     # Input validation framework
    hpgl_wrap.py      # Native library loading and ctypes structure definitions
tests/
  python/             # pytest test suite
```

## Contribution Workflow

1. **Open an issue** to discuss the proposed change (bug fix, feature, documentation).
2. **Fork the repository** and create a feature branch.
3. **Make your changes** — follow the existing code style and conventions.
4. **Add tests** for new functionality or bug fixes.
5. **Run the full test suite** and lint checks.
6. **Submit a pull request** with a clear description of the change.

### Pull Request Guidelines

- Keep PRs focused on a single concern.
- Include before/after behavior description for bug fixes.
- Update `CHANGELOG.md` for user-facing changes.
- Ensure all existing tests pass — do not reduce test coverage.
- Add type hints for new Python code.
- For C++ changes, ensure the code compiles on all supported platforms (Windows, Linux, macOS).

## Reporting Issues

- Use the [GitHub Issues](https://github.com/hpgl/hpgl/issues) tracker.
- Include: HPGL version, Python version, NumPy version, OS, and a minimal reproduction script.
- For build issues, attach the full CMake/build output.

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License (see `license.txt`).
