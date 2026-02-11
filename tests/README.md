# HPGL Test Suite

This directory contains the comprehensive test suite for the HPGL (High Performance Geostatistics Library).

## Quick Start

```bash
# Run all tests
python tests/run_tests.py all

# Run with pytest directly
pytest tests/python/ -v

# Run specific test category
pytest tests/python/test_kriging.py -v
pytest tests/python/test_simulation.py -v
pytest tests/python/test_numpy2_compat.py -v
pytest tests/python/test_memory_leaks.py -v
pytest tests/python/test_performance.py -v
```

## Test Files

| File | Description | Tests |
|------|-------------|-------|
| `test_kriging.py` | Kriging algorithms | OK, SK, IK, numerical stability |
| `test_simulation.py` | Simulation algorithms | SGS, SIS |
| `test_numpy2_compat.py` | NumPy 2.0+ compatibility | Array handling, ctypes |
| `test_memory_leaks.py` | Memory leak detection | Memory cleanup tests |
| `test_performance.py` | Performance benchmarks | Timing and scaling |
| `test_integration.py` | Integration tests | Complete workflows |

## Test Categories

### Unit Tests
- Individual algorithm tests
- Function-level tests
- Input validation tests

### Integration Tests
- Complete geostatistical workflows
- Multi-step operations
- I/O operations

### Performance Tests
- Benchmarking
- Scaling tests
- Comparison tests

### Memory Tests
- Leak detection
- Cleanup verification
- Garbage collection

## Requirements

- Python >= 3.9
- NumPy >= 1.24
- pytest >= 7.0
- HPGL library built successfully

## Pre-Test Checklist

Before running tests, ensure:
- [ ] Build completed successfully
- [ ] HPGL can be imported: `import geo`
- [ ] NumPy >= 1.24: `python -c "import numpy; print(numpy.__version__)"`
- [ ] pytest installed: `pip install pytest pytest-cov pytest-benchmark`

## Coverage

Run tests with coverage:

```bash
pytest tests/python/ --cov=geo --cov-report=html
```

View coverage report:
```bash
# On Windows
start htmlcov/index.html

# On macOS/Linux
open htmlcov/index.html
```

## CI/CD Integration

These tests are designed to run in CI environments:
- Fast unit tests for every commit
- Full suite for pull requests
- Performance tracked over time
- Memory tests on releases

## Troubleshooting

### Import Errors
```
ImportError: No module named 'geo'
```
- Ensure HPGL is built
- Check PYTHONPATH includes `src/geo_bsd`

### NumPy Version Warnings
```
RuntimeWarning: numpy.ndarray size changed
```
- Update NumPy: `pip install --upgrade numpy`

### Skipped Tests
Tests with "HPGL not available" are skipped when:
- HPGL library is not built
- Python extension cannot be imported

## Test Results

### Expected Pass Rate
- All tests should pass when build is successful
- Tests are skipped when HPGL is not available
- No tests should fail due to import errors

### Performance Benchmarks
| Test | Grid Size | Expected Time |
|------|-----------|---------------|
| OK small | 10x10x5 | < 10s |
| OK medium | 50x50x20 | < 120s |
| SGS small | 10x10x5 | < 30s |
| Mean calc | 100k | < 1s |

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Use fixtures from conftest.py
3. Add appropriate markers (@pytest.mark.unit, etc.)
4. Include docstrings explaining what is tested
5. Ensure tests are deterministic (use random seeds)

## License

These tests are part of HPGL and follow the same BSD-3-Clause license.
