"""
Pytest configuration and fixtures for HPGL tests

This module provides comprehensive fixtures for testing HPGL (Geostatistical
Python Library) functionality including grids, properties, covariance models,
CDF data, and various test data scenarios.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Check if HPGL is available
try:
    # Import modules individually to avoid __init__.py triggering
    # geo.py syntax errors in unrelated modules.
    # Use importlib to avoid unused-import lint warnings.
    import importlib.util
    _spec = importlib.util.find_spec("geo_bsd.geo")
    HPGL_AVAILABLE = _spec is not None
except (ImportError, SyntaxError, IndentationError, OSError):
    HPGL_AVAILABLE = False


# =============================================================================
# Autouse Fixture: Handler Cleanup (M10)
# =============================================================================

@pytest.fixture(autouse=True)
def _clear_handlers():
    """Autouse fixture: clear output and progress handlers on teardown.

    Ensures output_handler and progress_handler are always reset to None
    after every test, even if a C++ crash or assertion failure interrupts
    the test before inline cleanup runs. This is a yield-based fixture:
    setup runs before the test, teardown after.
    """
    yield
    if HPGL_AVAILABLE:
        try:
            from geo_bsd.geo import set_output_handler, set_progress_handler
            set_output_handler(None, None)
            set_progress_handler(None, None)
        except (ImportError, OSError):
            pass


# =============================================================================
# Grid Fixtures
# =============================================================================


@pytest.fixture
def sample_grid():
    """Create a sample 3D grid for testing (10x10x5).

    Returns:
        SugarboxGrid: A grid with dimensions 10x10x5 suitable for general testing.
    """
    import geo_bsd
    return geo_bsd.geo.SugarboxGrid(x=10, y=10, z=5)


@pytest.fixture
def small_grid():
    """Create a small grid for quick tests (5x5x3).

    This fixture is ideal for fast unit tests where minimal grid size
    is sufficient to verify functionality.

    Returns:
        SugarboxGrid: A compact 5x5x3 grid.
    """
    import geo_bsd
    return geo_bsd.geo.SugarboxGrid(x=5, y=5, z=3)


@pytest.fixture
def medium_grid():
    """Create a medium-sized grid (20x20x10).

    Suitable for tests that need more cells than sample_grid but still
    execute quickly.

    Returns:
        SugarboxGrid: A 20x20x10 grid.
    """
    import geo_bsd
    return geo_bsd.geo.SugarboxGrid(x=20, y=20, z=10)


@pytest.fixture
def large_grid():
    """Create a large grid for stress tests (50x50x20).

    This fixture is intended for performance testing and stress testing.
    Use sparingly as it contains 50,000 cells.

    Returns:
        SugarboxGrid: A large 50x50x20 grid.
    """
    import geo_bsd
    return geo_bsd.geo.SugarboxGrid(x=50, y=50, z=20)


@pytest.fixture
def non_cubic_grid():
    """Create a non-cubic grid with different dimensions (30x20x10).

    Useful for testing algorithms that should work correctly with
    anisotropic grid dimensions.

    Returns:
        SugarboxGrid: A 30x20x10 grid with non-equal dimensions.
    """
    import geo_bsd
    return geo_bsd.geo.SugarboxGrid(x=30, y=20, z=10)


# =============================================================================
# Property Fixtures
# =============================================================================


@pytest.fixture
def sample_property():
    """Create sample continuous property data (10x10x5 grid size).

    The property has approximately 90% informed values with the remaining
    10% marked as uninformed.

    Returns:
        ContProperty: A continuous property with 500 cells.
    """
    import geo_bsd
    rng = np.random.RandomState(42)
    data = rng.rand(500).astype('float32') * 100  # 10x10x5 = 500
    mask = np.ones(500, dtype='uint8')
    # Add some uninformed values
    mask[::10] = 0
    return geo_bsd.geo.ContProperty(data, mask)


@pytest.fixture
def sparse_property():
    """Create a property with 90% uninformed values.

    This fixture is useful for testing algorithms that need to handle
    very sparse data scenarios.

    Returns:
        ContProperty: A property with only 10% informed values (100 out of 1000).
    """
    import geo_bsd
    rng = np.random.RandomState(42)
    data = rng.rand(1000).astype('float32') * 100
    mask = np.zeros(1000, dtype='uint8')
    mask[::10] = 1  # Only 10% informed
    return geo_bsd.geo.ContProperty(data, mask)


@pytest.fixture
def dense_property():
    """Create a property with 100% informed values.

    All cells have informed values, useful for testing algorithms on
    fully populated data.

    Returns:
        ContProperty: A fully informed property with 1000 cells.
    """
    import geo_bsd
    rng = np.random.RandomState(42)
    data = rng.rand(1000).astype('float32') * 100
    mask = np.ones(1000, dtype='uint8')
    return geo_bsd.geo.ContProperty(data, mask)


@pytest.fixture
def uniform_property():
    """Create a property where all values are the same.

    Useful for testing edge cases and boundary conditions.

    Returns:
        ContProperty: A property with all values set to 50.0.
    """
    import geo_bsd
    data = np.full(1000, 50.0, dtype='float32')
    mask = np.ones(1000, dtype='uint8')
    return geo_bsd.geo.ContProperty(data, mask)


@pytest.fixture
def extreme_values_property():
    """Create a property with extreme numerical values.

    Useful for testing numerical stability and handling of very large
    and very small values.

    Returns:
        ContProperty: A property with values ranging from 1e-30 to 1e30.
    """
    import geo_bsd
    data = np.array([1e-30, 1e30, -1e10, 1e10] * 250, dtype='float32')
    mask = np.ones(1000, dtype='uint8')
    return geo_bsd.geo.ContProperty(data, mask)


@pytest.fixture
def sample_indicator_property():
    """Create sample indicator property data (10x10x5 grid size).

    Creates an indicator property with 3 categories and approximately
    90% informed values.

    Returns:
        IndProperty: An indicator property with 3 categories.
    """
    import geo_bsd
    rng = np.random.RandomState(42)
    data = rng.randint(0, 3, 500, dtype='uint8')  # 3 indicators
    mask = np.ones(500, dtype='uint8')
    mask[::10] = 0
    return geo_bsd.geo.IndProperty(data, mask, 3)


# =============================================================================
# Covariance Model Fixtures
# =============================================================================


@pytest.fixture
def sample_covariance_model():
    """Create a sample covariance model (spherical).

    Returns:
        CovarianceModel: A spherical covariance model with isotropic ranges.
    """
    import geo_bsd
    return geo_bsd.geo.CovarianceModel(
        type=geo_bsd.geo.covariance.spherical,
        ranges=(5.0, 5.0, 3.0),
        angles=(0.0, 0.0, 0.0),
        sill=1.0,
        nugget=0.1
    )


@pytest.fixture
def spherical_cov_model():
    """Create a spherical covariance model.

    The spherical model is one of the most commonly used variogram models
    in geostatistics.

    Returns:
        CovarianceModel: A spherical covariance with ranges (10, 10, 5).
    """
    import geo_bsd
    return geo_bsd.geo.CovarianceModel(
        type=geo_bsd.geo.covariance.spherical,
        ranges=(10.0, 10.0, 5.0),
        sill=1.0,
        nugget=0.1
    )


@pytest.fixture
def exponential_cov_model():
    """Create an exponential covariance model.

    The exponential model has a more gradual approach to the sill than
    the spherical model.

    Returns:
        CovarianceModel: An exponential covariance with ranges (15, 15, 8).
    """
    import geo_bsd
    return geo_bsd.geo.CovarianceModel(
        type=geo_bsd.geo.covariance.exponential,
        ranges=(15.0, 15.0, 8.0),
        sill=1.0,
        nugget=0.1
    )


@pytest.fixture
def gaussian_cov_model():
    """Create a Gaussian covariance model.

    The Gaussian model is very smooth and approaches the sill asymptotically.

    Returns:
        CovarianceModel: A Gaussian covariance with ranges (12, 12, 6).
    """
    import geo_bsd
    return geo_bsd.geo.CovarianceModel(
        type=geo_bsd.geo.covariance.gaussian,
        ranges=(12.0, 12.0, 6.0),
        sill=1.0,
        nugget=0.1
    )


@pytest.fixture
def anisotropic_cov_model():
    """Create an anisotropic covariance model.

    This model has different ranges in each direction and rotation angles,
    useful for testing geostatistical operations with anisotropy.

    Returns:
        CovarianceModel: An anisotropic spherical covariance with
            ranges (20, 10, 5) and rotation angles (30, 0, 0).
    """
    import geo_bsd
    return geo_bsd.geo.CovarianceModel(
        type=geo_bsd.geo.covariance.spherical,
        ranges=(20.0, 10.0, 5.0),
        angles=(30.0, 0.0, 0.0),
        sill=1.0,
        nugget=0.1
    )


# =============================================================================
# CDF Data Fixtures
# =============================================================================


@pytest.fixture
def uniform_cdf():
    """Create a uniform distribution CDF.

    Represents a simple two-point uniform distribution from 0 to 100.

    Returns:
        CdfData: A CDF with uniform distribution.
    """
    import geo_bsd
    values = np.array([0.0, 100.0], dtype='float32')
    probs = np.array([0.0, 1.0], dtype='float32')
    return geo_bsd.CdfData(values, probs)


@pytest.fixture
def normal_cdf():
    """Create a normal-like distribution CDF.

    Represents a distribution that approximates a normal distribution
    with quintile points.

    Returns:
        CdfData: A CDF with 5 points approximating a normal distribution.
    """
    import geo_bsd
    values = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32')
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
    return geo_bsd.CdfData(values, probs)


@pytest.fixture
def multi_cdf():
    """Create a multi-value CDF with many points.

    Useful for testing algorithms that work with complex CDF representations.

    Returns:
        CdfData: A CDF with 20 points linearly spaced from 0 to 100.
    """
    import geo_bsd
    values = np.linspace(0, 100, 20, dtype='float32')
    probs = np.linspace(0, 1, 20, dtype='float32')
    return geo_bsd.CdfData(values, probs)


# =============================================================================
# LVM (Locally Varying Mean) Data Fixtures
# =============================================================================


@pytest.fixture
def lvm_mean_data():
    """Create locally varying mean data array.

    Provides mean values for LVM kriging operations.

    Returns:
        ndarray: Array of 5000 random float32 values scaled to 0-100 range.
    """
    rng = np.random.RandomState(42)
    return rng.rand(5000).astype('float32') * 100


@pytest.fixture
def lvm_mean_data_grid():
    """Create LVM data matching sample grid size.

    Returns mean data sized appropriately for the 10x10x5 sample grid.

    Returns:
        ndarray: Array of 1000 random float32 values scaled to 0-100 range.
    """
    rng = np.random.RandomState(42)
    return rng.rand(1000).astype('float32') * 100


# =============================================================================
# Mask Fixtures
# =============================================================================


@pytest.fixture
def checkerboard_mask():
    """Create a checkerboard pattern mask.

    Creates an alternating pattern of informed/uninformed cells useful
    for testing algorithms with regular patterns of missing data.

    Returns:
        ndarray: A uint8 mask array with checkerboard pattern.
    """
    size = 1000
    mask = np.zeros(size, dtype='uint8')
    for i in range(size):
        if (i // 10 + i % 10) % 2 == 0:
            mask[i] = 1
    return mask


@pytest.fixture
def random_mask():
    """Create a random sparse mask with ~50% informed values.

    Provides a random pattern of informed cells for testing stochastic
    behavior.

    Returns:
        ndarray: A uint8 mask array with random binary values.
    """
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, 1000, dtype='uint8')


# =============================================================================
# Array Compatibility Fixtures
# =============================================================================


@pytest.fixture
def numpy2_compatible_array():
    """Create array compatible with NumPy 2.0+.

    Returns a simple array that works across NumPy versions for testing
    compatibility.

    Returns:
        ndarray: A float32 array with 5 elements.
    """
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')


@pytest.fixture
def fortran_order_array():
    """Create a Fortran-order (column-major) array.

    HPGL uses Fortran-order arrays internally. This fixture provides
    a properly ordered array for testing.

    Returns:
        ndarray: A float32 array in Fortran order.
    """
    return np.asfortranarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32'))


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir():
    """Path to test data directory.

    Returns:
        Path: Path object pointing to the test data directory.
    """
    data_path = Path(__file__).parent / "test_data"
    if not data_path.exists():
        # Fallback to the old (incorrect) path for backward compat
        data_path = Path(__file__).parent.parent / "data"
    return data_path


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings.

    This function is called once at the start of the test run and can
    be used to register custom markers or perform other setup.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "legacy: marks tests migrated from legacy test suite"
    )
    config.addinivalue_line(
        "markers", "hpgl: skip test when HPGL (geo_bsd) is not available"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.hpgl when HPGL is unavailable."""
    if not HPGL_AVAILABLE:
        skip_hpgl = pytest.mark.skip(reason="HPGL (geo_bsd) not available")
        for item in items:
            if "hpgl" in item.keywords:
                item.add_marker(skip_hpgl)


@pytest.fixture
def skip_if_hpgl_not_available():
    """Fixture to skip tests if HPGL is not available.

    Use this fixture in tests that require the HPGL library:

        @pytest.mark.usefixtures("skip_if_hpgl_not_available")
        def test_something():
            # Test code here
    """
    if not HPGL_AVAILABLE:
        pytest.skip("HPGL (geo_bsd) not available")



