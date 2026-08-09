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
    data = rng.rand(500).astype("float32") * 100  # 10x10x5 = 500
    mask = np.ones(500, dtype="uint8")
    # Add some uninformed values
    mask[::10] = 0
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
    data = rng.randint(0, 3, 500, dtype="uint8")  # 3 indicators
    mask = np.ones(500, dtype="uint8")
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
        nugget=0.1,
    )


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
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "legacy: marks tests migrated from legacy test suite")
    config.addinivalue_line("markers", "hpgl: skip test when HPGL (geo_bsd) is not available")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.hpgl when HPGL is unavailable."""
    if not HPGL_AVAILABLE:
        skip_hpgl = pytest.mark.skip(reason="HPGL (geo_bsd) not available")
        for item in items:
            if "hpgl" in item.keywords:
                item.add_marker(skip_hpgl)
