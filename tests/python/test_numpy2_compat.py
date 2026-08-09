"""
NumPy 2.0+ compatibility tests for HPGL
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import os

os.environ["PATH"] = (
    str(Path(__file__).parent.parent.parent / "src" / "geo_bsd") + ";" + os.environ.get("PATH", "")
)

try:
    from geo_bsd.geo import (
        ContProperty,
        IndProperty,
        _require_cont_data,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


@pytest.mark.hpgl
class TestNumPy2Compatibility:
    """Test NumPy 2.0+ compatibility of HPGL wrapper code.

    Only the tests that exercise HPGL code (ContProperty / IndProperty /
    _require_cont_data / mask semantics) are kept — the 14 pure-numpy
    self-tests (array creation, reshape, copy, ctypes, np.strings,
    float_power, isin, unique, ...) tested NumPy primitives with zero HPGL
    detection power and were removed (T-18).
    """

    def test_contproperty_numpy2(self):
        """Test ContProperty with NumPy 2.0 arrays"""
        data = np.zeros(100, dtype="float32", order="F")
        mask = np.ones(100, dtype="uint8", order="F")

        prop = ContProperty(data, mask)
        assert prop.data.shape == (100,)
        assert prop.mask.shape == (100,)

    def test_indproperty_numpy2(self):
        """Test IndProperty with NumPy 2.0 arrays"""
        data = np.zeros(100, dtype="uint8", order="F")
        mask = np.ones(100, dtype="uint8", order="F")

        prop = IndProperty(data, mask, 3)
        assert prop.data.shape == (100,)
        assert prop.indicator_count == 3

    def test_require_cont_data(self):
        """Test _require_cont_data with NumPy 2.0.

        _require_cont_data wraps numpy.require(data, dtype="float32",
        requirements="F") — the discriminating assertion is F-contiguity,
        not merely that a result object exists.
        """
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        result = _require_cont_data(data)
        assert result.dtype == np.float32
        assert result.flags["F_CONTIGUOUS"]

    def test_masked_array_operations(self):
        """Test operations with masked arrays"""
        data = np.arange(100, dtype="float32")
        mask = np.ones(100, dtype="uint8")
        mask[::10] = 0  # Mask every 10th element

        prop = ContProperty(data, mask)
        # Count informed values
        informed_count = np.sum(prop.mask)
        assert informed_count == 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
