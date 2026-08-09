"""Tests for HPGL parser consistency — H-08, H-09.

Covers:
- H-08 (weakened to MED): Fast C++ parser vs slow Python parser consistency
- H-09 (weakened to MED): _validate_and_reshape_fallback coverage
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        _validate_and_reshape_fallback,
        load_cont_property,
        read_inc_file_float,
        write_property,
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


# =============================================================================
# H-08: Fast C++ vs slow Python parser consistency
# =============================================================================


@pytest.mark.hpgl
class TestParserConsistency:
    """Verify fast C++ and slow Python parsers produce same results (H-08)."""

    def test_fast_slow_parser_same_data_1d(self, tmp_path):
        """Fast and slow parsers return identical data/mask for 1D property."""
        data = np.array([10.0, 20.0, -99.0, 40.0, 50.0], dtype="float32")
        mask = np.array([1, 1, 0, 1, 1], dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "consistency_1d.inc")
        # F-28: pass an explicit trusted base — the default (DEFAULT_BASE_DIR)
        # is the process cwd, so writing to tmp_path requires an explicit base.
        write_property(prop_data, filename, "test", -99.0, basedir=str(tmp_path))

        # Slow parser (no size)
        slow_loaded = load_cont_property(filename, -99.0, basedir=str(tmp_path))

        # Fast parser (with size)
        fast_loaded = read_inc_file_float(filename, -99.0, 5, basedir=str(tmp_path))

        # Data should match
        np.testing.assert_array_equal(slow_loaded.data, fast_loaded.data)
        np.testing.assert_array_equal(slow_loaded.mask, fast_loaded.mask)

    def test_fast_slow_parser_same_mask_1d(self, tmp_path):
        """Fast and slow parsers produce identical masks."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float32")
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 1], dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "consistency_mask.inc")
        # F-28: pass an explicit trusted base.
        write_property(prop_data, filename, "masked", -99.0, basedir=str(tmp_path))

        slow = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast = read_inc_file_float(filename, -99.0, 8, basedir=str(tmp_path))

        np.testing.assert_array_equal(slow.data, fast.data)
        np.testing.assert_array_equal(slow.mask, fast.mask)

    def test_fast_slow_parser_consistent_reproducibility(self, tmp_path):
        """Repeated parser calls produce consistent results each time."""
        data = np.array([42.0, 0.0, 100.0, -50.0, 3.14], dtype="float32")
        mask = np.ones(5, dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "consistency_repeat.inc")
        # F-28: pass an explicit trusted base.
        write_property(prop_data, filename, "repeat", -99.0, basedir=str(tmp_path))

        # Read twice with each parser
        slow1 = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        slow2 = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast1 = read_inc_file_float(filename, -99.0, 5, basedir=str(tmp_path))
        fast2 = read_inc_file_float(filename, -99.0, 5, basedir=str(tmp_path))

        assert np.array_equal(slow1.data, slow2.data)
        assert np.array_equal(fast1.data, fast2.data)
        assert np.array_equal(slow1.data, fast1.data)

    def test_fast_slow_parser_extreme_values(self, tmp_path):
        """Parsers handle large magnitude values consistently."""
        data = np.array([1e-20, 1e20, -1e10, 1e10, 0.0], dtype="float32")
        mask = np.ones(5, dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "consistency_extreme.inc")
        # F-28: pass an explicit trusted base.
        write_property(prop_data, filename, "extreme", -99.0, basedir=str(tmp_path))

        slow = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast = read_inc_file_float(filename, -99.0, 5, basedir=str(tmp_path))

        # Values should be close (float precision may differ slightly)
        np.testing.assert_array_almost_equal(slow.data, fast.data, decimal=3)
        np.testing.assert_array_equal(slow.mask, fast.mask)


# =============================================================================
# H-09: _validate_and_reshape_fallback coverage
# =============================================================================


@pytest.mark.hpgl
class TestValidateAndReshapeFallback:
    """Test _validate_and_reshape_fallback (H-09)."""

    def test_validate_reshape_with_3tuple_size(self):
        """3-tuple size: reshapes to 3D Fortran order."""
        data = np.arange(24, dtype="float32")
        mask = np.ones(24, dtype="uint8")
        prop = ContProperty(data, mask)

        _validate_and_reshape_fallback(prop, (4, 3, 2), "test_func")

        assert prop.data.ndim == 3
        assert prop.data.shape == (4, 3, 2)
        assert prop.mask.ndim == 3
        assert prop.mask.shape == (4, 3, 2)

    def test_validate_reshape_with_scalar_size(self):
        """Scalar size: validates element count but does NOT reshape."""
        data = np.arange(10, dtype="float32")
        mask = np.ones(10, dtype="uint8")
        prop = ContProperty(data, mask)

        # Scalar size — should validate count but not reshape
        _validate_and_reshape_fallback(prop, 10, "test_func")
        assert prop.data.ndim == 1  # Should remain 1D

    def test_validate_reshape_mismatch_raises(self):
        """Element count mismatch raises RuntimeError."""
        data = np.arange(5, dtype="float32")
        mask = np.ones(5, dtype="uint8")
        prop = ContProperty(data, mask)

        with pytest.raises(RuntimeError, match="Slow parser read"):
            _validate_and_reshape_fallback(prop, (2, 3, 2), "test_func")
        # 5 != 12 (2*3*2)

    def test_validate_reshape_mismatch_scalar_raises(self):
        """Scalar size mismatch raises RuntimeError."""
        data = np.arange(7, dtype="float32")
        mask = np.ones(7, dtype="uint8")
        prop = ContProperty(data, mask)

        with pytest.raises(RuntimeError, match="Slow parser read"):
            _validate_and_reshape_fallback(prop, 100, "test_func")
        # 7 != 100
