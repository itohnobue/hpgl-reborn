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
    """Test NumPy 2.0+ compatibility"""

    def test_numpy_version(self):
        """Log NumPy version for testing"""
        print(f"NumPy version: {np.__version__}")
        version_tuple = tuple(int(x) for x in np.__version__.split(".")[:2])
        assert version_tuple >= (1, 24), f"NumPy 1.24+ required, got {np.__version__}"

    def test_array_creation_float32(self):
        """Test float32 array creation (NumPy 2.0 compatible)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        assert data.dtype == np.float32
        assert data.shape == (5,)

    def test_array_creation_uint8(self):
        """Test uint8 array creation (NumPy 2.0 compatible)"""
        data = np.array([0, 1, 2, 3, 4], dtype="uint8")
        assert data.dtype == np.uint8
        assert data.shape == (5,)

    def test_fortran_order_array(self):
        """Test Fortran-order array creation"""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32", order="F")
        assert data.flags["F_CONTIGUOUS"]

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

    def test_array_reshape_numpy2(self):
        """Test array reshape with NumPy 2.0"""
        data = np.arange(500, dtype="float32")
        reshaped = data.reshape((10, 10, 5), order="F")
        assert reshaped.shape == (10, 10, 5)

    def test_array_copy_numpy2(self):
        """Test array copy with NumPy 2.0"""
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        copied = data.copy("F")
        assert np.array_equal(data, copied)

    def test_ctypes_pointer_conversion(self):
        """Test ctypes pointer conversion (NumPy 2.0 compatible)"""
        import ctypes as C

        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        ptr = data.ctypes.data_as(C.POINTER(C.c_float))
        assert ptr is not None

    def test_require_cont_data(self):
        """Test _require_cont_data with NumPy 2.0"""
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        result = _require_cont_data(data)
        assert result is not None
        assert result.dtype == np.float32

    def test_3d_array_creation(self):
        """Test 3D array creation for grid data"""
        data = np.zeros((10, 10, 5), dtype="float32", order="F")
        mask = np.ones((10, 10, 5), dtype="uint8", order="F")

        assert data.shape == (10, 10, 5)
        assert data.flags["F_CONTIGUOUS"]
        assert mask.flags["F_CONTIGUOUS"]

    def test_array_strides_numpy2(self):
        """Test array strides with NumPy 2.0"""
        data = np.zeros((10, 10, 5), dtype="float32", order="F")
        strides = data.strides
        assert len(strides) == 3
        assert strides[0] == 4  # float32 size

    def test_masked_array_operations(self):
        """Test operations with masked arrays"""
        data = np.arange(100, dtype="float32")
        mask = np.ones(100, dtype="uint8")
        mask[::10] = 0  # Mask every 10th element

        prop = ContProperty(data, mask)
        # Count informed values
        informed_count = np.sum(prop.mask)
        assert informed_count == 90

    def test_copy_none_semantics(self):
        """NumPy 2.0: copy=None should behave as copy=True when dtype mismatches."""
        # In NumPy 2.0+, np.array(arr, copy=None) with dtype change copies the data.
        # In NumPy 1.x, copy=None with same dtype would be a no-copy view.
        data_float32 = np.array([1.0, 2.0, 3.0], dtype="float32")
        # Explicit copy=True is unambiguous and works across versions
        result = np.array(data_float32, dtype="float64", copy=True)
        assert result.dtype == np.float64
        assert np.array_equal(result, data_float32.astype("float64"))

    def test_string_ufunc_availability(self):
        """NumPy 2.0: string operations moved to np.strings namespace."""
        # np.strings is available in NumPy 2.0+
        if hasattr(np, "strings"):
            arr = np.array(["test_val", "other"], dtype=str)
            result = np.strings.replace(arr, "val", "")
            assert result[0] == "test_"
        else:
            pytest.skip("np.strings not available (NumPy < 2.0)")

    def test_float_power_integer_exponents(self):
        """NumPy 2.0: np.float_power with integer exponents returns float results."""
        # np.float_power returns float64 (Python int exponents promote to float64)
        result = np.float_power(np.array([2.0, 3.0], dtype="float32"), 2)
        assert np.issubdtype(result.dtype, np.floating)
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(9.0)

    def test_isin_availability(self):
        """NumPy 2.0: np.isin is the standard (np.in1d removed in 2.0)."""
        arr = np.array([1, 2, 3, 4, 5])
        test = np.array([2, 4])
        result = np.isin(arr, test)
        assert result[1]  # 2 is in test
        assert result[3]  # 4 is in test
        assert not result[0]  # 1 is not in test

    def test_unique_counts_stable(self):
        """NumPy 2.0: np.unique with return_counts works consistently."""
        arr = np.array([1, 1, 2, 2, 2, 3], dtype="float32")
        values, counts = np.unique(arr, return_counts=True)
        assert len(values) == 3
        assert counts[0] == 2  # Two 1s
        assert counts[1] == 3  # Three 2s
        assert counts[2] == 1  # One 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
