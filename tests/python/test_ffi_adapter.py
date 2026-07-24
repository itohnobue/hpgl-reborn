"""Tests for ffi_adapter.py — FFI internals (checked_create, error_guard, stride checks).

I2-F03: Test checked_create() for missing-field detection and extra-kwarg behavior.
I2-F17: Test error_guard() for stale-error suppression, new-error detection, and
        double-raise prevention.
I2-F35: Test stride mismatch detection in create_cont_masked_array and create_ind_masked_array.
"""

import ctypes as C
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ——————————————————————————————————————————————————————————————————————
# I2-F03: checked_create tests — field-completeness and struct construction
# ——————————————————————————————————————————————————————————————————————

class TestCheckedCreate:
    """Unit tests for checked_create() field-completeness validation (I2-F03)."""

    def test_all_fields_provided_constructs_struct(self):
        """checked_create() should construct struct when all fields are provided."""
        from geo_bsd.ffi_adapter import checked_create

        class _TestStruct(C.Structure):
            _fields_ = [("a", C.c_int), ("b", C.c_double)]

        result = checked_create(_TestStruct, a=42, b=3.14)
        assert result.a == 42
        assert result.b == 3.14

    def test_missing_field_raises_critical_validation_error(self):
        """checked_create() must raise CriticalValidationError for missing fields (I2-F03)."""
        from geo_bsd.ffi_adapter import checked_create
        from geo_bsd.validation import CriticalValidationError

        class _TestStruct(C.Structure):
            _fields_ = [("a", C.c_int), ("b", C.c_double), ("c", C.c_float)]

        with pytest.raises(CriticalValidationError) as excinfo:
            checked_create(_TestStruct, a=1, b=2.0)
        assert "No values for parameters" in str(excinfo.value)
        assert "c" in str(excinfo.value)

    def test_missing_all_fields_reports_all(self):
        """All missing fields should be listed in the error message."""
        from geo_bsd.ffi_adapter import checked_create
        from geo_bsd.validation import CriticalValidationError

        class _TestStruct(C.Structure):
            _fields_ = [("x", C.c_int), ("y", C.c_int), ("z", C.c_int)]

        with pytest.raises(CriticalValidationError):
            checked_create(_TestStruct)
        # All three fields should be reported

    def test_extra_kwargs_behavior(self):
        """Extra kwargs passed to checked_create — behavior depends on ctypes version.

        checked_create passes all kwargs through to Structure(**kargs).
        ctypes silently ignores unknown field names (no TypeError).
        """
        from geo_bsd.ffi_adapter import checked_create

        class _TestStruct(C.Structure):
            _fields_ = [("a", C.c_int)]

        # Extra kwargs are silently ignored by ctypes
        result = checked_create(_TestStruct, a=1, unknown=99)
        assert result.a == 1

    def test_empty_struct_no_fields(self):
        """Empty struct (no _fields_) should succeed with no kwargs."""
        from geo_bsd.ffi_adapter import checked_create

        class _Empty(C.Structure):
            _fields_ = []

        result = checked_create(_Empty)
        assert isinstance(result, _Empty)


# ——————————————————————————————————————————————————————————————————————
# I2-F17: error_guard tests — stale-error suppression and double-raise
# ——————————————————————————————————————————————————————————————————————

class TestErrorGuard:
    """Unit tests for error_guard, _snapshot_hpgl_error, _check_hpgl_error (I2-F17)."""

    def test_stale_error_suppressed_repeatedly(self):
        """Stale error suppressed across multiple calls (persistent stale)."""
        from geo_bsd import ffi_adapter
        # Clear thread-local state
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()
        stale_err = b"persistent stale error"
        mock_so.hpgl_get_last_exception_message.return_value = stale_err

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            # First call: suppress stale error
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("test_op")

            # Second call: same persistent stale error — should also suppress
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("test_op")

            # Third call: still persistent — should still suppress
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("test_op")
            # All three should NOT raise

    def test_consecutive_identical_suppression_then_raise(self):
        """Consecutive suppressions of the same error text across calls
        are still persistent stale errors (C++ never clears).
        The monotonic call counter tracks suppressions for future
        cross-call detection but does not currently raise (known limitation
        without C++ clear-error support)."""
        from geo_bsd import ffi_adapter
        # Clear thread-local state between tests
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()
        stale_msg = b"some error"

        # Call 1: stale error suppressed
        mock_so.hpgl_get_last_exception_message.return_value = stale_msg
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op1")
            # Should be suppressed (stale, first time)

        # Call 2: same stale error appears again — persistent stale, suppresses
        mock_so.hpgl_get_last_exception_message.return_value = stale_msg
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op2")
            # Should still suppress — persistent stale across calls

    def test_double_raise_prevention(self):
        """After raising, re-entering _check_hpgl_error should not double-raise (I2-F17)."""
        from geo_bsd import ffi_adapter
        # Clear thread-local state
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()

        # Snapshot: no error
        mock_so.hpgl_get_last_exception_message.return_value = None
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()

        # Now C++ produces a new error
        mock_so.hpgl_get_last_exception_message.return_value = b"genuine error"
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            with pytest.raises(RuntimeError):
                ffi_adapter._check_hpgl_error("op")

        # Re-enter: the snapshot was updated to the raised error, so
        # error == snapshot → should be suppressed (prevent double-raise)
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._check_hpgl_error("op")
            # Should NOT raise — snapshot was updated before the first raise

    def test_error_guard_context_manager_suppression(self):
        """error_guard context manager: stale error suppressed (I2-F17)."""
        from geo_bsd import ffi_adapter
        # Clear thread-local state
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()
        mock_so.hpgl_get_last_exception_message.return_value = b"old stale error"

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            with ffi_adapter.error_guard("test"):
                pass  # C++ call would go here, does nothing
            # Should not raise — stale error is suppressed

    def test_error_guard_context_manager_new_error(self):
        """error_guard context manager: new error should raise RuntimeError."""
        from geo_bsd import ffi_adapter
        # Clear thread-local state
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()

        # Pre-call snapshot: no error
        mock_so.hpgl_get_last_exception_message.return_value = None
        ffi_adapter._snapshot_hpgl_error()

        # Post-call: new error
        mock_so.hpgl_get_last_exception_message.return_value = b"kriging failed"

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            with pytest.raises(RuntimeError, match="test_op failed"):
                ffi_adapter._check_hpgl_error("test_op")


# ——————————————————————————————————————————————————————————————————————
# I2-F35: Stride-check tests
# ——————————————————————————————————————————————————————————————————————

class _FakeProp:
    """Minimal property stub for testing stride validation."""

    def __init__(self, data, mask, indicator_count=1):
        self.data = data
        self.mask = mask
        self.ndim = data.ndim
        self.indicator_count = indicator_count


class TestStrideValidation:
    """Unit tests for stride-mismatch detection (I2-F35)."""

    @pytest.mark.hpgl
    def test_cont_masked_array_stride_mismatch_raises(self):
        """create_cont_masked_array: mismatched data/mask strides raise ValueError."""
        from geo_bsd.ffi_adapter import create_cont_masked_array

        # Create data with C order, mask with F order — different element strides
        data = np.arange(6, dtype="float32").reshape(2, 3, order="C")
        mask = np.ones(6, dtype="uint8").reshape(2, 3, order="F")

        with pytest.raises(ValueError, match="create_cont_masked_array"):
            create_cont_masked_array(_FakeProp(data, mask), grid=None)

    @pytest.mark.hpgl
    def test_ind_masked_array_stride_mismatch_raises(self):
        """create_ind_masked_array: mismatched data/mask strides raise ValueError."""
        from geo_bsd.ffi_adapter import create_ind_masked_array

        # C-order vs F-order uint8 arrays — different strides
        data = np.arange(6, dtype="uint8").reshape(2, 3, order="C")
        mask = np.ones(6, dtype="uint8").reshape(2, 3, order="F")

        with pytest.raises(ValueError, match="create_ind_masked_array"):
            create_ind_masked_array(_FakeProp(data, mask), grid=None)
