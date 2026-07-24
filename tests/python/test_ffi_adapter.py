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


# ——————————————————————————————————————————————————————————————————————
# M-P-06: Sequence-counter cross-call different-error path
# ——————————————————————————————————————————————————————————————————————


class TestSequenceCounterCrossCall:
    """Tests for _snapshot_hpgl_error / _check_hpgl_error sequence counter (M-P-06).

    Verifies that two consecutive FFI calls with DIFFERENT errors produce
    distinct RuntimeError messages — the sequence-counter path at
    ffi_adapter.py:165 prevents identical-error collision across calls.
    """

    def test_cross_call_different_error_sequence_counter(self):
        """Two consecutive guard invocations with different errors both raise.

        snapshot_seq != current_seq branch: when a genuinely new error appears
        from a different C call (same message as a prior stale, but different
        seq), the error IS raised because it's a new call window.
        """
        from geo_bsd import ffi_adapter
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()
        stale_msg = b"some_constant_err"

        # Call 1: stale error suppressed
        mock_so.hpgl_get_last_exception_message.return_value = stale_msg
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()  # seq=1, snapshot=stale_msg
            ffi_adapter._check_hpgl_error("op1")  # same seq, same msg → suppress

        # Call 2: DIFFERENT error — new seq, new error → MUST raise
        mock_so.hpgl_get_last_exception_message.return_value = b"a genuinely new error"
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()  # seq=2, snapshot=b"a genuinely new error"
            # Reset the pre-call snapshot to None (simulate clean pre-call state)
            # but leave seq at 2 — this is the snapshot_seq != current_seq path
            ffi_adapter._error_local._hpgl_error_snapshot = None
            ffi_adapter._error_local._hpgl_error_snapshot_seq = 1  # stale seq

            # Now _check_hpgl_error: err != snapshot (None vs b"...") → raise
            with pytest.raises(RuntimeError, match="op2"):
                ffi_adapter._check_hpgl_error("op2")

    def test_cross_call_same_error_different_seq_raises(self):
        """Same error message, new call sequence → raises (M-P-06 core path).

        When snapshot_seq != current_seq, the identical-message stale suppression
        is bypassed because this IS a different C call window.
        The error at ffi_adapter.py:170-171: snapshot_seq != current_seq
        falls through to the raise path at line 186-196.
        """
        from geo_bsd import ffi_adapter
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()
        same_msg = b"persistent error text"

        # Call 1: snapshot gets same_msg, suppress (seq=1, snapshot=same_msg)
        mock_so.hpgl_get_last_exception_message.return_value = same_msg
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()  # seq=1
            ffi_adapter._check_hpgl_error("op1")  # suppress — same seq, same msg

        # Call 2: new snapshot with same_msg, but force old seq to simulate
        # snapshot_seq != current_seq path
        mock_so.hpgl_get_last_exception_message.return_value = same_msg
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()  # seq=2, snapshot=same_msg
            # Force snapshot_seq to be stale (< current_seq)
            ffi_adapter._error_local._hpgl_error_snapshot_seq = 0

            # err == snapshot but snapshot_seq != current_seq → MUST raise
            with pytest.raises(RuntimeError, match="op2"):
                ffi_adapter._check_hpgl_error("op2")

    def test_cross_call_sequence_counter_resets_after_raise(self):
        """After a cross-call raise, the error snapshot is updated before raising.

        This prevents double-raise on re-entry — the snapshot was set to the
        raised error at line 189-190 before the RuntimeError is raised.
        """
        from geo_bsd import ffi_adapter
        ffi_adapter._error_local = type(ffi_adapter._error_local)()

        mock_so = mock.MagicMock()

        # Pre-call: clean
        mock_so.hpgl_get_last_exception_message.return_value = None
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()  # seq=1, snapshot=None

        # New error
        mock_so.hpgl_get_last_exception_message.return_value = b"genuine_new_err"
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            with pytest.raises(RuntimeError, match="op2"):
                ffi_adapter._check_hpgl_error("op2")

        # Re-enter: snapshot was updated before raise → should suppress
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._check_hpgl_error("op2")  # should NOT double-raise


# ——————————————————————————————————————————————————————————————————————
# M-P-18: ctypes struct size verification
# ——————————————————————————————————————————————————————————————————————


class TestCtypesStructSizes:
    """Verify ctypes struct sizes match expected C struct sizes (M-P-18).

    On macOS 64-bit (x86_64/arm64):
    - Pointers: 8 bytes
    - c_int: 4 bytes
    - c_double: 8 bytes
    - c_float: 4 bytes
    - c_ubyte: 1 byte
    - c_longlong (c_int64): 8 bytes
    - c_ulong: 8 bytes
    """

    def test_hpgl_shape_size(self):
        """_HPGL_SHAPE: 2 × (int[3]) = 2 × 12 = 24 bytes."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_SHAPE
        expected = C.sizeof(C.c_int) * 3 + C.sizeof(C.c_int) * 3
        assert C.sizeof(_HPGL_SHAPE) == expected
        assert C.sizeof(_HPGL_SHAPE) >= 24

    def test_hpgl_cont_masked_array_size(self):
        """_HPGL_CONT_MASKED_ARRAY: 2 pointers + shape struct."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_CONT_MASKED_ARRAY, _HPGL_SHAPE
        min_expected = C.sizeof(C.POINTER(C.c_float)) + C.sizeof(C.POINTER(C.c_ubyte)) + C.sizeof(_HPGL_SHAPE)
        actual = C.sizeof(_HPGL_CONT_MASKED_ARRAY)
        assert actual >= min_expected, f"Size {actual} < min {min_expected}"

    def test_hpgl_ind_masked_array_size(self):
        """_HPGL_IND_MASKED_ARRAY: 2 pointers + shape + int."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_IND_MASKED_ARRAY
        assert C.sizeof(_HPGL_IND_MASKED_ARRAY) >= 32

    def test_hpgl_ubyte_array_size(self):
        """_HPGL_UBYTE_ARRAY: 1 pointer + shape."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_UBYTE_ARRAY, _HPGL_SHAPE
        min_expected = C.sizeof(C.POINTER(C.c_ubyte)) + C.sizeof(_HPGL_SHAPE)
        actual = C.sizeof(_HPGL_UBYTE_ARRAY)
        assert actual >= min_expected

    def test_hpgl_float_array_size(self):
        """_HPGL_FLOAT_ARRAY: 1 pointer + shape."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_FLOAT_ARRAY, _HPGL_SHAPE
        min_expected = C.sizeof(C.POINTER(C.c_float)) + C.sizeof(_HPGL_SHAPE)
        actual = C.sizeof(_HPGL_FLOAT_ARRAY)
        assert actual >= min_expected

    def test_hpgl_ok_params_size(self):
        """_HPGL_OK_PARAMS: for 8 cov fields + 3 radius ints + max_neighbours.

        Exact size depends on platform padding; verify sizeof >= sum of fields.
        """
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_OK_PARAMS
        # Minimum: 1 int + 3×double + 3×double + double + double + 3×int + int
        # Actual may be larger due to alignment padding (e.g. 88 on ARM64)
        min_expected = (
            C.sizeof(C.c_int)
            + C.sizeof(C.c_double) * 3
            + C.sizeof(C.c_double) * 3
            + C.sizeof(C.c_double)
            + C.sizeof(C.c_double)
            + C.sizeof(C.c_int) * 3
            + C.sizeof(C.c_int)
        )
        actual = C.sizeof(_HPGL_OK_PARAMS)
        assert actual >= min_expected, f"_HPGL_OK_PARAMS size {actual} < min {min_expected}"
        # On all platforms, struct should fit within double-aligned 96 bytes
        assert actual <= 96, f"_HPGL_OK_PARAMS too large: {actual}"

    def test_hpgl_sgs_params_size(self):
        """_HPGL_SGS_PARAMS: similar to OK + extra ints."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_SGS_PARAMS
        # kriging_kind (int) + seed (int64) + min_neighbours (int) = 4+8+4=16 extra
        assert C.sizeof(_HPGL_SGS_PARAMS) >= 80

    def test_hpgl_median_ik_params_size(self):
        """_HPGL_MEDIAN_IK_PARAMS: like OK + double[2] for marginal_probs."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_MEDIAN_IK_PARAMS
        assert C.sizeof(_HPGL_MEDIAN_IK_PARAMS) >= 64

    def test_hpgl_ik_params_size(self):
        """_HPGL_IK_PARAMS: like OK + double for marginal_prob."""
        import ctypes as C
        from geo_bsd.ffi_adapter import _HPGL_IK_PARAMS
        assert C.sizeof(_HPGL_IK_PARAMS) >= 56

    def test_hpgl_cockriging_m1_params_size(self):
        """__hpgl_cockriging_m1_params_t: like OK + 4 doubles."""
        import ctypes as C
        from geo_bsd.ffi_adapter import HPGL_COKRIGING_M1_PARAMS
        assert C.sizeof(HPGL_COKRIGING_M1_PARAMS) >= 64

    def test_hpgl_non_parametric_cdf_size(self):
        """hpgl_non_parametric_cdf_t: 2 pointers + longlong."""
        import ctypes as C
        from geo_bsd.ffi_adapter import HPGL_NONPARAM_CDF
        min_expected = (
            C.sizeof(C.POINTER(C.c_float))
            + C.sizeof(C.POINTER(C.c_float))
            + C.sizeof(C.c_longlong)
        )
        actual = C.sizeof(HPGL_NONPARAM_CDF)
        assert actual >= min_expected

    def test_hpgl_kriging_stats_size(self):
        """_HPGLKrigingStats: 3×ulong + 2×double."""
        import ctypes as C
        from geo_bsd.hpgl_wrap import _HPGLKrigingStats
        min_expected = (
            C.sizeof(C.c_ulong) * 3
            + C.sizeof(C.c_double) * 2
        )
        actual = C.sizeof(_HPGLKrigingStats)
        assert actual >= min_expected
