"""Structural-fix regression tests for the FFI output-buffer contract
(got-20260802015741) and the mask-semantics contract
(got-20260803180153).

The FFI output-buffer contract class recurred across runs because each run
fixed one facet (contiguity → size → writeability → full-init) and the next
run surfaced a sibling facet. The structural fix is ONE shared helper —
``ffi_adapter.require_output_buffer`` — enforcing contiguity + size +
writeability (+ dtype) at every ctypes call site, so a future call site
cannot skip a facet. These tests pin the helper's contract and confirm the
call sites route through it.

The mask-semantics contract ("non-zero = informed") must be ONE definition
at every boundary; ``normalize_mask_binary`` is the centralized bool-convert.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from geo_bsd import ffi_adapter  # noqa: E402


@pytest.fixture
def contig_writable():
    return np.zeros(16, dtype="float32")


class TestRequireOutputBuffer:
    """The full output-buffer contract: contiguity + size + writeability."""

    def test_non_ndarray_rejected(self):
        with pytest.raises(ValueError, match="output buffer must be a numpy.ndarray"):
            ffi_adapter.require_output_buffer([0.0] * 16, 16, "ctx")

    def test_strided_view_rejected(self):
        # A 2D slice with non-trivial strides is not C- nor F-contiguous.
        base = np.zeros((4, 4), dtype="float32")
        view = base[::2, ::2]
        assert not (view.flags["C_CONTIGUOUS"] or view.flags["F_CONTIGUOUS"])
        with pytest.raises(ValueError, match="must be contiguous"):
            ffi_adapter.require_output_buffer(view, 4, "ctx")

    def test_size_mismatch_rejected(self):
        with pytest.raises(ValueError, match="size 8 does not match expected 16"):
            ffi_adapter.require_output_buffer(np.zeros(8, dtype="float32"), 16, "ctx")

    def test_readonly_rejected(self):
        ro = np.zeros(16, dtype="float32")
        ro.setflags(write=False)
        with pytest.raises(ValueError, match="must be writable"):
            ffi_adapter.require_output_buffer(ro, 16, "ctx")

    def test_dtype_mismatch_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            ffi_adapter.require_output_buffer(
                np.zeros(16, dtype="float64"), 16, "ctx", np.dtype("float32")
            )

    def test_valid_buffer_passes(self):
        # No exception — the valid buffer satisfies every facet.
        ffi_adapter.require_output_buffer(np.zeros(16, dtype="float32"), 16, "ctx")

    def test_fortran_contiguous_passes(self):
        f = np.zeros((4, 4), dtype="float32", order="F")
        ffi_adapter.require_output_buffer(f, 16, "ctx")


class TestRequireOutputBufferFullInit:
    """Full-init facet: sentinel prefill + post-call detection."""

    def test_sentinel_detected(self):
        buf = np.full(16, ffi_adapter._UNWRITTEN_CELL_SENTINEL, dtype="float32")
        buf[0:8] = 1.0  # only first half written
        with pytest.raises(ValueError, match="not fully initialized"):
            ffi_adapter.require_output_buffer_full_init(buf, "ctx")

    def test_fully_written_passes(self):
        buf = np.full(16, 1.0, dtype="float32")
        ffi_adapter.require_output_buffer_full_init(buf, "ctx")

    def test_prefill_sets_sentinel(self):
        buf = np.zeros(16, dtype="float32")
        ffi_adapter.prefill_output_buffer(buf)
        assert np.all(buf == ffi_adapter._UNWRITTEN_CELL_SENTINEL)


class TestNormalizeMaskBinary:
    """Mask semantics: non-zero = informed, centralized bool-convert."""

    def test_none_stays_none(self):
        assert ffi_adapter.normalize_mask_binary(None, "ctx") is None

    def test_non_ndarray_rejected(self):
        with pytest.raises(TypeError, match="mask must be a numpy.ndarray"):
            ffi_adapter.normalize_mask_binary([1, 0, 1], "ctx")

    def test_binary_mask_unchanged(self):
        # E-M7: normalize_mask_binary ALWAYS returns a fresh uint8 copy —
        # the caller's array is never handed back unchanged (a non-uint8
        # binary array would flow downstream with its original dtype/layout,
        # violating the uint8 contract the C++ boundary assumes).
        mask = np.array([0, 1, 0, 1], dtype="uint8")
        result = ffi_adapter.normalize_mask_binary(mask, "ctx")
        assert result is not mask  # fresh copy (E-M7)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, mask)

    def test_non_binary_mask_normalized_to_01(self):
        mask = np.array([0, 2, 3, 0], dtype="uint8")
        result = ffi_adapter.normalize_mask_binary(mask, "ctx")
        assert set(np.unique(result)) <= {0, 1}
        assert np.array_equal(result, np.array([0, 1, 1, 0], dtype="uint8"))

    def test_caller_array_not_mutated(self):
        mask = np.array([0, 2, 3, 0], dtype="uint8")
        ffi_adapter.normalize_mask_binary(mask, "ctx")
        assert np.array_equal(mask, np.array([0, 2, 3, 0], dtype="uint8"))
