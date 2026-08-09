"""
F-123: Unit tests for the Python ctypes FFI wrapper contract.

Verifies the ctypes interface contract (argument types, return types, error
handling) across hpgl_wrap / ffi_adapter. Tests that import geo_bsd modules
(which load the compiled HPGL library at import time) are marked
@pytest.mark.hpgl and auto-skip on library-less environments; tests importing
cvariogram use guarded try/except skips.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestHPGLWrapContract:
    """Verify the ctypes interface contract for hpgl_wrap module.

    All tests here import geo_bsd.hpgl_wrap / geo_bsd.geo, which load the
    compiled HPGL library at import time — the @pytest.mark.hpgl marker makes
    them skip (not error) on library-less environments.
    """

    @pytest.mark.hpgl
    def test_safe_load_library_error_handling(self):
        """F-123: _safe_load_library handles missing library gracefully."""
        from geo_bsd.hpgl_wrap import _safe_load_library

        # Try to load a non-existent library — must raise an error
        with pytest.raises((ImportError, OSError)):
            _safe_load_library("nonexistent_library_xyz", __file__)

    # ---- F-213: _safe_load_library exception path tests ----

    @pytest.mark.hpgl
    def test_safe_load_library_path_escape_raises(self, tmp_path):
        """F-213: _safe_load_library raises ValueError when resolved path escapes lib_dir.

        H-03 rewrite: the escape is exercised through REAL filesystem paths —
        a lib_name containing '..' resolves to a real file outside the
        reference file's directory, so the relative_to(lib_dir) guard fires.
        No pathlib.Path class patching (the old global monkeypatch family was
        a pytest INTERNALERROR runner-breaker under -v).
        """
        from geo_bsd import hpgl_wrap

        lib_dir = tmp_path / "libdir"
        lib_dir.mkdir()
        # Real file the '..'-escaping candidate path resolves to, so the search
        # loop enters the exists() branch and the escape check fires. All
        # platform extensions are created so the first candidate triggers.
        for suffix in (".dylib", ".so", ".dll", ".pyd"):
            (tmp_path / f"outside{suffix}").write_bytes(b"")

        with pytest.raises(ValueError, match="outside allowed directory"):
            hpgl_wrap._safe_load_library(
                "../outside", str(lib_dir / "ref.py")
            )

    @pytest.mark.hpgl
    def test_safe_load_library_oserror_continue(self, monkeypatch, tmp_path):
        """F-213: _safe_load_library continues to next path on OSError from _load_lib_func.

        H-03/N2-16 rewrite: no pathlib.Path class patching. Real candidate
        library files exist in lib_dir, and _load_lib_func is patched (module
        attribute only) to raise OSError for every candidate, so the
        continue-on-OSError loop + fallback re-raise path is exercised.
        """
        from geo_bsd import hpgl_wrap

        lib_dir = tmp_path / "libdir"
        lib_dir.mkdir()
        # Create real candidate files for the running platform so the search
        # loop enters the exists() branch and reaches _load_lib_func.
        if sys.platform.startswith("win"):
            candidate_names = [
                "test_lib.dll",
                "test_lib.pyd",
                "libtest_lib.dll",
                "test_lib_d.dll",
                "libtest_lib_d.dll",
            ]
        elif sys.platform.startswith("darwin"):
            candidate_names = [
                "test_lib.dylib",
                "test_lib.so",
                "libtest_lib.dylib",
                "libtest_lib.so",
            ]
        else:
            candidate_names = ["test_lib.so", "libtest_lib.so"]
        for name in candidate_names:
            (lib_dir / name).write_bytes(b"")

        call_count = [0]

        def mock_load_func(path):
            call_count[0] += 1
            raise OSError("Incompatible library")

        monkeypatch.setattr(hpgl_wrap, "_load_lib_func", mock_load_func)

        # Should cycle through all platform paths + fallback, eventually
        # raising OSError
        with pytest.raises(OSError, match="Cannot load library"):
            hpgl_wrap._safe_load_library("test_lib", str(lib_dir / "ref.py"))

        # Verify _load_lib_func was called multiple times (OSError was
        # caught, continued)
        assert call_count[0] >= 2, (
            f"Expected _load_lib_func to be called at least 2 times (continue on OSError), "
            f"got {call_count[0]}"
        )

    @pytest.mark.hpgl
    def test_error_local_thread_safe(self):
        """F-123: Thread-local error storage exists in geo.py (ctypes FFI layer)."""
        from geo_bsd import geo

        assert hasattr(geo, "_error_local"), (
            "geo._error_local must exist for thread-safe error tracking"
        )
        assert isinstance(geo._error_local, threading.local), (
            "_error_local must be a threading.local instance"
        )


class TestCVariogramContract:
    """Verify the ctypes interface contract for cvariogram module."""

    def test_cvar_error_local_thread_safe(self):
        """F-123: Thread-local error storage exists in cvariogram."""
        try:
            from geo_bsd import cvariogram
        except ImportError:
            pytest.skip("cvariogram module not available")
        assert hasattr(cvariogram, "_cvar_error_local"), "cvariogram._cvar_error_local must exist"
        assert isinstance(cvariogram._cvar_error_local, threading.local), (
            "_cvar_error_local must be threading.local"
        )

    def test_cvar_call_lock_exists(self):
        """F-123: Call lock exists in cvariogram for thread safety."""
        try:
            from geo_bsd import cvariogram
        except ImportError:
            pytest.skip("cvariogram module not available")
        assert hasattr(cvariogram, "_cvar_call_lock"), "cvariogram._cvar_call_lock must exist"
        assert isinstance(cvariogram._cvar_call_lock, type(threading.Lock())), (
            "_cvar_call_lock must be a threading.Lock"
        )

    def test_max_num_lags_constant(self):
        """F-123: MAX_NUM_LAGS constant prevents runaway allocation."""
        try:
            from geo_bsd import cvariogram
        except ImportError:
            pytest.skip("cvariogram module not available")
        assert hasattr(cvariogram, "MAX_NUM_LAGS"), "MAX_NUM_LAGS must be defined"
        assert cvariogram.MAX_NUM_LAGS > 0, "MAX_NUM_LAGS must be positive"
        assert cvariogram.MAX_NUM_LAGS == 10000, (
            f"MAX_NUM_LAGS should be 10000, got {cvariogram.MAX_NUM_LAGS}"
        )

    def test_max_point_set_size_constant(self):
        """F-123: MAX_POINT_SET_SIZE prevents OOM in variogram.

        Exact-value pin (RC-3 work-cap class, mirrors MAX_NUM_LAGS) — a
        silently-shrunken cap would otherwise pass a mere ``> 0`` check.
        """
        try:
            from geo_bsd import cvariogram
        except ImportError:
            pytest.skip("cvariogram module not available")
        assert hasattr(cvariogram, "MAX_POINT_SET_SIZE"), "MAX_POINT_SET_SIZE must be defined"
        assert cvariogram.MAX_POINT_SET_SIZE == 1_000_000, (
            f"MAX_POINT_SET_SIZE should be 1000000, got {cvariogram.MAX_POINT_SET_SIZE}"
        )


class _FakePropSize:
    """Property stub with distinct data/mask sizes (F-24/F-25)."""

    def __init__(self, data, mask, indicator_count=1):
        self.data = data
        self.mask = mask
        self.ndim = data.ndim
        self.indicator_count = indicator_count


class _FakeGrid:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class TestMaskedArraySizeValidation:
    """F-24/F-25: grid=None paths must validate data.size == mask.size.

    C++ indexes both arrays with the same stride values, so a
    data.size != mask.size mismatch reads/writes past the end of the shorter
    array (heap OOB read/write). The stride check alone misses length
    mismatches.
    """

    @pytest.mark.hpgl
    def test_cont_masked_array_size_mismatch_raises(self):
        from geo_bsd.ffi_adapter import create_cont_masked_array

        data = np.arange(6, dtype="float32").reshape(2, 3, order="F")
        mask = np.ones(3, dtype="uint8").reshape(1, 3, order="F")  # size 3 vs 6

        with pytest.raises(ValueError, match="data size 6 does not match mask size 3"):
            create_cont_masked_array(_FakePropSize(data, mask), grid=None)

    @pytest.mark.hpgl
    def test_ind_masked_array_size_mismatch_raises(self):
        from geo_bsd.ffi_adapter import create_ind_masked_array

        data = np.arange(6, dtype="uint8").reshape(2, 3, order="F")
        mask = np.ones(3, dtype="uint8").reshape(1, 3, order="F")

        with pytest.raises(ValueError, match="data size 6 does not match mask size 3"):
            create_ind_masked_array(_FakePropSize(data, mask), grid=None)

    @pytest.mark.hpgl
    def test_cont_masked_array_equal_sizes_still_constructs(self):
        from geo_bsd.ffi_adapter import create_cont_masked_array

        data = np.arange(6, dtype="float32").reshape(2, 3, order="F")
        mask = np.ones(6, dtype="uint8").reshape(2, 3, order="F")
        result = create_cont_masked_array(_FakePropSize(data, mask), grid=None)
        assert result._array_refs == (data, mask)

    @pytest.mark.hpgl
    def test_ind_masked_array_equal_sizes_still_constructs(self):
        from geo_bsd.ffi_adapter import create_ind_masked_array

        data = np.arange(6, dtype="uint8").reshape(2, 3, order="F")
        mask = np.ones(6, dtype="uint8").reshape(2, 3, order="F")
        result = create_ind_masked_array(_FakePropSize(data, mask), grid=None)
        assert result._array_refs == (data, mask)


class TestFloatArrayGridSizeValidation:
    """F-26: create_float_array must validate grid size in the grid path.

    Pre-fix a float array smaller than the grid volume was passed to C++
    with a grid-sized shape struct, and the kernel read past the end of the
    buffer (heap OOB read).
    """

    @pytest.mark.hpgl
    def test_grid_size_mismatch_raises(self):
        from geo_bsd.ffi_adapter import create_float_array

        arr = np.zeros(5, dtype="float32", order="F")
        with pytest.raises(RuntimeError, match="Invalid data size"):
            create_float_array(arr, _FakeGrid(2, 2, 2))  # volume 8 != 5

    @pytest.mark.hpgl
    def test_grid_size_match_still_constructs(self):
        from geo_bsd.ffi_adapter import create_float_array

        arr = np.zeros(8, dtype="float32", order="F")
        result = create_float_array(arr, _FakeGrid(2, 2, 2))
        assert result._array_ref is arr

    @pytest.mark.hpgl
    def test_grid_none_still_constructs(self):
        from geo_bsd.ffi_adapter import create_float_array

        arr = np.zeros((2, 2, 2), dtype="float32", order="F")
        result = create_float_array(arr, None)
        assert tuple(result.shape.m_data) == (2, 2, 2)


class TestUbyteArrayShapePreservation:
    """III-14: create_ubyte_array must preserve the caller's 3D mask shape.

    Pre-fix the shape was stamped to (grid.x, grid.y, grid.z), so an
    equal-volume (2,8,1) mask on a (4,4,1) grid passed the volume check AND
    the C++ per-dimension shape guard (the stamped shape always matched the
    grid), silently permuting the simulated cells.
    """

    @pytest.mark.hpgl
    def test_preserves_3d_caller_shape(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        mask3d = np.ones((2, 8, 1), dtype="uint8", order="F")  # volume 16
        result = create_ubyte_array(mask3d, _FakeGrid(4, 4, 1))
        assert tuple(result.shape.m_data) == (2, 8, 1), (
            "III-14: create_ubyte_array must preserve the caller's 3D mask "
            "shape so the C++ per-dim guard can fire"
        )

    @pytest.mark.hpgl
    def test_flat_mask_uses_grid_dims(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        flat = np.ones(16, dtype="uint8", order="F")
        result = create_ubyte_array(flat, _FakeGrid(4, 4, 1))
        assert tuple(result.shape.m_data) == (4, 4, 1)

    @pytest.mark.hpgl
    def test_volume_mismatch_still_raises(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        small = np.ones(8, dtype="uint8", order="F")
        with pytest.raises(RuntimeError, match="Invalid data size"):
            create_ubyte_array(small, _FakeGrid(4, 4, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
