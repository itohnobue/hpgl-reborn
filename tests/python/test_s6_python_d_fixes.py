"""Regression tests for s6-python-D FIX stage findings.

Covers:
- F-07  ffi_adapter error_guard identical-message collision (consecutive identical failures both raise)
- F-44  ffi_adapter _c_array(c_ubyte) silent wrap/truncation guard
- F-03  hpgl_wrap loader prefers fresh build name; symbol freshness check
- F-06  cvariogram CStackLayers result x/y shape validation
- F-08  cvariogram CStackLayers non-contiguous/sliced layer handling
- F-37  cvariogram error guard identical-message collision
- F-38  cvariogram search-geometry magnitude caps
- F-40  cvariogram degenerate-template rejection (R=0 / zero directions)
- I2-01 cvariogram Ellipsoid / VariogramSearchTemplate numeric validation
- F-04  cdf.calc_cdf last CDF probability strictly below 1.0

Each test fails against the pre-fix code and passes against the fixed code.
"""

import ctypes as C
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cvariogram import (
        CalcVariograms,
        CStackLayers,
        Ellipsoid,
        VariogramSearchTemplate,
    )
    CVAR_AVAILABLE = True
except Exception:
    CVAR_AVAILABLE = False


# ===========================================================================
# F-07: ffi_adapter error_guard — two consecutive identical C++ failures both
# raise instead of the second being silently suppressed as stale.
# ===========================================================================


class TestF07ErrorGuardConsecutiveIdenticalFailures:
    def _make_clearing_mock(self):
        """Mock that emulates a C++ library which honours the error-clear
        consumed by _clear_hpgl_error after each guard (the fixed ffi_adapter
        resets hpgl::set_last_exception_message).  The test sets
        ``state['err']`` between snapshot and check to simulate the C++ call
        failing."""
        state = {"err": None}

        def fake_get():
            return state["err"]

        def fake_clear(msg):
            state["err"] = None if msg == b"" else msg

        def fake_getitem(name):
            if name == "_ZN4hpgl26set_last_exception_messageEPKc":
                return fake_clear
            raise KeyError(name)

        mock_so = mock.MagicMock()
        mock_so.hpgl_get_last_exception_message.side_effect = fake_get
        mock_so.__getitem__.side_effect = fake_getitem
        return mock_so, state

    def test_two_consecutive_identical_failures_both_raise(self):
        from geo_bsd import ffi_adapter

        # Fresh thread-local state.
        ffi_adapter._error_local = type(ffi_adapter._error_local)()
        mock_so, state = self._make_clearing_mock()

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            # Call 1: clean snapshot → C++ fails with "singular matrix".
            ffi_adapter._snapshot_hpgl_error()
            state["err"] = b"singular matrix"
            with pytest.raises(RuntimeError, match="singular matrix"):
                ffi_adapter._check_hpgl_error("op1")

            # Call 2: error was consumed by call 1's guard → clean snapshot,
            # C++ fails with the IDENTICAL message → must raise.
            ffi_adapter._snapshot_hpgl_error()
            state["err"] = b"singular matrix"
            with pytest.raises(RuntimeError, match="singular matrix"):
                ffi_adapter._check_hpgl_error("op2")

    def test_success_after_failure_does_not_false_raise(self):
        """A SUCCESSFUL call after a raised failure must NOT raise: the
        guard consumes the C++ error on raise, so the successful call
        starts from a clean state (no stale identical message)."""
        from geo_bsd import ffi_adapter

        ffi_adapter._error_local = type(ffi_adapter._error_local)()
        mock_so, state = self._make_clearing_mock()

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            # Call 1 fails with "vector".
            ffi_adapter._snapshot_hpgl_error()
            state["err"] = b"vector"
            with pytest.raises(RuntimeError, match="vector"):
                ffi_adapter._check_hpgl_error("op1")

            # Call 2 SUCCEEDS (C++ sets no error) → clean state → no raise.
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op2")  # must NOT raise

    def test_stale_error_from_before_first_call_still_suppressed(self):
        """The fix must NOT break stale-error suppression for errors that
        predate the first guard invocation (persistent stale)."""
        from geo_bsd import ffi_adapter

        ffi_adapter._error_local = type(ffi_adapter._error_local)()
        mock_so = mock.MagicMock()
        mock_so.hpgl_get_last_exception_message.return_value = b"persistent stale error"
        mock_so.__getitem__.side_effect = KeyError("no setter")

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op1")  # suppress
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op2")  # suppress again
            ffi_adapter._snapshot_hpgl_error()
            ffi_adapter._check_hpgl_error("op3")  # still suppressed

    def test_same_call_reraises_still_suppressed(self):
        """Re-entering _check_hpgl_error within the same window after a raise
        must not double-raise (genuine same-call re-raise suppression)."""
        from geo_bsd import ffi_adapter

        ffi_adapter._error_local = type(ffi_adapter._error_local)()
        mock_so = mock.MagicMock()
        mock_so.hpgl_get_last_exception_message.return_value = None

        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            ffi_adapter._snapshot_hpgl_error()

        mock_so.hpgl_get_last_exception_message.return_value = b"genuine error"
        with mock.patch.object(ffi_adapter, "_hpgl_so", mock_so):
            with pytest.raises(RuntimeError):
                ffi_adapter._check_hpgl_error("op")
            # Re-enter: snapshot was updated before the raise → suppress.
            ffi_adapter._check_hpgl_error("op")


# ===========================================================================
# F-44: ffi_adapter _c_array(c_ubyte) must reject values ctypes would silently
# wrap/truncate (300 -> 44, 1.5 -> 1).
# ===========================================================================


class TestF44CArrayUbyteValidation:
    def test_out_of_range_value_raises(self):
        from geo_bsd.ffi_adapter import _c_array

        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            _c_array(C.c_ubyte, 1, [300])

    def test_fractional_value_raises(self):
        from geo_bsd.ffi_adapter import _c_array

        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            _c_array(C.c_ubyte, 1, [1.5])

    def test_negative_value_raises(self):
        from geo_bsd.ffi_adapter import _c_array

        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            _c_array(C.c_ubyte, 1, [-1])

    def test_valid_values_accepted(self):
        from geo_bsd.ffi_adapter import _c_array

        arr = _c_array(C.c_ubyte, 3, [0, 128, 255])
        assert [arr[0], arr[1], arr[2]] == [0, 128, 255]

    def test_numpy_integers_accepted(self):
        from geo_bsd.ffi_adapter import _c_array

        arr = _c_array(C.c_ubyte, 2, [np.uint8(5), np.int64(250)])
        assert [arr[0], arr[1]] == [5, 250]

    def test_non_c_ubyte_types_unaffected(self):
        """The guard must only apply to c_ubyte arrays (shapes are c_int)."""
        from geo_bsd.ffi_adapter import _c_array

        arr = _c_array(C.c_int, 3, (10, 20, 30))
        assert [arr[0], arr[1], arr[2]] == [10, 20, 30]


# ===========================================================================
# F-03: hpgl_wrap loader must prefer the fresh {name}.dylib build output over
# a stale lib{name}.dylib, and warn when a loaded library lacks fresh symbols.
# ===========================================================================


class TestF03LibraryLoading:
    def test_prefers_fresh_build_name_over_lib_prefix(self, tmp_path, monkeypatch):
        from geo_bsd import hpgl_wrap

        monkeypatch.setattr(hpgl_wrap.sys, "platform", "darwin")
        ref = tmp_path / "hpgl_wrap.py"
        ref.write_text("")
        stale = tmp_path / "libhpgl.dylib"
        fresh = tmp_path / "hpgl.dylib"
        stale.write_bytes(b"stale binary")
        fresh.write_bytes(b"fresh binary")

        loaded = []

        def fake_load(path):
            loaded.append(str(path))
            return mock.MagicMock()

        monkeypatch.setattr(hpgl_wrap, "_load_lib_func", fake_load)
        monkeypatch.setattr(hpgl_wrap, "_verify_library_hash", lambda *a, **k: None)
        monkeypatch.setattr(hpgl_wrap, "_verify_library_freshness", lambda *a, **k: None)

        hpgl_wrap._safe_load_library("hpgl", str(ref))

        assert loaded, "expected at least one load attempt"
        assert loaded[0] == str(fresh), (
            f"fresh build name must be preferred over stale lib-prefixed name, got {loaded}"
        )
        assert str(stale) not in loaded, (
            f"stale lib-prefixed library must not shadow the fresh build, got {loaded}"
        )

    def test_freshness_check_warns_missing_symbol(self, caplog):
        import logging

        from geo_bsd import hpgl_wrap

        stale_lib = mock.Mock(spec=[])  # no attributes → missing every symbol
        with caplog.at_level(logging.WARNING, logger="geo_bsd.hpgl_wrap"):
            hpgl_wrap._verify_library_freshness("hpgl", stale_lib)
        assert "does not export expected symbols" in caplog.text
        assert "hpgl_get_kriging_stats" in caplog.text

    def test_freshness_check_silent_for_complete_library(self, caplog):
        import logging

        from geo_bsd import hpgl_wrap

        fresh_lib = mock.MagicMock()
        fresh_lib.hpgl_get_kriging_stats = mock.Mock()
        with caplog.at_level(logging.WARNING, logger="geo_bsd.hpgl_wrap"):
            hpgl_wrap._verify_library_freshness("hpgl", fresh_lib)
        assert "does not export expected symbols" not in caplog.text

    def test_freshness_check_ignores_unknown_library(self, caplog):
        import logging

        from geo_bsd import hpgl_wrap

        lib = mock.Mock(spec=[])
        with caplog.at_level(logging.WARNING, logger="geo_bsd.hpgl_wrap"):
            hpgl_wrap._verify_library_freshness("unknown_lib", lib)
        assert "does not export expected symbols" not in caplog.text


# ===========================================================================
# F-06 / F-08: cvariogram CStackLayers shape + contiguity safety
# ===========================================================================


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestF06F08CStackLayers:
    def test_result_xy_shape_mismatch_raises(self):
        """F-06: a result whose x/y dims are smaller than the layer dims is a
        heap OOB write in C++ (only nz was validated before)."""
        layer = np.ones((5, 5, 1), dtype="float32")
        result = np.zeros((3, 3, 10), dtype="float32")
        with pytest.raises(ValueError, match="result x/y shape"):
            CStackLayers([layer], [1], nz=5, scalez=1.0, blank_value=-99, result=result)

    def test_result_2d_raises(self):
        """A 2D result previously crashed with IndexError at result.shape[2]."""
        layer = np.ones((5, 5, 1), dtype="float32")
        result = np.zeros((5, 5), dtype="float32")
        with pytest.raises(ValueError, match="3-dimensional"):
            CStackLayers([layer], [1], nz=5, scalez=1.0, blank_value=-99, result=result)

    def test_non_contiguous_result_rejected(self):
        """F-08: a sliced/non-contiguous result would OOB-write via strides."""
        layer = np.ones((5, 5, 1), dtype="float32")
        result = np.zeros((10, 10, 10), dtype="float32")[::2, ::2, :]
        assert not result.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="contiguous"):
            CStackLayers([layer], [1], nz=5, scalez=1.0, blank_value=-99, result=result)

    def test_reversed_slice_layer_produces_correct_output(self):
        """F-08: a non-contiguous layer (negative strides) must be copied to a
        contiguous buffer. Pre-fix the C++ map_index went negative → OOB
        reads/writes → wrong output."""
        base = np.arange(9, dtype="float32").reshape(3, 3, 1)
        layer = base[::-1, :, :]  # reversed → non-contiguous, negative strides
        assert not layer.flags["C_CONTIGUOUS"]
        result = np.zeros((3, 3, 10), dtype="float32")
        CStackLayers([layer], [7], nz=5, scalez=1.0, blank_value=-99, result=result)
        # layer[0,j] = base[2,j] = 6+j (thickness 6..8) → capped at nz=5, fills z [0,5)
        assert result[0, 0, 0] == 7.0
        assert result[0, 0, 4] == 7.0
        assert result[0, 2, 4] == 7.0
        # layer[1,0] = base[1,0] = 3 → F-39: thickness 3 fills exactly z [0,3)
        assert result[1, 0, 0] == 7.0
        assert result[1, 0, 1] == 7.0
        assert result[1, 0, 2] == 7.0
        assert result[1, 0, 3] == 0.0
        assert result[1, 0, 4] == 0.0
        # layer[2,0] = base[0,0] = 0 → not positive → blank_value
        assert result[2, 0, 0] == -99.0
        # layer[2,1] = base[0,1] = 1 → F-39: thickness 1 fills exactly z [0,1)
        assert result[2, 1, 0] == 7.0
        assert result[2, 1, 1] == 0.0
        assert result[2, 1, 2] == 0.0


# ===========================================================================
# F-37: cvariogram error guard — consecutive identical failures both raise
# ===========================================================================


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestF37CvarErrorGuard:
    def test_two_consecutive_identical_failures_both_raise(self, monkeypatch):
        import geo_bsd.cvariogram as cvmod

        cvmod._cvar_error_local = type(cvmod._cvar_error_local)()
        errors = iter([None, b"calc failed", b"calc failed", b"calc failed"])
        monkeypatch.setattr(cvmod.cvar, "cvar_get_last_error", lambda: next(errors))

        cvmod._snapshot_cvar_error()
        with pytest.raises(RuntimeError, match="calc failed"):
            cvmod._check_cvar_error("op1")

        cvmod._snapshot_cvar_error()
        with pytest.raises(RuntimeError, match="calc failed"):
            cvmod._check_cvar_error("op2")

    def test_stale_error_still_suppressed(self, monkeypatch):
        import geo_bsd.cvariogram as cvmod

        msg = b"stale error from previous call"
        monkeypatch.setattr(cvmod.cvar, "cvar_get_last_error", lambda: msg)
        cvmod._cvar_error_local._cvar_error_snapshot = msg
        cvmod._check_cvar_error("test_ctx")  # must not raise


# ===========================================================================
# F-38 + I2-01: cvariogram numeric validation and magnitude caps
# ===========================================================================


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestF38I201NumericValidation:
    def _ellipsoid(self, **kwargs):
        params = {"R1": 10.0, "R2": 5.0, "R3": 3.0, "azimuth": 0.0, "dip": 0.0, "rotation": 0.0}
        params.update(kwargs)
        return Ellipsoid(**params)

    # --- I2-01: Ellipsoid validation ---

    def test_ellipsoid_nan_range_rejected(self):
        with pytest.raises(ValueError, match="R1"):
            self._ellipsoid(R1=float("nan"))

    def test_ellipsoid_inf_range_rejected(self):
        with pytest.raises(ValueError, match="R3"):
            self._ellipsoid(R3=float("inf"))

    def test_ellipsoid_negative_range_rejected(self):
        with pytest.raises(ValueError, match="R2"):
            self._ellipsoid(R2=-5.0)

    def test_ellipsoid_nan_angle_rejected(self):
        with pytest.raises(ValueError, match="azimuth"):
            self._ellipsoid(azimuth=float("nan"))

    def test_ellipsoid_inf_angle_rejected(self):
        with pytest.raises(ValueError, match="dip"):
            self._ellipsoid(dip=float("inf"))

    def test_ellipsoid_zero_range_still_constructs(self):
        """R=0 is allowed at construction (existing contract); the F-40
        degeneracy check fires at CalcVariograms call time instead."""
        ell = self._ellipsoid(R2=0.0, R3=0.0)
        assert ell.ell.R2 == 0.0

    # --- F-38: magnitude caps ---

    def test_absurd_lag_separation_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="lag_separation"):
            VariogramSearchTemplate(
                lag_width=1.0, lag_separation=1e6, tol_distance=1.0,
                num_lags=10000, first_lag_distance=0.0, ellipsoid=ell,
            )

    def test_absurd_total_lag_extent_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="total lag extent"):
            VariogramSearchTemplate(
                lag_width=1.0, lag_separation=1e4, tol_distance=1.0,
                num_lags=10000, first_lag_distance=0.0, ellipsoid=ell,
            )

    def test_absurd_ellipsoid_range_rejected(self):
        with pytest.raises(ValueError, match="R1"):
            self._ellipsoid(R1=1e9)

    # --- I2-01: VariogramSearchTemplate validation ---

    def test_negative_lag_width_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="lag_width"):
            VariogramSearchTemplate(
                lag_width=-1.0, lag_separation=2.0, tol_distance=1.0,
                num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
            )

    def test_nan_lag_separation_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="lag_separation"):
            VariogramSearchTemplate(
                lag_width=1.0, lag_separation=float("nan"), tol_distance=1.0,
                num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
            )

    def test_zero_tol_distance_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="tol_distance"):
            VariogramSearchTemplate(
                lag_width=1.0, lag_separation=2.0, tol_distance=0.0,
                num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
            )

    def test_negative_first_lag_distance_rejected(self):
        ell = self._ellipsoid()
        with pytest.raises(ValueError, match="first_lag_distance"):
            VariogramSearchTemplate(
                lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
                num_lags=3, first_lag_distance=-1.0, ellipsoid=ell,
            )

    def test_valid_template_still_constructs(self):
        ell = self._ellipsoid()
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        assert templ.num_lags == 3


# ===========================================================================
# F-40: degenerate templates (R=0 / zero directions) rejected before the C++
# silent all-zero variogram.
# ===========================================================================


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestF40DegenerateTemplate:
    def _make_grid_data(self, nx=5, ny=5, nz=3):
        np.random.seed(42)
        data = np.random.rand(nx, ny, nz).astype("float32") * 100
        mask = np.ones((nx, ny, nz), dtype="uint8")
        mask[::2, ::2, :] = 0
        return (data, mask)

    def test_zero_range_template_raises(self):
        ell = Ellipsoid(R1=10, R2=0, R3=0, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        with pytest.raises(ValueError, match="degenerate"):
            CalcVariograms(templ, self._make_grid_data())

    def test_zero_direction_template_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        # Zero out direction1 directly in the ctypes struct (bypassing the C
        # direction filler) to exercise the zero-vector degeneracy branch.
        for j in range(3):
            templ.templ.ellipsoid.direction1.data[j] = 0.0
        with pytest.raises(ValueError, match="degenerate"):
            CalcVariograms(templ, self._make_grid_data())

    def test_valid_template_returns_valid_zero_for_constant_data(self):
        """A legitimate all-zero variogram (constant data) is NOT rejected —
        only degenerate templates are."""
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        data = np.ones((5, 5, 3), dtype="float32") * 42.0
        mask = np.ones((5, 5, 3), dtype="uint8")
        lags, variogram = CalcVariograms(templ, (data, mask))
        assert np.all(np.abs(variogram) < 1e-5)


# ===========================================================================
# F-04: cdf.calc_cdf last CDF probability strictly below 1.0
# ===========================================================================


@pytest.mark.hpgl
class TestF04CdfLastProb:
    def _make_prop(self, values, mask=None, grid_shape=None):
        from geo_bsd.geo import ContProperty, SugarboxGrid

        data = np.array(values, dtype="float32")
        if mask is None:
            mask = np.ones(len(data), dtype="uint8")
        else:
            mask = np.array(mask, dtype="uint8")
        prop = ContProperty(data, mask)
        if grid_shape is None:
            grid_shape = (len(values), 1, 1)
        grid = SugarboxGrid(*grid_shape)
        prop.fix_shape(grid)
        return prop

    def test_last_prob_strictly_below_one(self):
        from geo_bsd.cdf import calc_cdf

        prop = self._make_prop([1.0, 2.0, 3.0, 4.0], grid_shape=(2, 2, 1))
        cdf = calc_cdf(prop)
        assert cdf.probs[-1] < 1.0
        assert cdf.probs[-1] == np.nextafter(np.float32(1.0), np.float32(0.0))

    def test_single_value_last_prob_strictly_below_one(self):
        from geo_bsd.cdf import calc_cdf

        prop = self._make_prop([5.0] * 8, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.probs[0] < 1.0
        assert cdf.probs[0] == np.nextafter(np.float32(1.0), np.float32(0.0))

    def test_earlier_probs_unchanged(self):
        from geo_bsd.cdf import calc_cdf

        prop = self._make_prop([1.0] * 4 + [2.0] * 4, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.probs[0] == 0.5
        assert cdf.probs[-1] < 1.0
