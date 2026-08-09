"""Regression tests for geo.py group 1 FIX findings (F-11..F-16, F-24, F-14 Python part).

Covers:
- F-11: sgs.py/sis.py clear geo._last_kriging_stats (the documented inspection point);
        dead module-local shadow globals removed
- F-12: _last_kriging_stats reset BEFORE the FFI call in median_ik / indicator_kriging /
        simple_cokriging_markI / simple_cokriging_markII (exception-safe)
- F-13: IndProperty rejects out-of-range / fractional values before uint8 wrap
- F-15: IndProperty accepts list/tuple data (documented asarray support)
- F-16 + I2-57: clear path defers BOTH CFUNCTYPE and param; kriging holds
        _hpgl_call_lock so a concurrent clear cannot free a trampoline mid-call
- F-24: sis_simulation validates indicator_count against len(data)
- F-14 (Python part): min_neighbours reaches the C API params struct
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        indicator_kriging,
        median_ik,
        set_output_handler,
        set_progress_handler,
        simple_cokriging_markI,
        simple_cokriging_markII,
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


def _cont_prop(size=75, seed=42):
    rng = np.random.RandomState(seed)
    data = rng.rand(size).astype("float32") * 100
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0
    return ContProperty(data, mask)


def _ind_prop(count, size=75, seed=7, values_hi=None):
    rng = np.random.RandomState(seed)
    hi = count if values_hi is None else values_hi
    data = rng.randint(0, hi, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0
    return IndProperty(data, mask, count)


def _cov_model():
    return CovarianceModel(
        type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
    )


def _lock_held(lock):
    """Return True if ``lock`` is held by any thread.

    Works for BOTH ``threading.Lock`` and ``threading.RLock``: the C
    ``_thread.RLock`` exposes no ``.locked()`` method (only the Python
    ``Lock`` wrapper does), so probe from another thread with a
    non-blocking acquire instead.
    """
    probe_result = []

    def _probe():
        if lock.acquire(blocking=False):
            lock.release()
            probe_result.append(False)
        else:
            probe_result.append(True)

    t = threading.Thread(target=_probe)
    t.start()
    t.join()
    return probe_result[0]


def _sis_data(count):
    return [
        {"cov_model": _cov_model(), "radiuses": (3, 3, 2), "max_neighbours": 8}
        for _ in range(count)
    ]


# =============================================================================
# F-11: SGS/SIS clear geo._last_kriging_stats
# =============================================================================


@pytest.mark.hpgl
class TestSimulationStatsSentinel:
    """F-11: sgs/sis write the documented geo._last_kriging_stats, not a dead local."""

    def test_sgs_simulation_populates_geo_last_kriging_stats(self):
        """sgs_simulation populates geo._last_kriging_stats (F-11 + F-M6/F-N4).

        F-M6 wiring: the C++ SGS path now calls set_kriging_stats, so a
        successful simulation leaves the documented inspection point
        populated with the simulation's failure counters (previously None).

        T-27: the pre-seed is 0 (not 777) so a stale surviving value FAILS
        the > 0 assertion — deleting the entire Python stats wiring
        (reset+populate) leaves the seed in place and the test turns red
        instead of passing vacuously.
        """
        from geo_bsd import geo
        from geo_bsd.sgs import sgs_simulation

        grid = SugarboxGrid(x=5, y=5, z=3)
        prop = _cont_prop()
        geo._last_kriging_stats = {"points_calculated": 0}  # T-27: stale value must fail > 0
        try:
            sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(3, 3, 2), max_neighbours=8,
                cov_model=_cov_model(), seed=42, kriging_type="sk",
            )
            assert geo._last_kriging_stats is not None, (
                "geo._last_kriging_stats should be populated after sgs_simulation"
            )
            assert geo._last_kriging_stats["points_calculated"] > 0, (
                f"Expected positive points_calculated after SGS, got {geo._last_kriging_stats}"
            )
        finally:
            # N2-L21: the module-global sentinel must not leak into later tests.
            geo._last_kriging_stats = None

    def test_dead_module_local_shadows_removed(self):
        """The dead sgs._/sis._last_kriging_stats shadow globals are gone (F-11)."""
        import geo_bsd.sgs as sgs_mod
        import geo_bsd.sis as sis_mod

        assert not hasattr(sgs_mod, "_last_kriging_stats")
        assert not hasattr(sis_mod, "_last_kriging_stats")


# =============================================================================
# F-12: _last_kriging_stats reset BEFORE the FFI call (exception-safe)
# =============================================================================


@pytest.mark.hpgl
class TestKrigingStatsResetBeforeCall:
    """F-12: reset-before-call in the 4 wrappers that previously reset after."""

    def test_median_ik_stats_reset_on_error_path(self):
        """median_ik resets stats before the FFI call — a raised call leaves None."""
        from unittest import mock

        from geo_bsd import geo

        grid = SugarboxGrid(x=5, y=5, z=3)
        prop = _ind_prop(count=2, values_hi=2)
        geo._last_kriging_stats = {"stale": True}
        with mock.patch("geo_bsd.geo.call_median_ik", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                median_ik(
                    prop=prop, grid=grid, marginal_probs=(0.5, 0.5),
                    radiuses=(3, 3, 2), max_neighbours=8, cov_model=_cov_model(),
                )
        assert geo._last_kriging_stats is None

    def test_indicator_kriging_stats_reset_on_error_path(self):
        """indicator_kriging resets stats before the FFI call (F-12)."""
        from unittest import mock

        from geo_bsd import geo

        grid = SugarboxGrid(x=5, y=5, z=3)
        prop = _ind_prop(count=3)
        geo._last_kriging_stats = {"stale": True}
        with mock.patch(
            "geo_bsd.geo.call_indicator_kriging", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                indicator_kriging(
                    prop=prop, grid=grid, data=_sis_data(3),
                    marginal_probs=[0.33, 0.33, 0.34],
                )
        assert geo._last_kriging_stats is None

    def test_simple_cokriging_markI_stats_reset_on_error_path(self):
        """simple_cokriging_markI resets stats before the FFI call (F-12)."""
        from unittest import mock

        from geo_bsd import geo

        grid = SugarboxGrid(x=5, y=5, z=3)
        prop = _cont_prop()
        sec = _cont_prop(seed=43)
        geo._last_kriging_stats = {"stale": True}
        with mock.patch(
            "geo_bsd.geo.call_simple_cokriging_mark1", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                simple_cokriging_markI(
                    prop=prop, grid=grid, radiuses=(3, 3, 2), max_neighbours=8,
                    cov_model=_cov_model(), secondary_data=sec,
                    primary_mean=50.0, secondary_mean=50.0,
                    secondary_variance=1.0, correlation_coef=0.5,
                )
        assert geo._last_kriging_stats is None

    def test_simple_cokriging_markII_stats_reset_on_error_path(self):
        """simple_cokriging_markII resets stats before the FFI call (F-12)."""
        from unittest import mock

        from geo_bsd import geo

        grid = SugarboxGrid(x=5, y=5, z=3)
        primary_data = {"data": _cont_prop(), "cov_model": _cov_model(), "mean": 50.0}
        secondary_data = {"data": _cont_prop(seed=43), "cov_model": _cov_model(), "mean": 50.0}
        geo._last_kriging_stats = {"stale": True}
        with mock.patch(
            "geo_bsd.geo.call_simple_cokriging_mark2", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                simple_cokriging_markII(
                    grid=grid, primary_data=primary_data,
                    secondary_data=secondary_data,
                    correlation_coef=0.5, radiuses=(3, 3, 2), max_neighbours=8,
                )
        assert geo._last_kriging_stats is None


# =============================================================================
# F-13: IndProperty pre-conversion range/integer validation
# =============================================================================


@pytest.mark.hpgl
class TestIndPropertyPreConversionValidation:
    """F-13: out-of-range / fractional values rejected before uint8 wrap."""

    def test_out_of_range_high_float_rejected(self):
        """256.0 previously wrapped to 0 and passed the post-conversion check."""
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            IndProperty(np.array([256.0, 0.0, 1.0]), np.ones(3, dtype="uint8"), 2)

    def test_out_of_range_low_float_rejected(self):
        """Negative values previously wrapped to 255 (silent corruption)."""
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            IndProperty(np.array([-1.0, 0.0, 1.0]), np.ones(3, dtype="uint8"), 2)

    def test_fractional_value_rejected(self):
        """1.5 previously truncated to 1 by the uint8 conversion."""
        with pytest.raises(ValueError, match="integer"):
            IndProperty(np.array([0.5, 0.0, 1.0]), np.ones(3, dtype="uint8"), 2)

    def test_integral_float_values_accepted(self):
        """Whole-number floats remain accepted (values are indicator categories)."""
        prop = IndProperty(np.array([2.0, 1.0, 0.0]), np.ones(3, dtype="uint8"), 3)
        assert prop.data.dtype == np.uint8
        np.testing.assert_array_equal(prop.data, np.array([2, 1, 0]))


# =============================================================================
# F-15: IndProperty list/tuple data support
# =============================================================================


@pytest.mark.hpgl
class TestIndPropertyListTupleSupport:
    """F-15: list/tuple data works despite documented asarray support."""

    def test_list_data_accepted(self):
        prop = IndProperty([0, 1, 2], [1, 1, 1], 3)
        assert prop.indicator_count == 3
        assert prop.data.dtype == np.uint8
        np.testing.assert_array_equal(prop.data, np.array([0, 1, 2], dtype="uint8"))

    def test_tuple_data_accepted(self):
        prop = IndProperty((0, 1, 2), (1, 1, 1), 3)
        assert prop.indicator_count == 3
        np.testing.assert_array_equal(prop.data, np.array([0, 1, 2], dtype="uint8"))

    def test_list_shape_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="does not match mask shape"):
            IndProperty([0, 1, 2], [1, 1], 3)


# =============================================================================
# F-16 + I2-57: clear-path CFUNCTYPE deferral and concurrent-clear safety
# =============================================================================


@pytest.mark.hpgl
class TestHandlerClearDeferral:
    """F-16/I2-57a: the clear path defers BOTH the CFUNCTYPE and the param."""

    def test_clear_output_handler_defers_old_handler_and_param(self):
        from geo_bsd import geo

        def h(msg, param):
            return 1

        set_output_handler(h, "param_out")
        set_output_handler(None, None)
        last = geo._old_handler_refs[-1]
        assert last[1] == "param_out"
        assert last[0] is not None

    def test_clear_progress_handler_defers_old_handler_and_param(self):
        from geo_bsd import geo

        def p(msg, progress, param):
            return 1

        set_progress_handler(p, "param_prog")
        set_progress_handler(None, None)
        last = geo._old_handler_refs[-1]
        assert last[1] == "param_prog"
        assert last[0] is not None

    def test_set_clear_cycles_respect_shared_bound(self):
        """The shared FIFO (output + progress) stays bounded (I2-57b)."""
        from geo_bsd import geo

        def h(msg, param):
            return 1

        def p(msg, progress, param):
            return 1

        for i in range(12):
            set_output_handler(h, f"o{i}")
            set_progress_handler(p, f"p{i}")
            set_output_handler(None, None)
            set_progress_handler(None, None)
        assert len(geo._old_handler_refs) <= geo._OLD_HANDLER_REFS_CAP


@pytest.mark.hpgl
class TestKrigingCallLockSerialization:
    """F-16/I2-57c: kriging/simulation FFI calls hold _hpgl_call_lock.

    A concurrent handler clear acquires the same lock, so it cannot free a
    CFUNCTYPE trampoline mid-call. Verified by asserting the lock is held
    during the FFI call (deterministic — no timing dependence).
    """

    def test_ordinary_kriging_holds_hpgl_call_lock_during_ffi(self, monkeypatch):
        from geo_bsd import geo
        from geo_bsd.ffi_adapter import call_ordinary_kriging as real_call
        from geo_bsd.geo import ordinary_kriging

        lock_held = []

        def spy(inp, okp, outp):
            lock_held.append(_lock_held(geo._hpgl_call_lock))
            return real_call(inp, okp, outp)

        monkeypatch.setattr(geo, "call_ordinary_kriging", spy)
        grid = SugarboxGrid(x=5, y=5, z=3)
        ordinary_kriging(
            prop=_cont_prop(), grid=grid, radiuses=(3, 3, 2),
            max_neighbours=8, cov_model=_cov_model(),
        )
        assert lock_held == [True]

    def test_sgs_simulation_holds_hpgl_call_lock_during_ffi(self, monkeypatch):
        from geo_bsd import geo
        from geo_bsd import sgs as sgs_mod
        from geo_bsd.ffi_adapter import call_sgs_simulation as real_call
        from geo_bsd.sgs import sgs_simulation

        lock_held = []

        def spy(cont_marr, params, cdf, mean, mask):
            lock_held.append(_lock_held(geo._hpgl_call_lock))
            return real_call(cont_marr, params, cdf, mean, mask)

        monkeypatch.setattr(sgs_mod, "call_sgs_simulation", spy)
        grid = SugarboxGrid(x=5, y=5, z=3)
        sgs_simulation(
            prop=_cont_prop(), grid=grid, cdf_data=None,
            radiuses=(3, 3, 2), max_neighbours=8,
            cov_model=_cov_model(), seed=42, kriging_type="sk",
        )
        assert lock_held == [True]


# =============================================================================
# F-24: sis_simulation validates indicator_count
# =============================================================================


@pytest.mark.hpgl
class TestSisIndicatorCountValidation:
    """F-24: indicator_count validated against len(data) instead of overridden."""

    def test_indicator_count_mismatch_raises(self):
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=5, y=5, z=2)
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(1)
        data01 = rng.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data01, mask, 3)  # count=3, values only 0/1
        with pytest.raises(ValueError, match="indicator_count"):
            sis_simulation(
                prop=prop, grid=grid, data=_sis_data(2),
                seed=42, marginal_probs=[0.5, 0.5],
            )


# =============================================================================
# F-14 (Python part): min_neighbours forwarding to the C API
# =============================================================================


@pytest.mark.hpgl
class TestSgsMinNeighboursForwarding:
    """F-14 Python part: min_neighbours reaches the C API params struct.

    Contract-pinning test: the Python forwarding (sgs.py -> _HPGL_SGS_PARAMS ->
    call_sgs_simulation) must populate the struct field. The C++-side wiring of
    m_min_neighbours into simulation logic is a separate (cpp-A) scope.
    """

    def test_min_neighbours_reaches_c_api_params(self, monkeypatch):
        from geo_bsd import sgs
        from geo_bsd.ffi_adapter import call_sgs_simulation as real_call
        from geo_bsd.sgs import sgs_simulation

        captured = {}

        def spy(cont_marr, params, cdf, mean, mask):
            captured["params"] = params
            return real_call(cont_marr, params, cdf, mean, mask)

        monkeypatch.setattr(sgs, "call_sgs_simulation", spy)
        grid = SugarboxGrid(x=5, y=5, z=3)
        sgs_simulation(
            prop=_cont_prop(), grid=grid, cdf_data=None,
            radiuses=(3, 3, 2), max_neighbours=8,
            cov_model=_cov_model(), seed=42, kriging_type="sk",
            min_neighbours=3,
        )
        assert captured["params"].min_neighbours == 3
