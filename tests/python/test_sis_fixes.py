"""Regression tests for SIS/SGS failure-statistics counting (F1/F11, stage-9 fix pass).

Covers:
- F1: 2-category SIS expected-count formula. The C++ median-SIS branch
  (sequential_indicator_simulation.cpp:122-155) performs ONE kriging
  evaluation per node for 2 categories (vs one evaluation per category for
  3+ categories, :156-179). The Python wrapper's expected count must mirror
  that per-branch counting, otherwise the failure check
  (geo._check_kriging_failure_stats) fires a spurious "could not be kriged"
  warning on every fully-successful 2-category run.
- F11: SGS with min_neighbours > 0. The C++ GSLIB ndmin semantics
  (sequential_simulation.h:104-114) leave nodes with fewer than
  min_neighbours conditioning data unsimulated and exclude them from ALL
  stats counters, so the wrapper passes expected=0 in this configuration to
  avoid a spurious warning; min_neighbours=0 keeps the documented warning.

Each SIS test runs on sparse data whose informed lattice guarantees every
uninformed cell finds a neighbour inside the search radius — so a warning
can only be caused by the expected-count mismatch, never by genuine
no-neighbour failures.
"""

import sys
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
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


def _sis_data(indicator_count, cov_model, radiuses, max_neighbours):
    """Build the per-indicator SIS data list (same shape as tests elsewhere)."""
    return [
        {"cov_model": cov_model, "radiuses": radiuses, "max_neighbours": max_neighbours}
        for _ in range(indicator_count)
    ]


def _sparse_lattice_mask(grid, spacing):
    """Fortran-order mask with informed cells on a lattice of ``spacing``.

    With ``spacing <= radius + 1`` in every axis, every uninformed cell is
    guaranteed to have at least one informed neighbour inside the search
    radius, so SIS has no genuine no-neighbour failures to report.
    """
    mask3 = np.zeros((grid.x, grid.y, grid.z), dtype="uint8", order="F")
    mask3[::spacing, ::spacing, :] = 1
    return mask3.ravel(order="F")


@pytest.mark.hpgl
@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd.geo) not available")
class TestSIS2CategoryFailureStats:
    """F1 regression: no spurious warning + per-node stats for 2-category SIS."""

    def test_sis_2category_no_spurious_warning(self, caplog):
        """F1: a fully-successful 2-category SIS must not warn about failures.

        Pre-fix: expected = uninformed * len(data) = 2 * uninformed, but the
        C++ 2-category branch counts ONE evaluation per node, so
        calculated < expected and "X of 2X cells could not be kriged" fired
        on every successful run. Post-fix: expected mirrors the 1-eval-per-node
        counting and no warning fires.
        """
        import logging

        import geo_bsd.geo as geo_mod
        from geo_bsd.sis import sis_simulation

        indicator_count = 2
        grid = SugarboxGrid(x=8, y=8, z=2)
        size = grid.x * grid.y * grid.z  # 128
        rng = np.random.RandomState(7)
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = _sparse_lattice_mask(grid, spacing=3)  # 18 informed → 110 uninformed
        prop = IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        # Radius (3,3,2) covers the lattice gap (max distance 1 in x/y, 1 in z).
        sis_data = _sis_data(indicator_count, cov_model, radiuses=(3, 3, 2), max_neighbours=12)

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = sis_simulation(
                prop=prop, grid=grid, data=sis_data,
                seed=42, marginal_probs=[0.5, 0.5],
            )

        # Core F1 regression: no spurious failure warning on full success.
        assert not any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"spurious 2-category SIS warning: {[r.message for r in caplog.records]}"

        # Stats populated correctly: 2-category branch counts ONE kriging
        # evaluation per node, so the three outcome counters must sum to the
        # number of uninformed cells (each node counted exactly once).
        assert geo_mod._last_kriging_stats is not None, (
            "geo._last_kriging_stats should be populated after sis_simulation"
        )
        stats = geo_mod._last_kriging_stats
        uninformed = size - int(np.sum(mask > 0))
        assert stats["points_calculated"] == uninformed, (
            f"expected all {uninformed} uninformed cells kriged in 2-cat SIS, "
            f"got stats={stats}"
        )
        assert (
            stats["points_calculated"]
            + stats["points_without_neighbours"]
            + stats["points_singularity"]
        ) == uninformed, f"2-category counters must sum to uninformed cells, got stats={stats}"

        # Output sanity: simulated values stay inside the 2-category range.
        assert result.indicator_count == 2
        assert np.all(result.data < 2)
        assert np.all(result.data >= 0)

    def test_sis_3category_no_spurious_warning_control(self, caplog):
        """F1 control: the 3+-category SIS path must keep counting correctly.

        The 3+-category branch (sequential_indicator_simulation.cpp:156-179)
        performs one evaluation PER CATEGORY, so the counters must sum to
        uninformed * indicator_count and no warning may fire on full success.
        """
        import logging

        import geo_bsd.geo as geo_mod
        from geo_bsd.sis import sis_simulation

        indicator_count = 3
        grid = SugarboxGrid(x=8, y=8, z=2)
        size = grid.x * grid.y * grid.z  # 128
        rng = np.random.RandomState(7)
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = _sparse_lattice_mask(grid, spacing=3)  # 18 informed → 110 uninformed
        prop = IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        sis_data = _sis_data(indicator_count, cov_model, radiuses=(3, 3, 2), max_neighbours=12)

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = sis_simulation(
                prop=prop, grid=grid, data=sis_data,
                seed=42, marginal_probs=[0.3, 0.4, 0.3],
            )

        assert not any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"spurious 3-category SIS warning: {[r.message for r in caplog.records]}"

        assert geo_mod._last_kriging_stats is not None
        stats = geo_mod._last_kriging_stats
        uninformed = size - int(np.sum(mask > 0))
        assert (
            stats["points_calculated"]
            + stats["points_without_neighbours"]
            + stats["points_singularity"]
        ) == uninformed * indicator_count, (
            f"3-category counters must sum to uninformed × indicator_count, "
            f"got stats={stats}"
        )
        assert stats["points_calculated"] == uninformed * indicator_count, (
            f"expected all {uninformed * indicator_count} category evals kriged, "
            f"got stats={stats}"
        )

        assert result.indicator_count == 3
        assert np.all(result.data < 3)
        assert np.all(result.data >= 0)


@pytest.mark.hpgl
@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd.geo) not available")
class TestSGSMinNeighboursFailureStats:
    """F11 regression: SGS with min_neighbours > 0 must not spuriously warn.

    The C++ GSLIB ndmin semantics (sequential_simulation.h:104-114) leave
    nodes with fewer than min_neighbours conditioning data unsimulated and
    exclude them from ALL stats counters; the skip count is stderr-only, so
    Python cannot compute a matching expected count a priori. The wrapper
    passes expected=0 in this configuration — no spurious "could not be
    kriged" warning — while genuine singularity failures still raise via
    geo._finalize_kriging_stats.
    """

    @staticmethod
    def _grid_and_cov():
        grid = SugarboxGrid(x=8, y=8, z=2)
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        return grid, cov_model

    def test_sgs_min_neighbours_sparse_no_spurious_warning(self, caplog):
        """F11: sparse data + min_neighbours=2 → ndmin skips must not warn.

        Pre-fix: expected = uninformed cells, but ndmin-skipped nodes are
        excluded from all C++ counters, so calculated < expected fired a
        spurious warning. Post-fix: expected=0 in this config → no warning.
        """
        import logging

        from geo_bsd.sgs import sgs_simulation

        grid, cov_model = self._grid_and_cov()
        size = grid.x * grid.y * grid.z  # 128
        rng = np.random.RandomState(7)
        data = rng.rand(size).astype("float32") * 100
        # Very sparse mask (8 informed): most cells have <2 conditioning data,
        # so the C++ ndmin-skip path (sequential_simulation.h:109-114) runs.
        mask = _sparse_lattice_mask(grid, spacing=4)
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(4, 4, 2), max_neighbours=12,
                cov_model=cov_model, seed=42, kriging_type="sk",
                min_neighbours=2,
            )

        assert not any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"spurious SGS ndmin warning: {[r.message for r in caplog.records]}"
        assert np.all(np.isfinite(result.data))

    def test_sgs_min_neighbours_zero_still_warns_on_genuine_failure(self, caplog):
        """F11 guard: min_neighbours=0 (default) keeps warning on genuine
        no-neighbour cells — suppression is limited to ndmin configs."""
        import logging

        from geo_bsd.sgs import sgs_simulation

        grid, cov_model = self._grid_and_cov()
        size = grid.x * grid.y * grid.z  # 128
        rng = np.random.RandomState(7)
        data = rng.rand(size).astype("float32") * 100
        single_mask = np.zeros(size, dtype="uint8")
        single_mask[0] = 1  # one informed cell → most cells have no neighbours
        prop = ContProperty(data, single_mask)
        prop.fix_shape(grid)

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(1, 1, 1), max_neighbours=4,
                cov_model=cov_model, seed=42, kriging_type="sk",
                min_neighbours=0,
            )

        assert any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), "min_neighbours=0 no-neighbour warning must still fire"


# =============================================================================
# A-N1 — P-02 no-output guard pin (post-P-02 regression test)
#
# Mirrors the E-M57 SGS analog (test_production_fixes_201.py:250-299). SIS has
# no ndmin gate, so the P-02 guard has two independently-pinnable halves:
#   1. C++ no-output guard (sequential_indicator_simulation.cpp, the
#      kriging_skipped >= property.size() block) — the normal-flow primary. A
#      run where EVERY node is skipped (fully-informed property) previously
#      returned the unchanged input clone with no signal at all; now the kernel
#      writes "HPGL: SIS produced no output ..." through hpgl::write() (G-23:
#      the documented output channel — handler, std::cout fallback — not raw
#      stderr).
#   2. Python logger warning (_warn_all_skipped, sis.py:66-96) — reachable only
#      on a NON-cancelled run whose outcome counters all stay zero while
#      uninformed > 0. F-379 makes the cancelled run throw instead of reaching
#      it, so the cancellation path is pinned by the raise itself (second test
#      below) and this Python analog carries the non-cancelled all-skipped
#      signal; it must not be removed as "dead code".
# =============================================================================


@pytest.mark.hpgl
@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd.geo) not available")
class TestSISNoOutputGuard:
    """A-N1: the P-02 SIS no-output signal — C++ message plus the F-379 raise."""

    def test_fully_informed_run_emits_cpp_stderr_guard(self, capfd):
        """C++ no-output guard: a fully-informed SIS run (every node skipped →
        kriging_skipped == property.size()) must emit "SIS produced no output".

        G-23: the message goes to the documented output channel — the C++ site
        calls ``hpgl::write()`` (sequential_indicator_simulation.cpp, the
        no-output block), which routes to the registered output handler with a
        ``std::cout`` fallback, NOT to stderr. The assertion therefore reads
        ``captured.out`` (capfd captures fd 1, ``std::cout`` included); the
        pre-G-23 stderr pin could not pass on the fixed tree. The test name
        predates that channel move and is kept for traceability.

        Pre-P-02: the run returned the unchanged input clone with no signal
        at all (the only message path required kriging_failures > 0). This pin
        is the normal-flow primary of the P-02 guard; a regression removing
        the C++ block passes the suite silently without it.
        """
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=4, y=4, z=2)
        size = grid.x * grid.y * grid.z  # 32
        rng = np.random.RandomState(7)
        data = rng.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")  # FULLY informed → all nodes skipped
        prop = IndProperty(data, mask, 2)
        prop.fix_shape(grid)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        sis_data = _sis_data(2, cov_model, radiuses=(3, 3, 2), max_neighbours=12)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.5, 0.5],
        )

        captured = capfd.readouterr()
        assert "SIS produced no output" in captured.out, (
            "P-02 C++ no-output guard did not fire on a fully-informed SIS run; "
            f"stdout={captured.out!r}"
        )
        # The output is the unchanged input clone (nothing was simulated).
        assert result.indicator_count == 2
        np.testing.assert_array_equal(result.data, prop.data)

    def test_cancelled_progress_handler_fires_python_logger_warning(self):
        """F-379: a cancelling progress handler makes SIS RAISE (contract pin).

        Re-pinned from the pre-F-379 contract. Until F-379 the cancelled run
        fell through to the success path: the C++ loop broke on
        ``reporter.cancelled()`` and the wrapper then reached
        ``_warn_all_skipped`` (the Python half of the P-02 signal), so this test
        asserted the "no nodes were simulated" logger warning plus the
        zero-attempt stats. F-379 makes cancellation a control-flow outcome:
        ``sequential_indicator_simulation`` now throws after publishing the
        stats and warnings (``throw hpgl_exception("do_sis", "cancelled by
        progress handler; the output property is only partially simulated")``),
        the C API records it, and the FFI guard turns it into a RuntimeError.
        The wrapper therefore never returns a property and never reaches
        ``_warn_all_skipped`` / ``_finalize_kriging_stats`` — the old
        result / caplog / zero-stat assertions are unreachable, not stale
        values, and the partially simulated buffer is an internal clone
        (``__prepare_sis`` → ``_clone_prop``) the caller cannot inspect.

        What this pin still guarantees, all measured: the raise carries the
        F-379 message, the cancelling handler is actually invoked (the cancel
        path ran, not an unrelated failure), and the CALLER's property is left
        untouched by an aborted run.
        """
        import geo_bsd.geo as geo_mod
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=2, y=2, z=2)
        size = grid.x * grid.y * grid.z  # 8
        data = np.zeros(size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[0] = 0  # 1 uninformed cell (not the seed-0 first path node, index 5)
        prop = IndProperty(data, mask, 2)
        prop.fix_shape(grid)
        caller_data = prop.data.copy()
        caller_mask = prop.mask.copy()

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
        )
        sis_data = _sis_data(2, cov_model, radiuses=(1, 1, 1), max_neighbours=4)

        # Cancel on the first progress tick: the handler's non-zero return is
        # propagated by update_progress → progress_reporter_t::next_lap sets
        # m_cancelled, and the SIS loop breaks before simulating the cell.
        handler_calls = []

        def cancel_handler(stage, percent, param):
            handler_calls.append(percent)
            return 1

        geo_mod.set_progress_handler(cancel_handler, None)
        try:
            with pytest.raises(
                RuntimeError, match="cancelled by progress handler"
            ):
                sis_simulation(
                    prop=prop, grid=grid, data=sis_data,
                    seed=0, marginal_probs=[0.5, 0.5],
                )
        finally:
            # MUST restore — a lingering cancelling handler would abort every
            # later simulation in this process (cross-test pollution).
            geo_mod.set_progress_handler(None, None)

        # The cancel path really ran (the poll fired at least once) — this
        # keeps the raise above from being satisfied by an unrelated failure.
        assert handler_calls, "the progress handler was never invoked"
        # Cancellation is reported, not written into the caller's property:
        # the C++ kernel mutates an internal clone, so an aborted run must
        # leave the input untouched rather than half-simulated.
        np.testing.assert_array_equal(prop.data, caller_data)
        np.testing.assert_array_equal(prop.mask, caller_mask)

