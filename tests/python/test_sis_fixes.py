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
#   1. C++ stderr guard (sequential_indicator_simulation.cpp:244-251) — the
#      normal-flow primary. A run where EVERY node is skipped (fully-informed
#      property) previously returned the unchanged input clone with no signal
#      at all; now the kernel writes "HPGL: SIS produced no output ..." to
#      stderr when kriging_skipped >= property.size().
#   2. Python logger warning (_warn_all_skipped, sis.py:66-96) — reachable via
#      the progress-handler-cancellation path (E-04, adversarially verified):
#      the C++ loop breaks on reporter.cancelled() before any uninformed cell
#      is processed, leaving all three outcome counters at zero while
#      uninformed > 0 — the same all-skipped signature the C++ guard surfaces
#      in normal flow. The Python analog therefore carries the cancelled-run
#      signal and must not be removed as "dead code".
# =============================================================================


@pytest.mark.hpgl
@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd.geo) not available")
class TestSISNoOutputGuard:
    """A-N1: both halves of the P-02 SIS no-output guard emit a warning."""

    def test_fully_informed_run_emits_cpp_stderr_guard(self, capfd):
        """C++ stderr guard: a fully-informed SIS run (every node skipped →
        kriging_skipped == property.size()) must emit "SIS produced no output".

        Pre-P-02: the run returned the unchanged input clone with no signal
        at all (the only stderr path required kriging_failures > 0). This pin
        is the normal-flow primary of the P-02 guard; a regression removing
        the C++ block (sequential_indicator_simulation.cpp:244-251) passes
        the suite silently without it.
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
        assert "SIS produced no output" in captured.err, (
            "P-02 C++ stderr guard did not fire on a fully-informed SIS run; "
            f"stderr={captured.err!r}"
        )
        # The output is the unchanged input clone (nothing was simulated).
        assert result.indicator_count == 2
        np.testing.assert_array_equal(result.data, prop.data)

    def test_cancelled_progress_handler_fires_python_logger_warning(self, caplog):
        """Python logger half (_warn_all_skipped): a cancelling progress
        handler leaves every outcome counter at zero while uninformed > 0 →
        the "no nodes were simulated" warning fires (E-04 reachable path).

        E-04 (adversarially verified): the C++ SIS loop breaks on
        reporter.cancelled() (sequential_indicator_simulation.cpp:89-96)
        before any uninformed cell is processed, producing the all-zero
        signature the _warn_all_skipped analog detects. This is the
        cancellation-path counterpart of the normal-flow C++ stderr guard
        above; removing the Python analog would silently lose the cancelled-run
        signal.
        """
        import logging

        import geo_bsd.geo as geo_mod
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=2, y=2, z=2)
        size = grid.x * grid.y * grid.z  # 8
        data = np.zeros(size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[0] = 0  # 1 uninformed cell (not the seed-0 first path node, index 5)
        prop = IndProperty(data, mask, 2)
        prop.fix_shape(grid)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
        )
        sis_data = _sis_data(2, cov_model, radiuses=(1, 1, 1), max_neighbours=4)

        # Cancel on the first progress tick: the handler's non-zero return is
        # propagated by update_progress → progress_reporter_t::next_lap sets
        # m_cancelled, and the SIS loop breaks before simulating any cell.
        geo_mod.set_progress_handler(lambda stage, percent, param: 1, None)
        try:
            with caplog.at_level(logging.WARNING, logger="geo_bsd.sis"):
                result = sis_simulation(
                    prop=prop, grid=grid, data=sis_data,
                    seed=0, marginal_probs=[0.5, 0.5],
                )
        finally:
            # MUST restore — a lingering cancelling handler would abort every
            # later simulation in this process (cross-test pollution).
            geo_mod.set_progress_handler(None, None)

        assert any(
            "no nodes were simulated" in rec.message for rec in caplog.records
        ), f"P-02 Python no-output warning did not fire: {[r.message for r in caplog.records]}"
        # The zero-attempt signature: no kriging evaluation, no marginal
        # substitution, no singular failure.
        stats = geo_mod._last_kriging_stats or {}
        assert stats.get("points_calculated", -1) == 0, f"stats={stats}"
        assert stats.get("points_without_neighbours", -1) == 0, f"stats={stats}"
        assert stats.get("points_singularity", -1) == 0, f"stats={stats}"
        # Output stays finite (cancellation leaves the initial state).
        assert np.all(np.isfinite(result.data))
