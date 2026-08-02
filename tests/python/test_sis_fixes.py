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
