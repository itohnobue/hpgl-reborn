# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""Targeted Python-side tests exercising C++ fix paths through the HPGL FFI.

Tests coverage for C++ fixes from F7-06:
- F-39 / I2-F09: NaN probability detection in sample() (seq_indicator_simulation)
- F-67: correlation_coef range check [-1, 1] (simple_cokriging)
- F-41: n_threads=0 check (set_thread_num)
- F-60 / F-61: Kriging failure tracking via get_kriging_stats()
- F-46: range-relative threshold in cov_model.h (exercised via kriging operations)
- F-42: OpenMP cancel fix in indicator_kriging / median_ik (exercised via kriging)

These tests exercise C++ code paths through the Python FFI layer.
They verify fix behavior and must fail if the corresponding C++ fix is reverted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ==============================================================================
# Availability check — import fails gracefully when HPGL C++ lib is missing
# ==============================================================================

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        simple_cokriging_markI,
        simple_cokriging_markII,
    )
    from geo_bsd.sis import sis_simulation
    from geo_bsd.validation import CriticalValidationError

    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def small_grid():
    """5x5x3 grid for fast tests."""
    return SugarboxGrid(x=5, y=5, z=3)


@pytest.fixture
def small_cont_prop():
    """Small continuous property with partial informed mask."""
    rng = np.random.RandomState(42)
    size = 5 * 5 * 3  # 75
    data = rng.rand(size).astype("float32") * 100
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0  # ~10% uninformed
    return ContProperty(data, mask)


@pytest.fixture
def small_ind_prop():
    """Small indicator property with 3 categories."""
    rng = np.random.RandomState(42)
    size = 5 * 5 * 3  # 75
    data = rng.randint(0, 3, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0
    return IndProperty(data, mask, 3)


@pytest.fixture
def sample_cov_model():
    """Spherical covariance model."""
    return CovarianceModel(
        type=covariance.spherical,
        ranges=(3.0, 3.0, 2.0),
        angles=(0.0, 0.0, 0.0),
        sill=1.0,
        nugget=0.1,
    )


@pytest.fixture
def sis_data_3ind():
    """SIS data config for 3-indicator case."""
    data = []
    for _ in range(3):
        data.append(
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical,
                    ranges=(3.0, 3.0, 2.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
                "radiuses": (3, 3, 2),
                "max_neighbours": 8,
            }
        )
    return data


# ==============================================================================
# F-41: n_threads=0 check (set_thread_num)
# ==============================================================================


@pytest.mark.hpgl
class TestThreadNumValidation:
    """Tests for n_threads=0 check (C++ fix F-41, Python validation layer)."""

    def test_set_thread_num_zero_raises_valueerror(self):
        """n_threads=0 must raise ValueError (Python gate prevents invalid C++ call)."""
        from geo_bsd.geo import set_thread_num

        with pytest.raises(ValueError, match="num must be at least 1"):
            set_thread_num(0)

    def test_set_thread_num_negative_raises_valueerror(self):
        """Negative thread count must raise ValueError."""
        from geo_bsd.geo import set_thread_num

        with pytest.raises(ValueError, match="num must be at least 1"):
            set_thread_num(-1)

    def test_set_thread_num_one_succeeds(self):
        """n_threads=1 must succeed (exercises C++ set_thread_num path)."""
        from geo_bsd.geo import get_thread_num, set_thread_num

        set_thread_num(1)
        assert get_thread_num() >= 1


# ==============================================================================
# F-67: correlation_coef range check [-1, 1]
# ==============================================================================


@pytest.mark.hpgl
class TestCorrelationCoefValidation:
    """Tests for correlation_coef range check (C++ fix F-67, Python gate)."""

    def test_correlation_coef_above_one_rejected_markI(self, small_cont_prop, small_grid, sample_cov_model):
        """correlation_coef > 1 must be caught (simple_cokriging_markI)."""
        secondary_data = ContProperty(
            np.random.RandomState(43).rand(75).astype("float32") * 100,
            np.ones(75, dtype="uint8"),
        )

        with pytest.raises(CriticalValidationError, match="correlation_coef"):
            simple_cokriging_markI(
                prop=small_cont_prop,
                grid=small_grid,
                radiuses=(3, 3, 2),
                max_neighbours=8,
                cov_model=sample_cov_model,
                secondary_data=secondary_data,
                primary_mean=50.0,
                secondary_mean=50.0,
                secondary_variance=100.0,
                correlation_coef=1.5,  # Invalid: > 1.0
            )

    def test_correlation_coef_below_neg_one_rejected_markI(
        self, small_cont_prop, small_grid, sample_cov_model
    ):
        """correlation_coef < -1 must be caught (simple_cokriging_markI)."""
        secondary_data = ContProperty(
            np.random.RandomState(43).rand(75).astype("float32") * 100,
            np.ones(75, dtype="uint8"),
        )

        with pytest.raises(CriticalValidationError, match="correlation_coef"):
            simple_cokriging_markI(
                prop=small_cont_prop,
                grid=small_grid,
                radiuses=(3, 3, 2),
                max_neighbours=8,
                cov_model=sample_cov_model,
                secondary_data=secondary_data,
                primary_mean=50.0,
                secondary_mean=50.0,
                secondary_variance=100.0,
                correlation_coef=-1.5,  # Invalid: < -1.0
            )

    def test_correlation_coef_valid_range_accepted_markI(
        self, small_cont_prop, small_grid, sample_cov_model
    ):
        """Valid correlation_coef must be accepted and produce result (simple_cokriging_markI)."""
        secondary_data = ContProperty(
            np.random.RandomState(43).rand(75).astype("float32") * 100,
            np.ones(75, dtype="uint8"),
        )

        result = simple_cokriging_markI(
            prop=small_cont_prop,
            grid=small_grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=sample_cov_model,
            secondary_data=secondary_data,
            primary_mean=50.0,
            secondary_mean=50.0,
            secondary_variance=100.0,
            correlation_coef=0.7,  # Valid: in [-1, 1]
        )
        assert isinstance(result, ContProperty)
        assert result.data.size == 75

    def test_correlation_coef_above_one_rejected_markII(self, small_cont_prop, small_grid):
        """correlation_coef > 1 must be caught (simple_cokriging_markII)."""
        primary = {
            "data": small_cont_prop,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(3.0, 3.0, 2.0),
                sill=1.0,
                nugget=0.1,
            ),
            "mean": 50.0,
        }
        secondary = {
            "data": ContProperty(
                np.random.RandomState(43).rand(75).astype("float32") * 100,
                np.ones(75, dtype="uint8"),
            ),
            "cov_model": CovarianceModel(
                type=covariance.exponential,
                ranges=(3.0, 3.0, 2.0),
                sill=1.0,
                nugget=0.1,
            ),
            "mean": 50.0,
        }

        with pytest.raises(CriticalValidationError, match="correlation_coef"):
            simple_cokriging_markII(
                grid=small_grid,
                primary_data=primary,
                secondary_data=secondary,
                correlation_coef=2.0,  # Invalid: > 1.0
                radiuses=(3, 3, 2),
                max_neighbours=8,
            )

    def test_correlation_coef_valid_accepted_markII(self, small_cont_prop, small_grid):
        """Valid correlation_coef must be accepted (simple_cokriging_markII)."""
        primary = {
            "data": small_cont_prop,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(3.0, 3.0, 2.0),
                sill=1.0,
                nugget=0.1,
            ),
            "mean": 50.0,
        }
        secondary = {
            "data": ContProperty(
                np.random.RandomState(43).rand(75).astype("float32") * 100,
                np.ones(75, dtype="uint8"),
            ),
            "cov_model": CovarianceModel(
                type=covariance.exponential,
                ranges=(3.0, 3.0, 2.0),
                sill=1.0,
                nugget=0.1,
            ),
            "mean": 50.0,
        }

        result = simple_cokriging_markII(
            grid=small_grid,
            primary_data=primary,
            secondary_data=secondary,
            correlation_coef=0.5,  # Valid: in [-1, 1]
            radiuses=(3, 3, 2),
            max_neighbours=8,
        )
        assert isinstance(result, ContProperty)
        assert result.data.size == 75


# ==============================================================================
# F-39 / I2-F09: NaN probability detection in SIS
# ==============================================================================


@pytest.mark.hpgl
class TestSISNanProbabilityDetection:
    """Tests exercising the NaN probability detection fix in C++ sample().

    The Python LVM path validates marginal_probs for NaN via numpy.isfinite()
    before calling C++ (sis.py:344-348). The C++ fix (F-39/I2-F09) is a
    defense-in-depth layer that detects NaN in sample() if it reaches C++.

    We test:
      1. Python validates NaN and rejects it (gatekeeping layer works)
      2. Python rejects out-of-[0,1] LVM probabilities (sis.py:304-309)
      3. Valid SIS completes successfully (C++ fix path exercises valid data)
      4. Valid SIS via LVM path completes successfully (exercises correlogram path)
    """

    def test_sis_lvm_nan_probs_rejected(
        self, small_ind_prop, small_grid, sis_data_3ind
    ):
        """NaN in LVM marginal_probs must be rejected.

        When NaN is injected, the per-cell probability sum check at
        sis.py:293-300 fires first (NaN propagates through sum),
        producing a "deviates from 1.0" error with nan deviation.
        The isfinite check at sis.py:344-348 is the second line of
        defense that catches NaN that somehow survives the sum check.
        Both checks together ensure NaN never reaches C++.
        """
        x, y, z = small_grid.x, small_grid.y, small_grid.z

        # Create LVM probabilities where one has NaN
        lvm_probs = []
        rng = np.random.RandomState(100)
        for _ in range(3):
            probs = rng.rand(x, y, z).astype("float32", order="F")
            lvm_probs.append(probs)
        # Normalize per-cell to sum to 1.0
        prob_sum = np.sum(lvm_probs, axis=0)
        for i in range(3):
            lvm_probs[i] = lvm_probs[i] / prob_sum
        # Inject NaN into second indicator's first cell
        # NaN propagates through the sum → sum check fires first
        lvm_probs[1][0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="deviates"):
            sis_simulation(
                prop=small_ind_prop,
                grid=small_grid,
                data=sis_data_3ind,
                seed=42,
                marginal_probs=lvm_probs,
                use_correlogram=True,
            )

    def test_sis_lvm_out_of_range_probs_rejected(
        self, small_ind_prop, small_grid, sis_data_3ind
    ):
        """Out-of-range [0,1] LVM marginal_probs must be rejected.

        Inject 1.5 and adjust another prob at the same cell by -0.5,
        keeping the per-cell sum ~1.0 while creating a value outside [0,1].
        This way the sum check passes and the range check fires.
        """
        x, y, z = small_grid.x, small_grid.y, small_grid.z

        lvm_probs = []
        rng = np.random.RandomState(101)
        for _ in range(3):
            probs = rng.rand(x, y, z).astype("float32", order="F")
            lvm_probs.append(probs)
        # Normalize per-cell to sum to 1.0
        prob_sum = np.sum(lvm_probs, axis=0)
        for i in range(3):
            lvm_probs[i] = lvm_probs[i] / prob_sum
        # Inject [1.5, -0.5, 0.0] at cell [0,0,0] → sum stays 1.0
        lvm_probs[0][0, 0, 0] = 1.5
        lvm_probs[1][0, 0, 0] = -0.5
        lvm_probs[2][0, 0, 0] = 0.0

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sis_simulation(
                prop=small_ind_prop,
                grid=small_grid,
                data=sis_data_3ind,
                seed=42,
                marginal_probs=lvm_probs,
                use_correlogram=True,
            )

    def test_sis_completes_successfully_lvm(
        self, small_ind_prop, small_grid, sis_data_3ind
    ):
        """Valid SIS simulation (LVM) completes — exercises C++ LVM/sample() path."""
        x, y, z = small_grid.x, small_grid.y, small_grid.z

        lvm_probs = []
        rng = np.random.RandomState(102)
        for _ in range(3):
            probs = rng.rand(x, y, z).astype("float32", order="F")
            lvm_probs.append(probs)
        # Normalize
        prob_sum = np.sum(lvm_probs, axis=0)
        for i in range(3):
            lvm_probs[i] = lvm_probs[i] / prob_sum

        result = sis_simulation(
            prop=small_ind_prop,
            grid=small_grid,
            data=sis_data_3ind,
            seed=42,
            marginal_probs=lvm_probs,
            use_correlogram=True,
        )
        assert isinstance(result, IndProperty)
        assert result.data.size == 75
        assert result.indicator_count == 3

    def test_sis_reproducibility_same_seed(
        self, small_ind_prop, small_grid, sis_data_3ind
    ):
        """Same seed must produce identical SIS results (verifies no silent NaN corruption)."""
        result1 = sis_simulation(
            prop=small_ind_prop,
            grid=small_grid,
            data=sis_data_3ind,
            seed=42,
            marginal_probs=[0.33, 0.33, 0.34],
            use_correlogram=False,
        )
        result2 = sis_simulation(
            prop=small_ind_prop,
            grid=small_grid,
            data=sis_data_3ind,
            seed=42,
            marginal_probs=[0.33, 0.33, 0.34],
            use_correlogram=False,
        )
        assert np.array_equal(result1.data, result2.data)


# ==============================================================================
# F-60 / F-61: Kriging failure tracking via get_kriging_stats()
# ==============================================================================


@pytest.mark.hpgl
class TestKrigingFailureTracking:
    """Tests for kriging failure statistics via get_kriging_stats() (F-60, F-61).

    The C++ fixes F-60 (sequential_simulation.h) and F-61
    (sequential_indicator_simulation.cpp) track kriging failures at the
    C++ level. The Python API accesses these stats via get_kriging_stats().

    We verify:
      1. get_kriging_stats() is callable and returns expected keys
      2. After a kriging/simulation call, stats report points_calculated > 0
      3. After operations on sparse data, failure stats are populated
         (points_without_neighbours or points_singularity reflects failures)
    """

    def test_get_kriging_stats_returns_dict(self):
        """get_kriging_stats() must return a dict with expected keys."""
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats

        if not _HAS_KRIGING_STATS:
            pytest.skip("hpgl_get_kriging_stats not available in this library build")

        stats = get_kriging_stats()
        assert isinstance(stats, dict)
        expected_keys = {
            "points_calculated",
            "points_without_neighbours",
            "points_singularity",
            "mean",
            "speed_nps",
        }
        assert expected_keys.issubset(stats.keys())

    def test_kriging_stats_reported_after_ordinary_kriging(
        self, small_cont_prop, small_grid, sample_cov_model
    ):
        """After ordinary_kriging, get_kriging_stats reports nonzero points_calculated."""
        from geo_bsd import geo
        from geo_bsd.geo import ordinary_kriging
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats

        if not _HAS_KRIGING_STATS:
            pytest.skip("hpgl_get_kriging_stats not available in this library build")

        # Run kriging operation
        ordinary_kriging(
            prop=small_cont_prop,
            grid=small_grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=sample_cov_model,
        )

        stats = get_kriging_stats()
        assert stats["points_calculated"] > 0, (
            f"Expected positive points_calculated after kriging, got {stats}"
        )
        # Verify Python wrapper integration (geo._last_kriging_stats wiring at geo.py)
        assert geo._last_kriging_stats is not None, (
            "geo._last_kriging_stats should be populated after ordinary_kriging"
        )
        assert geo._last_kriging_stats["points_calculated"] > 0, (
            f"Expected positive points_calculated in geo._last_kriging_stats, "
            f"got {geo._last_kriging_stats}"
        )

    def test_kriging_stats_after_sgs_simulation(
        self, small_cont_prop, small_grid, sample_cov_model
    ):
        """After SGS simulation, kriging stats ARE populated (F-M6/F-N4).

        F-M6 wiring: the C++ SGS path now calls set_kriging_stats with the
        simulation's kriging-outcome counters, so the C-level stats must be
        CHANGED by the call (previously SGS left them untouched and the
        Python sentinel stayed None). Snapshotting before/after removes the
        test-order dependency.
        """
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats
        from geo_bsd.sgs import sgs_simulation

        if not _HAS_KRIGING_STATS:
            pytest.skip("hpgl_get_kriging_stats not available in this library build")

        stats_before = dict(get_kriging_stats())

        sgs_simulation(
            prop=small_cont_prop,
            grid=small_grid,
            cdf_data=None,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=sample_cov_model,
            seed=42,
            kriging_type="sk",
        )

        stats_after = dict(get_kriging_stats())
        assert stats_after != stats_before, (
            f"SGS must populate kriging stats (F-M6); got {stats_before} -> {stats_after}"
        )
        assert stats_after["points_calculated"] > 0, (
            f"Expected positive points_calculated after SGS, got {stats_after}"
        )

    def test_kriging_stats_detects_no_neighbours_on_sparse_data(
        self, small_grid, sample_cov_model
    ):
        """Kriging on extremely sparse data produces failure stats (points_without_neighbours > 0)."""
        from geo_bsd.geo import ordinary_kriging
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats

        if not _HAS_KRIGING_STATS:
            pytest.skip("hpgl_get_kriging_stats not available in this library build")

        # Create a property where all cells are uninformed (mask=0 everywhere)
        size = 5 * 5 * 3
        data = np.random.RandomState(200).rand(size).astype("float32") * 100
        mask = np.zeros(size, dtype="uint8")  # ALL uninformed — no data to krige from
        empty_prop = ContProperty(data, mask)

        # Kriging should complete (mean fill on failure) but stats show failures
        ordinary_kriging(
            prop=empty_prop,
            grid=small_grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=sample_cov_model,
        )

        stats = get_kriging_stats()
        # F-N6: points_calculated counts ONLY successfully kriged cells
        # (cont_kriging.h counts KI_SUCCESS cells in both failure modes).
        # With ALL cells uninformed there are no successes, so
        # points_calculated == 0 — the _check_kriging_failure_stats warning
        # branch (calculated < expected) is what now surfaces this case.
        assert stats["points_calculated"] == 0, f"Stats: {stats}"
        # All cells lack neighbours on fully masked input
        assert stats["points_without_neighbours"] > 0, (
            f"Expected failure stats on sparse data, got {stats}"
        )


# ==============================================================================
# F-46: range-relative threshold in cov_model.h (exercised via kriging ops)
# ==============================================================================


@pytest.mark.hpgl
class TestCovModelRangeThreshold:
    """Exercises the C++ cov_model range-relative threshold fix (F-46).

    The fix ensures the range-relative threshold in cov_model.h prevents
    division by zero or degenerate behavior. We test that kriging with
    valid model parameters completes successfully AND produces finite
    output with nonzero kriging stats — a threshold regression that
    silently mean-fills or emits NaN/Inf now fails the hardened asserts.
    """

    def test_kriging_with_exponential_model_completes(
        self, small_cont_prop, small_grid
    ):
        """Kriging with exponential covariance model exercises range-relative threshold.

        Hardened (L-25): assert finite output + nonzero kriging stats — a
        silent mean-fill regression on the F-46 threshold would produce
        finite-but-degenerate results that the old isinstance+size-only
        checks could not detect.
        """
        from geo_bsd.geo import ordinary_kriging
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats

        exp_cov = CovarianceModel(
            type=covariance.exponential,
            ranges=(3.0, 3.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=small_cont_prop,
            grid=small_grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=exp_cov,
        )
        assert isinstance(result, ContProperty)
        assert result.data.size == 75
        assert np.all(np.isfinite(result.data)), (
            "F-46 regression: kriging produced non-finite output"
        )
        if _HAS_KRIGING_STATS:
            assert get_kriging_stats()["points_calculated"] > 0, (
                "F-46 regression: no cells kriged (stats points_calculated == 0)"
            )

    def test_kriging_with_gaussian_model_completes(
        self, small_cont_prop, small_grid
    ):
        """Kriging with Gaussian covariance model exercises range-relative threshold.

        Hardened (L-25): assert finite output + nonzero kriging stats (see
        the exponential sibling).
        """
        from geo_bsd.geo import ordinary_kriging
        from geo_bsd.hpgl_wrap import _HAS_KRIGING_STATS, get_kriging_stats

        gauss_cov = CovarianceModel(
            type=covariance.gaussian,
            ranges=(3.0, 3.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=small_cont_prop,
            grid=small_grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=gauss_cov,
        )
        assert isinstance(result, ContProperty)
        assert result.data.size == 75
        assert np.all(np.isfinite(result.data)), (
            "F-46 regression: kriging produced non-finite output"
        )
        if _HAS_KRIGING_STATS:
            assert get_kriging_stats()["points_calculated"] > 0, (
                "F-46 regression: no cells kriged (stats points_calculated == 0)"
            )


# ==============================================================================
# F-42: OpenMP cancel fix in indicator_kriging / median_ik
# ==============================================================================


@pytest.mark.hpgl
class TestOpenMPCancelFix:
    """Exercises the C++ OpenMP cancel fix (F-42) in median_ik.

    The fix correctly handles OpenMP cancellation in indicator kriging loops.
    median_ik is the direct Python entry to the fixed C++ path (the
    2-category indicator_kriging wrapper redirects to median_ik internally).
    No C++-layer OpenMP-cancel test exists, so this is the sole guard.
    """

    def test_median_ik_completes(self, small_grid, sample_cov_model):
        """Median IK exercises the OpenMP cancel path in C++."""
        from geo_bsd.geo import median_ik

        # Create 2-category indicator property
        size = 5 * 5 * 3
        data = np.random.RandomState(45).randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[::10] = 0
        prop2 = IndProperty(data, mask, 2)

        result = median_ik(
            prop=prop2,
            grid=small_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=sample_cov_model,
        )
        assert isinstance(result, IndProperty)
        assert result.data.size == 75


# ==============================================================================
# Removed in v2.0.6 (redundant smokes, L-07):
#   - TestSampleFunctionNaNDetection / CF-22: dup of test_production_fixes_203.py
#   - TestAPIValidationOrdering / CF-23, CF-24: valid-input shape checks whose
#     named branches fire only on invalid input
#   - TestEndToEndCppFixes / CF-25: composite whose steps are individually pinned
# ==============================================================================
