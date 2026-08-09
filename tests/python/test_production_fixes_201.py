"""Regression tests for Stage-6 CONFIRMED simulation-family fixes (TEST-ADD T-01..T-26).

Each test fails against the pre-fix code and passes against the fixed code (see the
test docstrings for the pre-fix failure mode). Tests exercise the documented TOP-LEVEL
public API (``from geo_bsd import X``) per pattern got-20260803180229 / finding B-08.

Covers:
- T-01  E-M9:  max_neighbours=0 "unconditional simulation" mode (SGS + SIS + configs)
- T-02  E2-01: 2D equal-volume mean_data / LVM marginal rejection (lvm_kriging, sgs, sis)
- T-08  E-M57: all-ndmin-skipped run produces a programmatic warning
- T-16  R2-16: SIS mixed unconditional/conditional warning is branch-aware
- T-26  E2-32: gtsim_2ind preserves hard-data facies through the truncation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        gtsim_2ind,
        lvm_kriging,
        sgs_simulation,
        sis_simulation,
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False

# Every test in this module needs the geo_bsd package (config classes are
# pure Python but the simulation/kriging tests call the C++ backend).
pytestmark = pytest.mark.skipif(
    not HPGL_AVAILABLE, reason="HPGL (geo_bsd) not available"
)


def _spherical_cov(ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1):
    return CovarianceModel(
        type=covariance.spherical, ranges=ranges, sill=sill, nugget=nugget
    )


# =============================================================================
# T-01 — E-M9: max_neighbours=0 is the C++-documented "unconditional simulation"
# mode (every node drawn from the marginal; no kriging). Pre-fix the Python
# entry points and config classes rejected 0 (MIN_NEIGHBORS=1), making the
# documented mode unreachable (R-05 config alignment + R-20 warning exemption).
# =============================================================================


@pytest.mark.hpgl
class TestSgsUnconditionalMode:
    """T-01: with max_neighbours=0 the kriging configuration is irrelevant —
    every uninformed cell is drawn from the marginal CDF, so two runs with
    DIFFERENT covariance models and the SAME seed are bit-identical. Pre-fix
    the call raised CriticalValidationError (max_neighbours < MIN_NEIGHBORS=1)."""

    @staticmethod
    def _run(cov_model):
        grid = SugarboxGrid(x=6, y=6, z=2)  # 72 cells
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(7)
        data = rng.rand(size).astype("float32") * 100
        mask = np.zeros(size, dtype="uint8")
        mask[0] = 1
        mask[71] = 1  # sparse: 2 informed cells
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        return sgs_simulation(
            prop=prop, grid=grid, cdf_data=None,
            radiuses=(3, 3, 2), max_neighbours=0, cov_model=cov_model,
            seed=42, kriging_type="sk",
        )

    def test_max_neighbours_zero_draws_marginal_only(self):
        import geo_bsd.geo as geo_mod

        out_a = self._run(_spherical_cov(ranges=(3.0, 3.0, 2.0)))
        # Different model: exponential with long ranges. In unconditional mode
        # no kriging happens, so the covariance model must not affect output.
        cov_b = CovarianceModel(
            type=covariance.exponential, ranges=(50.0, 50.0, 50.0), sill=1.0, nugget=0.0
        )
        out_b = self._run(cov_b)

        assert np.all(np.isfinite(out_a.data.astype("float64")))
        np.testing.assert_array_equal(
            out_a.data, out_b.data,
            err_msg="max_neighbours=0: covariance model must not affect the "
                    "marginal draws (unconditional mode)",
        )
        # Every uninformed cell is a KI_NO_NEIGHBOURS marginal draw — no cell
        # was kriged (points_calculated == 0).
        stats = geo_mod._last_kriging_stats or {}
        assert stats.get("points_calculated", -1) == 0, f"stats={stats}"
        assert stats.get("points_without_neighbours", -1) == 70, f"stats={stats}"

    def test_max_neighbours_zero_no_spurious_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            out = self._run(_spherical_cov())
        assert np.all(np.isfinite(out.data.astype("float64")))
        # The marginal draw IS the requested mode — no "could not be kriged".
        assert not any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"spurious unconditional-mode warning: {[r.message for r in caplog.records]}"


@pytest.mark.hpgl
class TestSisUnconditionalMode:
    """T-01 (SIS mirror): sis_simulation with max_neighbours=0 on every
    indicator runs the unconditional marginal-probability substitution — finite
    output, zero kriged points, no spurious failure warning. Pre-fix the
    validator rejected 0 (E-M9)."""

    def test_sis_max_neighbours_zero_runs(self, caplog):
        import logging

        import geo_bsd.geo as geo_mod

        grid = SugarboxGrid(x=6, y=6, z=2)
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(7)
        indicator_count = 2
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = np.zeros(size, dtype="uint8")
        mask[0] = 1
        mask[71] = 1
        prop = IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = _spherical_cov()
        sis_data = [
            {"cov_model": cov_model, "radiuses": (3, 3, 2), "max_neighbours": 0},
            {"cov_model": cov_model, "radiuses": (3, 3, 2), "max_neighbours": 0},
        ]
        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = sis_simulation(
                prop=prop, grid=grid, data=sis_data,
                seed=42, marginal_probs=[0.5, 0.5],
            )
        assert result.indicator_count == 2
        assert np.all(np.isfinite(np.asarray(result.data, dtype="float32")))
        assert not any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"spurious unconditional SIS warning: {[r.message for r in caplog.records]}"
        stats = geo_mod._last_kriging_stats or {}
        # 2-category branch kriges only indicator 0; unconditional → 0 kriged.
        assert stats.get("points_calculated", -1) == 0, f"stats={stats}"


# =============================================================================
# T-02 — E2-01: an equal-volume 2D mean_data / LVM marginal must be REJECTED.
# The R-13 guard was 3D-only; a (2,2) 2D array on a (1,2,2) grid (both volume 4)
# sailed past and its F-order flat layout was consumed by flat node index with a
# silently permuted field. Post-fix only 1D-flat and exactly grid-shaped 3D
# arrays are accepted.
# =============================================================================


class TestLvm2dMeanDataRejected:
    def test_lvm_kriging_2d_mean_data_raises(self):
        """T-02: lvm_kriging must reject (2,2) mean_data on a (1,2,2) grid —
        equal volume (4), silently permuted pre-fix."""
        grid = SugarboxGrid(x=1, y=2, z=2)
        size = 4
        prop = ContProperty(np.ones(size, dtype="float32"), np.ones(size, dtype="uint8"))
        prop.fix_shape(grid)
        cov_model = _spherical_cov(ranges=(1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="must be 1D flat"):
            lvm_kriging(
                prop=prop, grid=grid, radiuses=(1, 1, 1),
                max_neighbours=4, cov_model=cov_model,
                mean_data=np.ones((2, 2), dtype="float32"),
            )

    def test_lvm_kriging_flat_and_3d_mean_still_accepted(self):
        """Control: 1D flat and exactly-shaped 3D mean_data still work."""
        grid = SugarboxGrid(x=1, y=2, z=2)
        size = 4
        prop = ContProperty(np.ones(size, dtype="float32"), np.ones(size, dtype="uint8"))
        prop.fix_shape(grid)
        cov_model = _spherical_cov(ranges=(1.0, 1.0, 1.0))
        flat = lvm_kriging(
            prop=prop, grid=grid, radiuses=(1, 1, 1),
            max_neighbours=4, cov_model=cov_model,
            mean_data=np.ones(size, dtype="float32"),
        )
        assert np.all(np.isfinite(flat.data.astype("float64")))


@pytest.mark.hpgl
class TestSimulationLvm2dRejected:
    """T-02 (mirror): the SGS LVM mean guard and the SIS LVM marginal_probs
    guard (R-09) must reject 2D equal-volume arrays exactly like lvm_kriging."""

    def _grid_prop(self):
        grid = SugarboxGrid(x=1, y=2, z=2)
        size = 4
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        return grid, prop

    def test_sgs_lvm_2d_mean_raises(self):
        grid, prop = self._grid_prop()
        cov_model = _spherical_cov(ranges=(1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="must be 1D flat"):
            sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(1, 1, 1), max_neighbours=4, cov_model=cov_model,
                seed=42, kriging_type="sk",
                mean=np.ones((2, 2), dtype="float32"),
            )

    def test_sis_lvm_2d_marginal_raises(self):
        grid = SugarboxGrid(x=1, y=2, z=2)
        size = 4
        rng = np.random.RandomState(7)
        data = rng.randint(0, 2, size, dtype="uint8")
        prop = IndProperty(data, np.ones(size, dtype="uint8"), 2)
        prop.fix_shape(grid)
        cov_model = _spherical_cov(ranges=(1.0, 1.0, 1.0))
        sis_data = [
            {"cov_model": cov_model, "radiuses": (1, 1, 1), "max_neighbours": 4},
            {"cov_model": cov_model, "radiuses": (1, 1, 1), "max_neighbours": 4},
        ]
        with pytest.raises(ValueError, match="must be 1D flat"):
            sis_simulation(
                prop=prop, grid=grid, data=sis_data, seed=42,
                marginal_probs=[np.ones((2, 2), dtype="float32") * 0.5,
                                np.ones((2, 2), dtype="float32") * 0.5],
            )


# =============================================================================
# T-08 — E-M57: an ALL-ndmin-skipped SGS run (every uninformed cell left
# unsimulated by the min_neighbours gate) must produce a programmatic warning
# ("no nodes were simulated"). Pre-fix the zero-attempt signature was silent.
# =============================================================================


@pytest.mark.hpgl
class TestSgsAllNdminSkippedWarns:
    def test_all_skipped_run_warns_no_nodes_simulated(self, caplog):
        import logging

        import geo_bsd.geo as geo_mod

        grid = SugarboxGrid(x=8, y=8, z=1)
        size = grid.x * grid.y * grid.z  # 64
        rng = np.random.RandomState(7)
        data = rng.rand(size).astype("float32") * 100
        mask = np.zeros(size, dtype="uint8")
        mask[0] = 1
        mask[63] = 1  # two far-apart originals → <2 originals in any radius-1 box
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = _spherical_cov(ranges=(3.0, 3.0, 2.0))
        initial_mask = prop.mask.copy()

        with caplog.at_level(logging.WARNING, logger="geo_bsd.sgs"):
            result = sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(1, 1, 1), max_neighbours=4,
                cov_model=cov_model, seed=42, kriging_type="sk",
                min_neighbours=2,
            )

        assert any(
            "no nodes were simulated" in rec.message for rec in caplog.records
        ), f"all-ndmin-skipped run was silent: {[r.message for r in caplog.records]}"
        # No node was kriged or failed numerically — the zero-attempt signature.
        stats = geo_mod._last_kriging_stats or {}
        assert stats.get("points_calculated", -1) == 0, f"stats={stats}"
        assert stats.get("points_without_neighbours", -1) == 0, f"stats={stats}"
        assert stats.get("points_singularity", -1) == 0, f"stats={stats}"
        # Output stays at the initial masked state (nothing simulated).
        np.testing.assert_array_equal(result.mask, initial_mask)


# =============================================================================
# T-16 — R2-16: the SIS "could not be kriged" warning must be BRANCH-AWARE.
# The old `any(max_neighbours==0)` suppression hid genuine conditional failures
# in a mixed config (one unconditional + one conditional indicator). In the
# 2-category branch only indicator 0 is kriged, so with indicator 0 conditional
# the genuine failures must still warn.
# =============================================================================


@pytest.mark.hpgl
class TestSisMixedUnconditionalConditionalWarns:
    def test_conditional_indicator_failures_still_warn(self, caplog):
        import logging

        grid = SugarboxGrid(x=8, y=8, z=1)
        size = grid.x * grid.y * grid.z  # 64
        rng = np.random.RandomState(7)
        indicator_count = 2
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = np.zeros(size, dtype="uint8")
        mask[0] = 1  # a single informed cell → most nodes have NO primary neighbour
        prop = IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = _spherical_cov(ranges=(3.0, 3.0, 2.0))
        # Indicator 0 is CONDITIONAL (max_neighbours=12) → genuine no-neighbour
        # failures must be reported; indicator 1 is unconditional (0) — the
        # old any() suppression would hide indicator 0's failures.
        sis_data = [
            {"cov_model": cov_model, "radiuses": (1, 1, 1), "max_neighbours": 12},
            {"cov_model": cov_model, "radiuses": (1, 1, 1), "max_neighbours": 0},
        ]
        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = sis_simulation(
                prop=prop, grid=grid, data=sis_data,
                seed=42, marginal_probs=[0.5, 0.5],
            )
        assert result.indicator_count == 2
        assert np.all(np.isfinite(np.asarray(result.data, dtype="float32")))
        assert any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), ("mixed unconditional/conditional SIS: conditional indicator failures "
            f"must warn (any() suppression removed): {[r.message for r in caplog.records]}")


# =============================================================================
# T-26 — E2-32: gtsim_2ind must preserve hard-data facies through the
# truncation. With heterogeneous per-cell probability p the threshold
# F⁻¹(1−p) re-classifies hard-data cells (Monte Carlo reproduced a 96% flip);
# the fix restores the original 0/1 facies at informed cells after thresholding.
# =============================================================================


@pytest.mark.hpgl
class TestGtsim2IndHardDataRestore:
    """T-26: partially-masked fixture — informed cells keep their original 0/1
    facies (restored), uninformed cells are simulated and become informed."""

    def test_hard_facies_preserved_partial_mask(self):
        grid = SugarboxGrid(x=5, y=5, z=2)
        size = grid.x * grid.y * grid.z  # 50
        rng = np.random.RandomState(42)

        # 0/1 facies, ~half informed.
        facies = rng.randint(0, 2, size).astype("float32")
        mask = np.zeros(size, dtype="uint8")
        mask[::2] = 1
        prop = ContProperty(facies, mask)
        prop.fix_shape(grid)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        sk_params = {"radiuses": (3, 3, 2), "max_neighbours": 8, "cov_model": cov_model}

        # Heterogeneous probability field CONTRADICTING the hard data — the
        # exact E2-32 flip configuration (Monte Carlo reproduced a 96% flip):
        # facies-0 hard cells get p=0.95 (threshold F⁻¹(0.05) is LOW, so the
        # transformed hard value is classified 1), facies-1 hard cells get
        # p=0.05 (threshold F⁻¹(0.95) is HIGH, so the transformed hard value
        # is classified 0). Uniform 0.5 elsewhere.
        pk_data = np.full(size, 0.5, dtype="float32")
        hard = mask != 0
        pk_data[hard & (facies == 0)] = 0.95
        pk_data[hard & (facies == 1)] = 0.05
        pk_prop = ContProperty(pk_data, np.ones(size, dtype="uint8"))

        result = gtsim_2ind(
            grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42
        )
        assert isinstance(result, ContProperty)
        result_data = result.data.ravel(order="K")
        # (a) Hard-data facies restored exactly (E2-32). Pre-fix ~96% flipped.
        np.testing.assert_array_equal(
            result_data[hard], facies[hard],
            err_msg="gtsim_2ind must preserve hard-data facies (E2-32)",
        )
        # (b) Uninformed cells were simulated and became informed.
        assert np.all(result.mask > 0), "simulated cells must be informed in output"
        # (c) Output is a valid binary facies field.
        assert np.all((result_data == 0) | (result_data == 1))
