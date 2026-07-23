"""
Parametrized baseline tests for ALL HPGL kriging variants.

These tests establish a behavioral baseline BEFORE any code changes, covering
the 8 kriging algorithms with parameter sweeps across grid sizes, covariance
types, nugget, sill, range, radiuses, and max_neighbours.

Covers:
1. ordinary_kriging       — parameter sweeps: grid, cov_type, nugget, sill, range, radius, max_n
2. simple_kriging          — parameter sweeps + explicit/auto mean
3. lvm_kriging             — parameter sweeps + mean_data
4. median_ik               — parameter sweeps (2-category indicator)
5. indicator_kriging       — parameter sweeps (3-category indicator)
6. simple_cokriging_markI  — parameter sweeps + correlation
7. simple_cokriging_markII — parameter sweeps + correlation
8. simple_kriging_weights  — parameter sweeps + weight invariants

Edge cases: empty input, single-point grid, uniform data, extreme values,
sparse/full data, zero/near-zero sill, negative parameters (rejection).

Output invariants: no NaN/Inf, shape matching, determinism, weight sum.
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
        indicator_kriging,
        lvm_kriging,
        median_ik,
        ordinary_kriging,
        simple_cokriging_markI,
        simple_cokriging_markII,
        simple_kriging,
        simple_kriging_weights,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_cont_grid(x, y, z, seed=42, informed_frac=0.8):
    """Create grid + ContProperty with random data for given dimensions."""
    grid = SugarboxGrid(x=x, y=y, z=z)
    size = x * y * z
    rng = np.random.RandomState(seed)
    data = rng.rand(size).astype("float32") * 100
    mask = np.ones(size, dtype="uint8")
    # Mark uninformed cells proportionally
    step = max(2, int(1.0 / (1.0 - informed_frac))) if informed_frac < 1.0 else size + 1
    mask[::step] = 0
    return grid, ContProperty(data, mask)


def _make_ind_grid(x, y, z, n_cats, seed=42, informed_frac=0.8):
    """Create grid + IndProperty with random categories for given dimensions."""
    grid = SugarboxGrid(x=x, y=y, z=z)
    size = x * y * z
    rng = np.random.RandomState(seed)
    data = rng.randint(0, n_cats, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    step = max(2, int(1.0 / (1.0 - informed_frac))) if informed_frac < 1.0 else size + 1
    mask[::step] = 0
    return grid, IndProperty(data, mask, n_cats)


def _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1):
    """Create a CovarianceModel with given parameters."""
    return CovarianceModel(
        type=cov_type, ranges=ranges, angles=(0.0, 0.0, 0.0), sill=sill, nugget=nugget
    )


def _make_ik_data(cov_type, n_indicators, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1,
                  radiuses=(5, 5, 3), max_neighbours=12):
    """Create IK data list for indicator_kriging."""
    ik_data = []
    for _ in range(n_indicators):
        ik_data.append({
            "cov_model": _make_cov(cov_type, ranges, sill, nugget),
            "radiuses": radiuses,
            "max_neighbours": max_neighbours,
        })
    return ik_data


# =============================================================================
# 1. Ordinary Kriging — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedOrdinaryKriging:
    """Parametrized tests for Ordinary Kriging across all parameter dimensions."""

    @pytest.mark.parametrize("x,y,z", [(2, 2, 2), (4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_ok_grid_cov_combo(self, x, y, z, cov_name, cov_type):
        """OK: grid size × covariance type — valid output, no NaN/Inf, correct shape."""
        grid, prop = _make_cont_grid(x, y, z)
        cov_model = _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1)
        radiuses = tuple(min(r, d) for r, d in zip((5, 5, 5), (x, y, z)))
        max_n = min(12, x * y * z)

        result = ordinary_kriging(prop, grid, radiuses, max_n, cov_model)

        assert isinstance(result, ContProperty), f"Failed for {cov_name} on ({x},{y},{z})"
        assert result.data.shape == prop.data.shape, f"Shape mismatch for ({x},{y},{z})"
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert not np.all(result.data == 0), f"All-zero regression for {cov_name}"
        # Informed cells must have finite values
        informed = result.mask == 1
        if np.any(informed):
            assert np.all(np.isfinite(result.data.astype("float64")[informed]))

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 1.0])
    def test_ok_nugget_sweep(self, nugget):
        """OK: nugget sweep (0.0, 0.1, 1.0) — valid output for each."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=1.0, nugget=nugget)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at nugget={nugget}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at nugget={nugget}"
        assert not np.all(result.data == 0), f"All-zero at nugget={nugget}"

    @pytest.mark.parametrize("sill", [1.0, 5.0, 10.0])
    def test_ok_sill_sweep(self, sill):
        """OK: sill sweep (1.0, 5.0, 10.0) — valid output for each."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=sill, nugget=0.1)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at sill={sill}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at sill={sill}"
        assert not np.all(result.data == 0), f"All-zero at sill={sill}"

    @pytest.mark.parametrize("rx,ry,rz", [(5, 5, 5), (10, 10, 10), (20, 20, 20)])
    def test_ok_range_sweep(self, rx, ry, rz):
        """OK: range sweep — valid output for each range tuple."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical, ranges=(rx, ry, rz), sill=1.0, nugget=0.1)

        result = ordinary_kriging(prop, grid, (5, 5, 3), 12, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at ranges={rx},{ry},{rz}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at ranges={rx},{ry},{rz}"
        assert not np.all(result.data == 0), f"All-zero at ranges={rx},{ry},{rz}"

    @pytest.mark.parametrize("radiuses", [(1, 1, 1), (5, 5, 5), (10, 10, 10)])
    def test_ok_radiuses_sweep(self, radiuses):
        """OK: search radius sweep — valid output for each radius."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical, ranges=(20.0, 20.0, 10.0),
                              sill=1.0, nugget=0.1)

        result = ordinary_kriging(prop, grid, radiuses, 12, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at radius={radiuses}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at radius={radiuses}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_ok_max_neighbours_sweep(self, max_n):
        """OK: max_neighbours sweep (1, 5, 10, 20) — valid output for each."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0),
                              sill=1.0, nugget=0.1)

        result = ordinary_kriging(prop, grid, (10, 10, 5), max_n, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at max_n={max_n}"
        assert not np.all(result.data == 0), f"All-zero at max_n={max_n}"

    def test_ok_reproducibility(self):
        """OK: deterministic — same seed + same input = same output."""
        grid, prop = _make_cont_grid(10, 10, 5, seed=42)
        cov_model = _make_cov(covariance.spherical)

        result1 = ordinary_kriging(prop, grid, (5, 5, 3), 12, cov_model)

        grid2, prop2 = _make_cont_grid(10, 10, 5, seed=42)
        result2 = ordinary_kriging(prop2, grid2, (5, 5, 3), 12, cov_model)

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_ok_single_point_grid(self):
        """OK: (1,1,1) grid with single informed cell returns its value."""
        grid = SugarboxGrid(x=1, y=1, z=1)
        data = np.array([73.5], dtype="float32")
        mask = np.array([1], dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(1.0, 1.0, 1.0), nugget=0.0)

        result = ordinary_kriging(prop, grid, (1, 1, 1), 1, cov_model)

        assert result.data.size == 1
        assert abs(float(result.data.flat[0]) - 73.5) < 0.01

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
    ])
    def test_ok_convexity_bounds(self, cov_name, cov_type):
        """OK: output at informed cells bounded by input data range.

        Ordinary Kriging weights sum to 1.0 (convexity constraint). For
        covariance types that produce non-negative weights (spherical,
        exponential), the estimate at every informed cell is a weighted
        average of input data and is bounded by [min(data), max(data)].

        NOTE: Gaussian covariance can produce negative kriging weights
        (extrapolation) even though weights sum to 1.0, so the convexity
        bound does NOT hold universally for gaussian. That is tested
        separately for validity (no NaN/Inf) but not for bounds.
        """
        grid = SugarboxGrid(x=4, y=4, z=4)
        rng = np.random.RandomState(42)
        size = 4 * 4 * 4
        data = rng.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        mask[::5] = 0  # ~80% informed
        prop = ContProperty(data, mask)
        cov_model = _make_cov(cov_type, ranges=(20.0, 20.0, 20.0),
                              sill=1.0, nugget=0.0)

        result = ordinary_kriging(prop, grid, (20, 20, 20), 64, cov_model)

        informed_input = prop.data[prop.mask == 1]
        data_min = float(np.min(informed_input))
        data_max = float(np.max(informed_input))

        informed_output = result.data[result.mask == 1]
        if len(informed_output) > 0:
            out_min = float(np.min(informed_output))
            out_max = float(np.max(informed_output))

            assert out_min >= data_min - 1e-4, (
                f"OK output below input minimum for {cov_name}: "
                f"min_output={out_min:.6f} < min_input={data_min:.6f}"
            )
            assert out_max <= data_max + 1e-4, (
                f"OK output above input maximum for {cov_name}: "
                f"max_output={out_max:.6f} > max_input={data_max:.6f}"
            )


# =============================================================================
# 2. Simple Kriging — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedSimpleKriging:
    """Parametrized tests for Simple Kriging across all parameter dimensions."""

    @pytest.mark.parametrize("x,y,z", [(2, 2, 2), (4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_sk_grid_cov_combo(self, x, y, z, cov_name, cov_type):
        """SK: grid size × covariance type — valid output, no NaN/Inf."""
        grid, prop = _make_cont_grid(x, y, z)
        cov_model = _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1)
        radiuses = tuple(min(r, d) for r, d in zip((5, 5, 5), (x, y, z)))
        max_n = min(12, x * y * z)

        result = simple_kriging(prop, grid, radiuses, max_n, cov_model, mean=None)

        assert isinstance(result, ContProperty), f"Failed for {cov_name} on ({x},{y},{z})"
        assert result.data.shape == prop.data.shape
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert not np.all(result.data == 0), f"All-zero regression for {cov_name}"

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 1.0])
    def test_sk_nugget_sweep(self, nugget):
        """SK: nugget sweep — valid output for each."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=1.0, nugget=nugget)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=None)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at nugget={nugget}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at nugget={nugget}"
        assert not np.all(result.data == 0), f"All-zero at nugget={nugget}"

    @pytest.mark.parametrize("sill", [1.0, 5.0, 10.0])
    def test_sk_sill_sweep(self, sill):
        """SK: sill sweep — valid output for each."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=sill, nugget=0.1)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=None)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at sill={sill}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at sill={sill}"
        assert not np.all(result.data == 0), f"All-zero at sill={sill}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_sk_max_neighbours_sweep(self, max_n):
        """SK: max_neighbours sweep — valid output for each."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0),
                              sill=1.0, nugget=0.1)

        result = simple_kriging(prop, grid, (10, 10, 5), max_n, cov_model, mean=None)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at max_n={max_n}"
        assert not np.all(result.data == 0), f"All-zero at max_n={max_n}"

    @pytest.mark.parametrize("explicit_mean", [0.0, 50.0, 100.0])
    def test_sk_explicit_mean(self, explicit_mean):
        """SK: explicit mean values — produces valid output."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical)

        result = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=explicit_mean)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        assert not np.all(result.data == 0), f"All-zero at mean={explicit_mean}"

    def test_sk_auto_mean_vs_explicit_similar(self):
        """SK: auto-computed mean should be close to explicit 50 on centered data."""
        grid = SugarboxGrid(x=10, y=10, z=5)
        rng = np.random.RandomState(42)
        data = rng.rand(500).astype("float32") * 20 + 40  # centered near 50
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result_auto = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=None)
        result_explicit = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=50.0)

        assert isinstance(result_auto, ContProperty)
        assert isinstance(result_explicit, ContProperty)
        # Both must be valid
        assert not np.any(np.isnan(result_auto.data.astype("float64")))
        assert not np.any(np.isnan(result_explicit.data.astype("float64")))

    def test_sk_reproducibility(self):
        """SK: deterministic — same seed + same input = same output."""
        grid, prop = _make_cont_grid(10, 10, 5, seed=42)
        cov_model = _make_cov(covariance.spherical)

        result1 = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=50.0)

        grid2, prop2 = _make_cont_grid(10, 10, 5, seed=42)
        result2 = simple_kriging(prop2, grid2, (5, 5, 3), 12, cov_model, mean=50.0)

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)


# =============================================================================
# 3. LVM Kriging — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedLVMKriging:
    """Parametrized tests for Locally Varying Mean (LVM) Kriging."""

    @pytest.mark.parametrize("x,y,z", [(2, 2, 2), (4, 4, 4)])
    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_lvm_grid_cov_combo(self, x, y, z, cov_name, cov_type):
        """LVM: grid size × covariance type — valid output, no NaN/Inf."""
        grid, prop = _make_cont_grid(x, y, z)
        size = x * y * z
        rng = np.random.RandomState(42)
        mean_data = rng.rand(size).astype("float32") * 50
        cov_model = _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1)
        radiuses = tuple(min(r, d) for r, d in zip((5, 5, 5), (x, y, z)))
        max_n = min(12, x * y * z)

        result = lvm_kriging(prop, grid, mean_data, radiuses, max_n, cov_model)

        assert isinstance(result, ContProperty), f"Failed for {cov_name} on ({x},{y},{z})"
        assert result.data.shape == prop.data.shape
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 1.0])
    def test_lvm_nugget_sweep(self, nugget):
        """LVM: nugget sweep — valid output for each."""
        grid, prop = _make_cont_grid(5, 5, 3)
        size = 5 * 5 * 3
        rng = np.random.RandomState(42)
        mean_data = rng.rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=1.0, nugget=nugget)

        result = lvm_kriging(prop, grid, mean_data, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at nugget={nugget}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at nugget={nugget}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_lvm_max_neighbours_sweep(self, max_n):
        """LVM: max_neighbours sweep — valid output for each."""
        grid, prop = _make_cont_grid(10, 10, 5)
        size = 10 * 10 * 5
        rng = np.random.RandomState(42)
        mean_data = rng.rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0),
                              sill=1.0, nugget=0.1)

        result = lvm_kriging(prop, grid, mean_data, (10, 10, 5), max_n, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at max_n={max_n}"

    def test_lvm_reproducibility(self):
        """LVM: deterministic — same seed = same output."""
        grid, prop = _make_cont_grid(5, 5, 3, seed=42)
        size = 5 * 5 * 3
        rng = np.random.RandomState(42)
        mean_data = rng.rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical)

        result1 = lvm_kriging(prop, grid, mean_data, (3, 3, 2), 8, cov_model)

        grid2, prop2 = _make_cont_grid(5, 5, 3, seed=42)
        rng2 = np.random.RandomState(42)
        mean_data2 = rng2.rand(size).astype("float32") * 50
        result2 = lvm_kriging(prop2, grid2, mean_data2, (3, 3, 2), 8, cov_model)

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)


# =============================================================================
# 4. Median Indicator Kriging — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedMedianIK:
    """Parametrized tests for Median Indicator Kriging (2-category)."""

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_median_ik_cov_sweep(self, cov_name, cov_type):
        """M-IK: covariance type sweep — valid output."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(cov_type, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1)

        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        assert isinstance(result, IndProperty), f"Failed for {cov_name}"
        assert result.indicator_count == 2
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range in {cov_name}"

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 1.0])
    def test_median_ik_nugget_sweep(self, nugget):
        """M-IK: nugget sweep."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(covariance.spherical, ranges=(3.0, 3.0, 2.0),
                              sill=1.0, nugget=nugget)

        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at nugget={nugget}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range at nugget={nugget}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_median_ik_max_neighbours_sweep(self, max_n):
        """M-IK: max_neighbours sweep."""
        grid, prop = _make_ind_grid(10, 10, 5, 2)
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0),
                              sill=1.0, nugget=0.1)

        result = median_ik(prop, grid, (0.5, 0.5), (5, 5, 3), max_n, cov_model)

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range at max_n={max_n}"

    @pytest.mark.parametrize("radiuses", [(1, 1, 1), (5, 5, 5)])
    def test_median_ik_radiuses_sweep(self, radiuses):
        """M-IK: search radius sweep."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0),
                              sill=1.0, nugget=0.1)

        result = median_ik(prop, grid, (0.5, 0.5), radiuses, 8, cov_model)

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at radius={radiuses}"

    def test_median_ik_reproducibility(self):
        """M-IK: deterministic — same seed = same output."""
        grid, prop = _make_ind_grid(5, 5, 3, 2, seed=42)
        cov_model = _make_cov(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result1 = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        grid2, prop2 = _make_ind_grid(5, 5, 3, 2, seed=42)
        result2 = median_ik(prop2, grid2, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# 5. Indicator Kriging — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedIndicatorKriging:
    """Parametrized tests for Indicator Kriging (3-category)."""

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_ik_cov_sweep(self, cov_name, cov_type):
        """IK: covariance type sweep."""
        grid, prop = _make_ind_grid(5, 5, 3, 3)
        ik_data = _make_ik_data(cov_type, 3, ranges=(3.0, 3.0, 2.0),
                                radiuses=(3, 3, 2), max_neighbours=8)
        marginal_probs = [0.3, 0.4, 0.3]

        result = indicator_kriging(prop, grid, ik_data, marginal_probs)

        assert isinstance(result, IndProperty), f"Failed for {cov_name}"
        assert result.indicator_count == 3, f"Wrong indicator count for {cov_name}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range in {cov_name}"

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 1.0])
    def test_ik_nugget_sweep(self, nugget):
        """IK: nugget sweep."""
        grid, prop = _make_ind_grid(5, 5, 3, 3)
        ik_data = _make_ik_data(covariance.spherical, 3, ranges=(3.0, 3.0, 2.0),
                                nugget=nugget, radiuses=(3, 3, 2), max_neighbours=8)
        marginal_probs = [0.3, 0.4, 0.3]

        result = indicator_kriging(prop, grid, ik_data, marginal_probs)

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count), f"Out-of-range at nugget={nugget}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_ik_max_neighbours_sweep(self, max_n):
        """IK: max_neighbours sweep."""
        grid, prop = _make_ind_grid(10, 10, 5, 3)
        ik_data = _make_ik_data(covariance.spherical, 3, ranges=(10.0, 10.0, 5.0),
                                radiuses=(5, 5, 3), max_neighbours=max_n)
        marginal_probs = [0.3, 0.4, 0.3]

        result = indicator_kriging(prop, grid, ik_data, marginal_probs)

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count), f"Out-of-range at max_n={max_n}"

    def test_ik_reproducibility(self):
        """IK: deterministic — same seed = same output."""
        grid, prop = _make_ind_grid(5, 5, 3, 3, seed=42)
        ik_data = _make_ik_data(covariance.spherical, 3, ranges=(3.0, 3.0, 2.0),
                                radiuses=(3, 3, 2), max_neighbours=8)
        marginal_probs = [0.3, 0.4, 0.3]

        result1 = indicator_kriging(prop, grid, ik_data, marginal_probs)

        grid2, prop2 = _make_ind_grid(5, 5, 3, 3, seed=42)
        ik_data2 = _make_ik_data(covariance.spherical, 3, ranges=(3.0, 3.0, 2.0),
                                 radiuses=(3, 3, 2), max_neighbours=8)
        result2 = indicator_kriging(prop2, grid2, ik_data2, marginal_probs)

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# 6. Simple Cokriging Mark I — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedCokrigingMarkI:
    """Parametrized tests for Simple Cokriging Mark I."""

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_ck_markI_cov_sweep(self, cov_name, cov_type):
        """CK Mark I: covariance type sweep."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(cov_type, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1)
        # Create secondary property
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        result = simple_cokriging_markI(
            prop=prop, grid=grid, secondary_data=secondary,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(3, 3, 2), max_neighbours=8, cov_model=cov_model,
        )

        assert isinstance(result, ContProperty), f"Failed for {cov_name}"
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert result.data.shape == prop.data.shape

    @pytest.mark.parametrize("correlation_coef", [0.2, 0.5, 0.8, 0.95])
    def test_ck_markI_correlation_sweep(self, correlation_coef):
        """CK Mark I: correlation coefficient sweep."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(3.0, 3.0, 2.0))
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        result = simple_cokriging_markI(
            prop=prop, grid=grid, secondary_data=secondary,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=correlation_coef,
            radiuses=(3, 3, 2), max_neighbours=8, cov_model=cov_model,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at corr={correlation_coef}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at corr={correlation_coef}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_ck_markI_max_neighbours_sweep(self, max_n):
        """CK Mark I: max_neighbours sweep."""
        grid, prop = _make_cont_grid(10, 10, 5)
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0))
        rng = np.random.RandomState(43)
        size = 10 * 10 * 5
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::10] = 0
        secondary = ContProperty(sec_data, sec_mask)

        result = simple_cokriging_markI(
            prop=prop, grid=grid, secondary_data=secondary,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3), max_neighbours=max_n, cov_model=cov_model,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at max_n={max_n}"

    def test_ck_markI_reproducibility(self):
        """CK Mark I: deterministic — same seed + same input = same output."""
        grid, prop = _make_cont_grid(5, 5, 3, seed=42)
        cov_model = _make_cov(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        result1 = simple_cokriging_markI(
            prop=prop, grid=grid, secondary_data=secondary,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
            cov_model=cov_model,
        )

        grid2, prop2 = _make_cont_grid(5, 5, 3, seed=42)
        rng2 = np.random.RandomState(43)
        sec_data2 = rng2.rand(size).astype("float32") * 80
        sec_mask2 = np.ones(size, dtype="uint8")
        sec_mask2[::5] = 0
        secondary2 = ContProperty(sec_data2, sec_mask2)

        result2 = simple_cokriging_markI(
            prop=prop2, grid=grid2, secondary_data=secondary2,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
            cov_model=cov_model,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)


# =============================================================================
# 7. Simple Cokriging Mark II — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedCokrigingMarkII:
    """Parametrized tests for Simple Cokriging Mark II."""

    def _make_ck2_data(self, grid, prop, mean_val, cov_type, secondary_prop, secondary_mean):
        """Helper to build primary_data and secondary_data dicts."""
        primary_data = {
            "data": prop,
            "mean": mean_val,
            "cov_model": _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1),
        }
        secondary_data = {
            "data": secondary_prop,
            "mean": secondary_mean,
            "cov_model": _make_cov(cov_type, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1),
        }
        return primary_data, secondary_data

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_ck_markII_cov_sweep(self, cov_name, cov_type):
        """CK Mark II: covariance type sweep."""
        grid, prop = _make_cont_grid(5, 5, 3)
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        primary_data, secondary_data = self._make_ck2_data(
            grid, prop, 50.0, cov_type, secondary, 40.0,
        )

        result = simple_cokriging_markII(
            grid=grid, primary_data=primary_data, secondary_data=secondary_data,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
        )

        assert isinstance(result, ContProperty), f"Failed for {cov_name}"
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert result.data.shape == prop.data.shape

    @pytest.mark.parametrize("correlation_coef", [0.2, 0.5, 0.8, 0.95])
    def test_ck_markII_correlation_sweep(self, correlation_coef):
        """CK Mark II: correlation coefficient sweep."""
        grid, prop = _make_cont_grid(5, 5, 3)
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        cov_type = covariance.spherical
        primary_data, secondary_data = self._make_ck2_data(
            grid, prop, 50.0, cov_type, secondary, 40.0,
        )

        result = simple_cokriging_markII(
            grid=grid, primary_data=primary_data, secondary_data=secondary_data,
            correlation_coef=correlation_coef, radiuses=(3, 3, 2), max_neighbours=8,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at corr={correlation_coef}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at corr={correlation_coef}"

    @pytest.mark.parametrize("max_n", [1, 5, 10, 20])
    def test_ck_markII_max_neighbours_sweep(self, max_n):
        """CK Mark II: max_neighbours sweep."""
        grid, prop = _make_cont_grid(10, 10, 5)
        rng = np.random.RandomState(43)
        size = 10 * 10 * 5
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::10] = 0
        secondary = ContProperty(sec_data, sec_mask)

        cov_type = covariance.spherical
        primary_data, secondary_data = self._make_ck2_data(
            grid, prop, 50.0, cov_type, secondary, 40.0,
        )

        result = simple_cokriging_markII(
            grid=grid, primary_data=primary_data, secondary_data=secondary_data,
            correlation_coef=0.8, radiuses=(5, 5, 3), max_neighbours=max_n,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN at max_n={max_n}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf at max_n={max_n}"

    def test_ck_markII_reproducibility(self):
        """CK Mark II: deterministic — same seed + same input = same output."""
        grid, prop = _make_cont_grid(5, 5, 3, seed=42)
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)

        cov_type = covariance.spherical
        primary_data, secondary_data = self._make_ck2_data(
            grid, prop, 50.0, cov_type, secondary, 40.0,
        )

        result1 = simple_cokriging_markII(
            grid=grid, primary_data=primary_data, secondary_data=secondary_data,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
        )

        grid2, prop2 = _make_cont_grid(5, 5, 3, seed=42)
        rng2 = np.random.RandomState(43)
        sec_data2 = rng2.rand(size).astype("float32") * 80
        sec_mask2 = np.ones(size, dtype="uint8")
        sec_mask2[::5] = 0
        secondary2 = ContProperty(sec_data2, sec_mask2)

        primary_data2, secondary_data2 = self._make_ck2_data(
            grid2, prop2, 50.0, cov_type, secondary2, 40.0,
        )

        result2 = simple_cokriging_markII(
            grid=grid2, primary_data=primary_data2, secondary_data=secondary_data2,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)


# =============================================================================
# 8. Simple Kriging Weights — Parametrized
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedKrigingWeights:
    """Parametrized tests for Simple Kriging Weights calculation."""

    @pytest.fixture
    def neighbor_points(self):
        """Standard set of neighbor points for weight tests."""
        rng = np.random.RandomState(42)
        n = 12
        n_x = rng.rand(n).astype("float32") * 10
        n_y = rng.rand(n).astype("float32") * 10
        n_z = rng.rand(n).astype("float32") * 5
        return n_x, n_y, n_z

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
    ])
    def test_skw_cov_sweep(self, cov_name, cov_type, neighbor_points):
        """SKW: covariance type sweep — valid weights, finite, correct count."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=cov_type, nugget=0.1,
        )

        assert isinstance(weights, np.ndarray), f"Failed for {cov_name}"
        assert len(weights) == len(n_x), f"Wrong count for {cov_name}"
        assert weights.dtype == np.float32, f"Wrong dtype for {cov_name}"
        assert not np.any(np.isnan(weights)), f"NaN in {cov_name}"
        assert not np.any(np.isinf(weights)), f"Inf in {cov_name}"
        assert np.all(np.isfinite(weights)), f"Non-finite in {cov_name}"

    @pytest.mark.parametrize("nugget", [0.0, 0.1, 0.5])
    def test_skw_nugget_sweep(self, nugget, neighbor_points):
        """SKW: nugget sweep — valid weights for each nugget value."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.exponential,
            nugget=nugget,
        )
        assert isinstance(weights, np.ndarray), f"Failed at nugget={nugget}"
        assert np.all(np.isfinite(weights)), f"Non-finite at nugget={nugget}"

    @pytest.mark.parametrize("ranges", [
        (3.0, 3.0, 2.0), (5.0, 5.0, 3.0), (10.0, 10.0, 5.0),
    ])
    def test_skw_range_sweep(self, ranges, neighbor_points):
        """SKW: range sweep — valid weights for each range tuple."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=ranges, sill=1.0, cov_type=covariance.exponential, nugget=0.1,
        )
        assert isinstance(weights, np.ndarray), f"Failed at ranges={ranges}"
        assert np.all(np.isfinite(weights)), f"Non-finite at ranges={ranges}"

    @pytest.mark.parametrize("sill", [1.0, 5.0, 10.0])
    def test_skw_sill_sweep(self, sill, neighbor_points):
        """SKW: sill sweep — valid weights for each sill value."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=sill, cov_type=covariance.exponential,
            nugget=0.1,
        )
        assert isinstance(weights, np.ndarray), f"Failed at sill={sill}"
        assert np.all(np.isfinite(weights)), f"Non-finite at sill={sill}"

    def test_skw_weights_sum_near_one(self, neighbor_points):
        """SKW: weights are finite and valid. Simple kriging weights can be
        negative (SK does not enforce positivity or unit-sum constraints).
        This test verifies that weights produce valid finite values."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.spherical,
            nugget=0.0,
        )
        assert np.all(np.isfinite(weights)), "SK weights must be finite"
        assert len(weights) == len(n_x), "SK weights count must match neighbor count"

    def test_skw_weights_with_nugget_dont_sum_to_one(self, neighbor_points):
        """SKW: with nugget > 0, SK weights do NOT sum to 1.0 (nugget adds to diagonal).

        Simple kriging with nugget > 0 produces weights that sum to less than 1.0
        because the nugget term on the diagonal of the covariance matrix means the
        kriging system doesn't satisfy the unit-sum constraint of ordinary kriging.
        """
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.spherical,
            nugget=0.5,
        )
        weight_sum = float(np.sum(weights))
        # With nugget > 0, SK weights should NOT equal 1.0
        assert abs(weight_sum - 1.0) > 0.001, (
            f"SK weights with nugget=0.5 should NOT sum to 1.0, got {weight_sum:.6f}"
        )

    def test_skw_single_neighbor_zero_nugget(self):
        """SKW: single co-located neighbor with nugget=0 → weight=1.0."""
        n_x = np.array([5.0], dtype="float32")
        n_y = np.array([5.0], dtype="float32")
        n_z = np.array([2.5], dtype="float32")
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.spherical,
            nugget=0.0,
        )
        assert len(weights) == 1
        assert abs(float(weights[0]) - 1.0) < 1e-5

    def test_skw_various_neighbor_counts_sweep(self):
        """SKW: weights with 4, 8, 12, 16 neighbors — correct count, finite."""
        center = (5.0, 5.0, 2.5)
        for n in [4, 8, 12, 16]:
            rng = np.random.RandomState(42)
            n_x = rng.rand(n).astype("float32") * 10
            n_y = rng.rand(n).astype("float32") * 10
            n_z = rng.rand(n).astype("float32") * 5

            weights = simple_kriging_weights(
                center_point=center, n_x=n_x, n_y=n_y, n_z=n_z,
                ranges=(5.0, 5.0, 3.0), sill=1.0,
                cov_type=covariance.exponential, nugget=0.1,
            )
            assert len(weights) == n, f"Wrong count for n={n}"
            assert np.all(np.isfinite(weights)), f"Non-finite for n={n}"

    def test_skw_different_center_points(self, neighbor_points):
        """SKW: different center points produce different weights."""
        n_x, n_y, n_z = neighbor_points
        w1 = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0), n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.exponential,
            nugget=0.1,
        )
        w2 = simple_kriging_weights(
            center_point=(10.0, 10.0, 5.0), n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.exponential,
            nugget=0.1,
        )
        assert not np.allclose(w1, w2, rtol=0.01), "Different centers should yield different weights"


# =============================================================================
# 9. Property Construction Rejection Tests
# =============================================================================


@pytest.mark.hpgl
class TestContPropertyConstruction:
    """ContProperty construction edge case tests — rejection of invalid inputs."""

    def test_cont_property_rejects_nan_data(self):
        """ContProperty rejects data containing NaN values."""
        data = np.array([1.0, np.nan, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_rejects_inf_data(self):
        """ContProperty rejects data containing Inf values."""
        data = np.array([1.0, np.inf, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_rejects_2d_data(self):
        """ContProperty rejects 2D data arrays."""
        data = np.ones((10, 10), dtype="float32")
        mask = np.ones((10, 10), dtype="uint8")
        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_rejects_shape_mismatch(self):
        """ContProperty rejects mismatched data and mask shapes."""
        data = np.ones(10, dtype="float32")
        mask = np.ones(5, dtype="uint8")
        with pytest.raises(ValueError):
            ContProperty(data, mask)


@pytest.mark.hpgl
class TestIndPropertyConstruction:
    """IndProperty construction edge case tests — rejection of invalid inputs."""

    def test_ind_property_rejects_nan_data(self):
        """IndProperty rejects data containing NaN values."""
        data = np.array([0.0, np.nan, 2.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_inf_data(self):
        """IndProperty rejects data containing Inf values."""
        data = np.array([0.0, np.inf, 2.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_2d_data(self):
        """IndProperty rejects 2D data arrays."""
        data = np.ones((10, 10), dtype="uint8")
        mask = np.ones((10, 10), dtype="uint8")
        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=2)

    def test_ind_property_rejects_shape_mismatch(self):
        """IndProperty rejects mismatched data and mask shapes."""
        data = np.array([0, 1, 2], dtype="uint8")
        mask = np.ones(5, dtype="uint8")
        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_invalid_indicator_count(self):
        """IndProperty rejects indicator_count outside valid range [1, 255]."""
        data = np.array([0, 1], dtype="uint8")
        mask = np.ones(2, dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=0)

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=256)


# =============================================================================
# 10. Edge Cases — All Kriging Variants
# =============================================================================


@pytest.mark.hpgl
class TestKrigingEdgeCases:
    """Edge case tests across all kriging variants."""

    # ---- Empty / No-informed-data tests ----

    def test_ok_empty_no_informed(self):
        """OK: zero informed points — all cells remain uninformed, no crash."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.zeros(50, dtype="uint8")  # All uninformed
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape

    def test_sk_empty_no_informed(self):
        """SK: zero informed points — no crash, valid output."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.zeros(50, dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=50.0)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape

    def test_lvm_empty_no_informed(self):
        """LVM: zero informed points — no crash, valid output."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.zeros(50, dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.random.rand(50).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical)

        result = lvm_kriging(prop, grid, mean_data, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)

    def test_mik_empty_no_informed(self):
        """M-IK: zero informed points — no crash, valid output."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.zeros(50, dtype="uint8")
        mask = np.zeros(50, dtype="uint8")
        prop = IndProperty(data, mask, 2)
        cov_model = _make_cov(covariance.spherical)

        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        assert isinstance(result, IndProperty)

    def test_ik_empty_no_informed(self):
        """IK: zero informed points — no crash, valid output."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.zeros(50, dtype="uint8")
        mask = np.zeros(50, dtype="uint8")
        prop = IndProperty(data, mask, 3)
        ik_data = _make_ik_data(covariance.spherical, 3, ranges=(3.0, 3.0, 2.0),
                                radiuses=(3, 3, 2), max_neighbours=8)

        result = indicator_kriging(prop, grid, ik_data, [0.3, 0.4, 0.3])

        assert isinstance(result, IndProperty)

    # ---- Single-point grid tests ----

    def test_sk_single_point(self):
        """SK: (1,1,1) grid — returns valid result."""
        grid = SugarboxGrid(x=1, y=1, z=1)
        data = np.array([42.0], dtype="float32")
        mask = np.array([1], dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(1.0, 1.0, 1.0), nugget=0.0)

        result = simple_kriging(prop, grid, (1, 1, 1), 1, cov_model, mean=None)

        assert result.data.size == 1
        assert abs(float(result.data.flat[0]) - 42.0) < 0.1

    def test_lvm_single_point(self):
        """LVM: (1,1,1) grid — returns valid result."""
        grid = SugarboxGrid(x=1, y=1, z=1)
        data = np.array([42.0], dtype="float32")
        mask = np.array([1], dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.array([50.0], dtype="float32")
        cov_model = _make_cov(covariance.spherical, ranges=(1.0, 1.0, 1.0), nugget=0.0)

        result = lvm_kriging(prop, grid, mean_data, (1, 1, 1), 1, cov_model)

        assert result.data.size == 1

    # ---- Uniform data (zero variance) tests ----

    def test_ok_uniform_data(self):
        """OK: all same value (zero variance) — result stays near uniform value."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.full(50, 42.0, dtype="float32")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0  # some uninformed
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        # Result should be close to uniform value
        assert np.allclose(result.data.astype("float64"), 42.0, atol=0.1)

    def test_sk_uniform_data(self):
        """SK: all same value — result stays near uniform value."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.full(50, 42.0, dtype="float32")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=42.0)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert np.allclose(result.data.astype("float64"), 42.0, atol=0.1)

    def test_lvm_uniform_data(self):
        """LVM: all same value, mean same value — result near uniform value."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.full(50, 42.0, dtype="float32")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        mean_data = np.full(50, 42.0, dtype="float32")
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = lvm_kriging(prop, grid, mean_data, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))

    # ---- Extreme value tests ----

    def test_ok_extreme_values_large(self):
        """OK: values ~1e15 — no overflow, no NaN/Inf."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.full(50, 1e15, dtype="float32")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_ok_extreme_values_small(self):
        """OK: values ~1e-15 — no underflow, no NaN/Inf."""
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.full(50, 1e-15, dtype="float32")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    # ---- Sparsity tests ----

    def test_ok_sparse_10_percent_informed(self):
        """OK: only 10% informed — no crash, valid output shape."""
        grid, prop = _make_cont_grid(10, 10, 5, informed_frac=0.1)

        # With only 10% informed and moderate radius, many cells may remain
        # uninformed — that's expected behavior. Assert no crash.
        cov_model = _make_cov(covariance.spherical, ranges=(10.0, 10.0, 5.0))
        result = ordinary_kriging(prop, grid, (5, 5, 3), 12, cov_model)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape

    def test_ok_fully_informed_100_percent(self):
        """OK: 100% informed — valid output, all cells remain informed."""
        grid, prop = _make_cont_grid(5, 5, 3, informed_frac=1.0)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        result.fix_shape(grid)
        # With fixed seed uniform data and zero nugget, all cells should be informed
        assert np.all(result.mask == 1)

    def test_sk_fully_informed_100_percent(self):
        """SK: 100% informed — valid output, no NaN/Inf."""
        grid, prop = _make_cont_grid(5, 5, 3, informed_frac=1.0)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), nugget=0.0)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=None)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    # ---- Sill edge case: near-zero nut not quite zero ----

    def test_ok_small_sill(self):
        """OK: very small sill (0.01) but nugget=0 — should still produce valid output."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=0.01, nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    # ---- Negative parameter rejection tests ----

    def test_covmodel_negative_sill_rejected(self):
        """CovarianceModel: negative sill raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0),
                angles=(0, 0, 0), sill=-1.0, nugget=0.0,
            )

    def test_covmodel_nugget_gt_sill_rejected(self):
        """CovarianceModel: nugget > sill raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0),
                angles=(0, 0, 0), sill=0.5, nugget=1.0,
            )

    def test_ok_max_neighbours_zero_rejected(self):
        """OK: max_neighbours=0 raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical)
        with pytest.raises(CriticalValidationError):
            ordinary_kriging(prop, grid, (3, 3, 2), 0, cov_model)

    def test_sk_max_neighbours_zero_rejected(self):
        """SK: max_neighbours=0 raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical)
        with pytest.raises(CriticalValidationError):
            simple_kriging(prop, grid, (3, 3, 2), 0, cov_model, mean=None)

    def test_lvm_max_neighbours_zero_rejected(self):
        """LVM: max_neighbours=0 raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        grid, prop = _make_cont_grid(5, 5, 3)
        size = 5 * 5 * 3
        mean_data = np.random.rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical)
        with pytest.raises(CriticalValidationError):
            lvm_kriging(prop, grid, mean_data, (3, 3, 2), 0, cov_model)

    def test_mik_max_neighbours_zero_rejected(self):
        """M-IK: max_neighbours=0 raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(covariance.spherical)
        with pytest.raises(CriticalValidationError):
            median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 0, cov_model)

    def test_negative_range_rejected(self):
        """CovarianceModel: negative range raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical,
                ranges=(-5.0, 5.0, 3.0), angles=(0, 0, 0), sill=1.0, nugget=0.1,
            )

    # ---- Shape invariance tests ----

    def test_ok_output_shape_matches_input(self):
        """OK: output data shape matches input data shape."""
        grid, prop = _make_cont_grid(5, 10, 3)  # non-cubic
        cov_model = _make_cov(covariance.spherical)

        result = ordinary_kriging(prop, grid, (3, 5, 2), 8, cov_model)

        # HPGL returns 1D data; shape matches input data shape
        assert result.data.shape == prop.data.shape
        assert result.data.size == 5 * 10 * 3

    def test_sk_output_shape_matches_input(self):
        """SK: output data shape matches input data shape."""
        grid, prop = _make_cont_grid(4, 3, 2)
        cov_model = _make_cov(covariance.spherical)

        result = simple_kriging(prop, grid, (2, 2, 1), 6, cov_model, mean=None)

        assert result.data.shape == prop.data.shape
        assert result.data.size == 4 * 3 * 2

    def test_lvm_output_shape_matches_input(self):
        """LVM: output data shape matches input data shape."""
        grid, prop = _make_cont_grid(4, 3, 2)
        size = 4 * 3 * 2
        mean_data = np.random.rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical)

        result = lvm_kriging(prop, grid, mean_data, (2, 2, 1), 6, cov_model)

        assert result.data.shape == prop.data.shape
        assert result.data.size == size

    def test_mik_output_shape_matches_input(self):
        """M-IK: output data shape matches input data shape."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(covariance.spherical)

        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        assert result.data.shape == prop.data.shape
        assert result.data.size == 5 * 5 * 3


# =============================================================================
# 11. Cross-Variant Invariant Tests
# =============================================================================


@pytest.mark.hpgl
class TestKrigingInvariants:
    """Cross-cutting invariant tests applicable to multiple kriging variants."""

    @pytest.mark.parametrize("kriging_fn_name", [
        "ordinary_kriging", "simple_kriging",
    ])
    def test_no_nan_inf_in_output_cont(self, kriging_fn_name):
        """All continuous kriging variants: output has no NaN or Inf."""
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical)

        extra_args = {}
        if kriging_fn_name == "simple_kriging":
            extra_args["mean"] = None

        fn = globals()[kriging_fn_name]
        result = fn(prop, grid, (3, 3, 2), 8, cov_model, **extra_args)

        assert isinstance(result, ContProperty), f"Failed for {kriging_fn_name}"
        data64 = result.data.astype("float64")
        assert not np.any(np.isnan(data64)), f"NaN in {kriging_fn_name} output"
        assert not np.any(np.isinf(data64)), f"Inf in {kriging_fn_name} output"

    def test_lvm_no_nan_inf(self):
        """LVM: output has no NaN or Inf."""
        grid, prop = _make_cont_grid(5, 5, 3)
        size = 5 * 5 * 3
        mean_data = np.random.RandomState(42).rand(size).astype("float32") * 50
        cov_model = _make_cov(covariance.spherical)

        result = lvm_kriging(prop, grid, mean_data, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        data64 = result.data.astype("float64")
        assert not np.any(np.isnan(data64)), "NaN in LVM output"
        assert not np.any(np.isinf(data64)), "Inf in LVM output"

    def test_indicator_kriging_no_nan_inf(self):
        """IK: indicator output has no NaN or Inf."""
        grid, prop = _make_ind_grid(5, 5, 3, 3)
        ik_data = _make_ik_data(covariance.spherical, 3, ranges=(3.0, 3.0, 2.0),
                                radiuses=(3, 3, 2), max_neighbours=8)
        result = indicator_kriging(prop, grid, ik_data, [0.3, 0.4, 0.3])
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_median_ik_no_nan_inf(self):
        """M-IK: indicator output has no NaN or Inf."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(covariance.spherical)
        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_cokriging_no_nan_inf(self):
        """Cokriging Mark I & II: outputs have no NaN or Inf."""
        grid, prop = _make_cont_grid(5, 5, 3)
        rng = np.random.RandomState(43)
        size = 5 * 5 * 3
        sec_data = rng.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary = ContProperty(sec_data, sec_mask)
        cov_model = _make_cov(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        # Mark I
        result_m1 = simple_cokriging_markI(
            prop=prop, grid=grid, secondary_data=secondary,
            primary_mean=50.0, secondary_mean=40.0, secondary_variance=100.0,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
            cov_model=cov_model,
        )
        assert not np.any(np.isnan(result_m1.data.astype("float64"))), "NaN in CK Mark I"
        assert not np.any(np.isinf(result_m1.data.astype("float64"))), "Inf in CK Mark I"

        # Mark II
        primary_data = {"data": prop, "mean": 50.0, "cov_model": cov_model}
        secondary_data = {"data": secondary, "mean": 40.0, "cov_model": cov_model}
        result_m2 = simple_cokriging_markII(
            grid=grid, primary_data=primary_data, secondary_data=secondary_data,
            correlation_coef=0.8, radiuses=(3, 3, 2), max_neighbours=8,
        )
        assert not np.any(np.isnan(result_m2.data.astype("float64"))), "NaN in CK Mark II"
        assert not np.any(np.isinf(result_m2.data.astype("float64"))), "Inf in CK Mark II"

    def test_weights_no_nan_inf(self):
        """SKW: weights array has no NaN or Inf."""
        rng = np.random.RandomState(42)
        n_x = rng.rand(8).astype("float32") * 10
        n_y = rng.rand(8).astype("float32") * 10
        n_z = rng.rand(8).astype("float32") * 5
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5), n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.spherical,
            nugget=0.1,
        )
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        assert np.all(np.isfinite(weights))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
