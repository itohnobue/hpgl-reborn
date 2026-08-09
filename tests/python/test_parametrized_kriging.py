"""
Parametrized baseline tests for HPGL kriging variants (slimmed core).

Covers the 8 kriging algorithms' value-level pins and edge cases. The
redundant sweep families (grid-size/cov-type/nugget/sill/range/radius/max_n
sweeps, reproducibility loops, cokriging smoke sweeps, invariants) live in
test_kriging_complete.py and are NOT duplicated here (v2.0.6 cleanup,
conflict #1 resolution: keep complete's sweeps).

This file keeps the unique survivors:
1. ordinary_kriging      — tiny-grid radius-clamp, single-point, convexity bounds
2. simple_kriging        — auto-mean vs explicit-mean comparison
3. median_ik             — covariance smoke (indicator_count==2 + range asserts)
4. indicator_kriging     — covariance smoke (indicator_count==3 + range asserts)
5. simple_kriging_weights— dtype contract + nugget reference solve + center effect
6. construction rejects  — ContProperty/IndProperty invalid input rejection
7. edge cases            — empty/no-informed ×5, single-point, uniform, sparse,
                           fully-informed, small-sill, OK max_n=0 rejection
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
        simple_kriging,
        simple_kriging_weights,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability

# N2-L03: the parametrize decorators below reference covariance.* at module
# level. If geo_bsd is unavailable the import above silently fails and
# `covariance` is undefined — collection would NameError 88 tests instead of
# skipping cleanly via the @hpgl marker. Provide a constant fallback so
# collection succeeds; the tests are still auto-skipped by conftest.
if "covariance" not in globals():
    class covariance:
        spherical = 0
        exponential = 1
        gaussian = 2


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
    # Mark uninformed cells proportionally. informed_frac=1.0 means ALL cells
    # stay informed (T-08: the old `size+1` step still masked index 0).
    if informed_frac < 1.0:
        step = max(2, int(1.0 / (1.0 - informed_frac)))
        mask[::step] = 0
    return grid, ContProperty(data, mask)


def _make_ind_grid(x, y, z, n_cats, seed=42, informed_frac=0.8):
    """Create grid + IndProperty with random categories for given dimensions."""
    grid = SugarboxGrid(x=x, y=y, z=z)
    size = x * y * z
    rng = np.random.RandomState(seed)
    data = rng.randint(0, n_cats, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    if informed_frac < 1.0:
        step = max(2, int(1.0 / (1.0 - informed_frac)))
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
# 1. Ordinary Kriging — value pins
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedOrdinaryKriging:
    """Value-level pins for Ordinary Kriging (sweep family lives in complete)."""

    def test_ok_tiny_grid_clamped_radius(self):
        """OK: (2,2,2) tiny grid with clamped search radius — no crash, valid output.

        The radius is clamped to the grid dimensions (radiuses = min(r, d)),
        the only tiny-grid radius-clamp path in the suite (complete's smallest
        grid is 5x5x3=75 cells). This exercises the clamp branch.
        """
        x, y, z = 2, 2, 2
        grid, prop = _make_cont_grid(x, y, z)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1)
        radiuses = tuple(min(r, d) for r, d in zip((5, 5, 5), (x, y, z)))
        max_n = min(12, x * y * z)

        result = ordinary_kriging(prop, grid, radiuses, max_n, cov_model)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        assert not np.all(result.data == 0), "All-zero regression on tiny grid"
        informed = result.mask == 1
        if np.any(informed):
            assert np.all(np.isfinite(result.data.astype("float64")[informed]))

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
# 2. Simple Kriging — mean-path pin
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedSimpleKriging:
    """Simple Kriging pins (sweep family lives in complete)."""

    def test_sk_auto_mean_vs_explicit_similar(self):
        """SK: auto-computed mean should be close to explicit 50 on centered data.

        The masked (uninformed) cells are set to a discriminating constant
        (100.0) far from the informed distribution (40-60), so a regression
        computing the C++ auto-mean over ALL cells (instead of informed-only)
        shifts the mean measurably and the comparison FAILS (N2-L01/L02).
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        rng = np.random.RandomState(42)
        data = rng.rand(500).astype("float32") * 20 + 40  # centered near 50
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        # Discriminating masked values: far from the informed distribution
        data = data.copy()
        data[mask == 0] = 100.0
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result_auto = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=None)
        result_explicit = simple_kriging(prop, grid, (5, 5, 3), 12, cov_model, mean=50.0)

        assert isinstance(result_auto, ContProperty)
        assert isinstance(result_explicit, ContProperty)
        # Both must be valid
        assert not np.any(np.isnan(result_auto.data.astype("float64")))
        assert not np.any(np.isnan(result_explicit.data.astype("float64")))
        # Auto mean (informed-only ≈ 50) must match explicit mean=50 closely.
        # N2-L01: atol=0.1 — iter-1's tighter tolerance FAILS on current
        # correct code (auto mean 50.0395, 0.04 off).
        np.testing.assert_allclose(
            result_auto.data.astype("float64"),
            result_explicit.data.astype("float64"),
            atol=0.1,
            err_msg="SK auto-mean should match explicit mean=50 on centered data",
        )


# =============================================================================
# 3. Median Indicator Kriging — covariance smoke
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedMedianIK:
    """Median IK (2-category) covariance smoke + range asserts."""

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
    ])
    def test_median_ik_cov_sweep(self, cov_name, cov_type):
        """M-IK: covariance smoke — indicator_count==2 and data < count."""
        grid, prop = _make_ind_grid(5, 5, 3, 2)
        cov_model = _make_cov(cov_type, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1)

        result = median_ik(prop, grid, (0.5, 0.5), (3, 3, 2), 8, cov_model)

        assert isinstance(result, IndProperty), f"Failed for {cov_name}"
        assert result.indicator_count == 2
        assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {cov_name}"
        assert not np.any(np.isinf(result.data.astype("float64"))), f"Inf in {cov_name}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range in {cov_name}"


# =============================================================================
# 4. Indicator Kriging — covariance smoke
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedIndicatorKriging:
    """Indicator Kriging (3-category) covariance smoke + range asserts."""

    @pytest.mark.parametrize("cov_name,cov_type", [
        ("spherical", covariance.spherical),
    ])
    def test_ik_cov_sweep(self, cov_name, cov_type):
        """IK: covariance smoke — indicator_count==3 and data < count."""
        grid, prop = _make_ind_grid(5, 5, 3, 3)
        ik_data = _make_ik_data(cov_type, 3, ranges=(3.0, 3.0, 2.0),
                                radiuses=(3, 3, 2), max_neighbours=8)
        marginal_probs = [0.3, 0.4, 0.3]

        result = indicator_kriging(prop, grid, ik_data, marginal_probs)

        assert isinstance(result, IndProperty), f"Failed for {cov_name}"
        assert result.indicator_count == 3, f"Wrong indicator count for {cov_name}"
        assert np.all(result.data < result.indicator_count), f"Out-of-range in {cov_name}"


# =============================================================================
# 5. Simple Kriging Weights — value pins
# =============================================================================


@pytest.mark.hpgl
class TestParametrizedKrigingWeights:
    """Simple Kriging Weights value-level pins."""

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
    ])
    def test_skw_cov_sweep(self, cov_name, cov_type, neighbor_points):
        """SKW: covariance smoke — valid weights, finite, correct count, float32 dtype."""
        n_x, n_y, n_z = neighbor_points
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=cov_type, nugget=0.1,
        )

        assert isinstance(weights, np.ndarray), f"Failed for {cov_name}"
        assert len(weights) == len(n_x), f"Wrong count for {cov_name}"
        # F-15: genuine dtype contract pin (float32) — not duplicated in
        # test_math_reference.py which asserts values only.
        assert weights.dtype == np.float32, f"Wrong dtype for {cov_name}"
        assert not np.any(np.isnan(weights)), f"NaN in {cov_name}"
        assert not np.any(np.isinf(weights)), f"Inf in {cov_name}"
        assert np.all(np.isfinite(weights)), f"Non-finite in {cov_name}"

    def test_skw_weights_with_nugget_dont_sum_to_one(self, neighbor_points):
        """SKW: weights with nugget>0 match a NumPy reference solve of the system.

        Simple kriging with nugget > 0 produces weights that do NOT sum to
        1.0 (the nugget term on the diagonal of the covariance matrix means
        the kriging system doesn't satisfy the unit-sum constraint of
        ordinary kriging). The old `abs(sum-1) > 0.001` assert passed for
        sum=0 or 2.0 (total solver breakage) — the reference solve below is
        discriminating (T-06/F-02).
        """
        n_x, n_y, n_z = neighbor_points
        center = (5.0, 5.0, 2.5)
        weights = simple_kriging_weights(
            center_point=center,
            n_x=n_x, n_y=n_y, n_z=n_z,
            ranges=(5.0, 5.0, 3.0), sill=1.0, cov_type=covariance.spherical,
            nugget=0.5,
        )

        # NumPy reference solve: A w = c with A[i][j] = C(h_ij) and
        # c[i] = C(h_i). C(h) uses the HPGL diagonal convention: C(0)=sill
        # (near-zero guard), C(h) = (sill - nugget) * shape(h_eff / rx) with
        # the zero-angle anisotropic effective distance
        # h_eff = ||(dx, rx/ry*dy, rx/rz*dz)||.
        rx, ry, rz = 5.0, 5.0, 3.0
        sill, nugget = 1.0, 0.5
        pts = np.stack([n_x, n_y, n_z], axis=1).astype("float64")
        cen = np.array(center, dtype="float64")

        def _sph_shape(x):
            return np.maximum(0.0, 1.0 - 1.5 * x + 0.5 * np.clip(x, 0.0, 1.0) ** 3)

        def _cov(h):
            if h < 1e-5 * rx:
                return sill
            if h >= rx:
                return 0.0
            return (sill - nugget) * _sph_shape(h / rx)

        def _h_eff(dx, dy, dz):
            return np.sqrt(dx * dx + (rx / ry * dy) ** 2 + (rx / rz * dz) ** 2)

        n = len(n_x)
        A = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = _cov(_h_eff(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1],
                                      pts[i, 2] - pts[j, 2]))
        c = np.empty(n)
        for i in range(n):
            c[i] = _cov(_h_eff(cen[0] - pts[i, 0], cen[1] - pts[i, 1], cen[2] - pts[i, 2]))
        w_ref = np.linalg.solve(A, c)

        np.testing.assert_allclose(
            weights.astype("float64"), w_ref, rtol=1e-4, atol=1e-4,
            err_msg="SKW weights should match the NumPy reference solve",
        )
        # The nugget>0 sum!=1 property (original intent, now secondary)
        weight_sum = float(np.sum(weights))
        assert abs(weight_sum - 1.0) > 0.001, (
            f"SK weights with nugget=0.5 should NOT sum to 1.0, got {weight_sum:.6f}"
        )

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
# 6. Property Construction Rejection Tests
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
# 7. Edge Cases — All Kriging Variants
# =============================================================================


@pytest.mark.hpgl
class TestKrigingEdgeCases:
    """Edge case tests across all kriging variants."""

    # ---- Empty / No-informed-data tests ----

    def test_ok_empty_no_informed(self):
        """OK: zero informed points — all cells remain uninformed, no crash.

        OK's undefined_on_failure contract leaves every cell mask==0 (the
        mean-fill fallback applies only to SK/LVM/cokriging) — assert the
        mask is preserved (F-16).
        """
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.zeros(50, dtype="uint8")  # All uninformed
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape
        # N2-L04/F-16: OK does NOT mean-fill — undefined cells keep mask 0.
        assert np.all(result.mask == 0), "OK with no informed cells should leave mask==0"

    def test_sk_empty_no_informed(self):
        """SK: zero informed points — mean-fill contract (mask==1, data==mean).

        SK's mean_on_failure path mean-fills every cell with the explicit
        mean and marks them informed (set_at mask=1) — assert the fill
        actually happened (N2-L04/D-73).
        """
        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.zeros(50, dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = _make_cov(covariance.spherical)

        result = simple_kriging(prop, grid, (3, 3, 2), 8, cov_model, mean=50.0)

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape
        # N2-L04: SK mean-fill sets mask=1 and data=50.0 on every cell
        assert np.all(result.mask == 1), "SK mean-fill should mark all cells informed"
        np.testing.assert_allclose(
            result.data.astype("float64"), 50.0, atol=1e-5,
            err_msg="SK mean-fill should equal the explicit mean",
        )

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
        """LVM: (1,1,1) grid — estimate copies through the informed value.

        A single informed cell at the only grid cell is copied through
        (cont_kriging.h copy-through branch) — estimate must equal the data
        value (D-77/V-12, mirrors the OK/SK single-point value asserts).
        """
        grid = SugarboxGrid(x=1, y=1, z=1)
        data = np.array([42.0], dtype="float32")
        mask = np.array([1], dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.array([50.0], dtype="float32")
        cov_model = _make_cov(covariance.spherical, ranges=(1.0, 1.0, 1.0), nugget=0.0)

        result = lvm_kriging(prop, grid, mean_data, (1, 1, 1), 1, cov_model)

        assert result.data.size == 1
        assert abs(float(result.data.flat[0]) - 42.0) < 0.1, (
            f"LVM single-point estimate {result.data.flat[0]} should equal 42.0"
        )

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
        """LVM: all same value, mean same value — result near uniform value.

        H-3: added the value assert the OK/SK uniform siblings already have.
        """
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
        assert np.allclose(result.data.astype("float64"), 42.0, atol=0.1)

    # ---- Sparsity tests ----

    def test_ok_sparse_partially_informed(self):
        """OK: partially informed grid (~50%) — no crash, valid output shape.

        The _make_cont_grid helper clamps the uninformed step to >= 2, so
        informed_frac < 0.5 cannot be produced and this fixture is ~50%
        informed, not the 10% the old name claimed (N2-L07).
        """
        grid, prop = _make_cont_grid(10, 10, 5, informed_frac=0.1)

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

    # ---- Sill edge case: near-zero but not quite zero ----

    def test_ok_small_sill(self):
        """OK: very small sill (0.01) but nugget=0 — valid output within input bounds.

        F-11: added the convexity-bound assert — informed-cell estimates must
        stay within [min, max] of the input (exact interpolation with zero
        nugget), catching the wild-solution solver-gate regression class.
        """
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical, ranges=(5.0, 5.0, 3.0),
                              sill=0.01, nugget=0.0)

        result = ordinary_kriging(prop, grid, (3, 3, 2), 8, cov_model)

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

        informed_input = prop.data[prop.mask == 1]
        data_min = float(np.min(informed_input))
        data_max = float(np.max(informed_input))
        informed_output = result.data[result.mask == 1]
        if len(informed_output) > 0:
            assert float(np.min(informed_output)) >= data_min - 1e-4, (
                "small-sill OK output below input minimum"
            )
            assert float(np.max(informed_output)) <= data_max + 1e-4, (
                "small-sill OK output above input maximum"
            )

    # ---- Negative parameter rejection tests ----

    def test_ok_max_neighbours_zero_rejected(self):
        """OK: max_neighbours=0 raises CriticalValidationError."""
        from geo_bsd.validation import CriticalValidationError
        grid, prop = _make_cont_grid(5, 5, 3)
        cov_model = _make_cov(covariance.spherical)
        with pytest.raises(CriticalValidationError):
            ordinary_kriging(prop, grid, (3, 3, 2), 0, cov_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
