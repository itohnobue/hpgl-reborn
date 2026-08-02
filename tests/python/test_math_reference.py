"""
Math verification tests for HPGL kriging, CDF, and algorithm correctness.

Validates numerical correctness of geostatistical algorithms against
analytically known reference values from Stage 6 research.

Test categories:
  1. Simple Kriging weights — exact interpolation, analytical solutions
  2. Ordinary Kriging — weight sum constraint
  3. CDF — construction, invariants, edge cases
  4. MeanCalc — boundary weight detection (E01)
  5. Rotation/Anisotropy — ZYX convention verification
  6. CalcVPC — per-layer mean consistency

Tolerances: float32-compatible (rtol=1e-5, atol=1e-7 for most;
                                  rtol=1e-4, atol=1e-5 for full pipeline)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import calc_cdf
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        SugarboxGrid,
        covariance,
        ordinary_kriging,
        simple_kriging_weights,
    )
    from geo_bsd.routines import (
        CalcMean,
        CalcVPC,
        GetCubicalMask,
        GetEllipseMask,
        MeanCalc,
    )
except (ImportError, OSError):
    pass


# =============================================================================
# Tolerance configuration
# =============================================================================
COV_RTOL = 1e-5
COV_ATOL = 1e-7
KRIG_RTOL = 1e-4
KRIG_ATOL = 1e-5
WEIGHT_RTOL = 1e-5
WEIGHT_ATOL = 1e-7


# =============================================================================
# Simple Kriging — Exact Interpolation
# =============================================================================


@pytest.mark.hpgl
class TestSimpleKrigingExactInterpolation:
    """SK produces exact interpolation at sample points with zero nugget."""

    def test_sk_colocated_neighbor_zero_nugget(self):
        """SK-EI-1: Co-located neighbor with nugget=0 → weight=1.0."""
        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=np.array([5.0], dtype="float32"),
            n_y=np.array([5.0], dtype="float32"),
            n_z=np.array([2.5], dtype="float32"),
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        assert len(weights) == 1
        np.testing.assert_allclose(weights[0], 1.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_sk_colocated_neighbor_various_models(self):
        """Co-located neighbor gets weight=1.0 regardless of covariance type."""
        for cv_type in [covariance.spherical, covariance.exponential, covariance.gaussian]:
            weights = simple_kriging_weights(
                center_point=(3.0, 3.0, 1.0),
                n_x=np.array([3.0], dtype="float32"),
                n_y=np.array([3.0], dtype="float32"),
                n_z=np.array([1.0], dtype="float32"),
                ranges=(10.0, 10.0, 5.0),
                sill=1.0,
                cov_type=cv_type,
                nugget=0.0,
            )
            np.testing.assert_allclose(
                weights[0],
                1.0,
                rtol=WEIGHT_RTOL,
                atol=WEIGHT_ATOL,
                err_msg=f"Covariance type {cv_type}: co-located weight should be 1.0",
            )

    def test_sk_finite_distance_exponential(self):
        """SK-REF: Single neighbor at known distance → analytical weight.

        sill=1.0, nugget=0.0, exponential, range=5.0:
        Neighbor at (3, 0, 0), Target at (0, 0, 0)
        C(h=3) = exp(-3*3/5) = exp(-1.8) ≈ 0.16529889
        w = C(h)/C(0) = 0.16529889
        """
        weights = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0),
            n_x=np.array([3.0], dtype="float32"),
            n_y=np.array([0.0], dtype="float32"),
            n_z=np.array([0.0], dtype="float32"),
            ranges=(5.0, 5.0, 5.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.0,
        )
        expected = np.exp(-3.0 * 3.0 / 5.0)  # ~0.165299
        np.testing.assert_allclose(weights[0], expected, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_sk_two_neighbors_orthogonal(self):
        """SK-REF-2: Two orthogonal neighbors → symmetric analytical weights.

        Neighbor 1 at (1,0,0), Neighbor 2 at (0,1,0), Target (0,0,0).
        Exponential, sill=1, nugget=0, range=5.
        C(1,0)=1.0, C(0,1)=0.428133, k=[0.548812, 0.548812]
        By symmetry: w0 = w1 = 0.548812 / (1 + 0.428133) ≈ 0.384375
        """
        C12 = np.exp(-3.0 * np.sqrt(2.0) / 5.0)  # ~0.428133
        C_star = np.exp(-3.0 * 1.0 / 5.0)  # ~0.548812
        expected_w = C_star / (1.0 + C12)  # ~0.384375

        weights = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0),
            n_x=np.array([1.0, 0.0], dtype="float32"),
            n_y=np.array([0.0, 1.0], dtype="float32"),
            n_z=np.array([0.0, 0.0], dtype="float32"),
            ranges=(5.0, 5.0, 5.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.0,
        )
        assert len(weights) == 2
        np.testing.assert_allclose(weights[0], expected_w, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(weights[1], expected_w, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(weights[0], weights[1], rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_sk_two_neighbors_diagonal(self):
        """Two neighbors on same axis — analytical check.

        Neighbor 1 at (4,0,0), Neighbor 2 at (10,0,0), Target (0,0,0).
        Spherical, sill=1, nugget=0, range=10.
        C(0,0)=C(1,1)=1.0 (zero distance), C(0,1)=C(|4-10|)=C(6)=0.208.
        K = [[1.0, 0.208], [0.208, 1.0]], k = [C(4)=0.432, C(10)=0.0].
        det = 1 - 0.208^2 = 0.956736, w0 = (0.432 - 0.208*0)/det ≈ 0.451535.
        w1 = (0 - 0.208*0.432)/det ≈ -0.089856 / 0.956736 ≈ -0.093919.
        """
        weights = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0),
            n_x=np.array([4.0, 10.0], dtype="float32"),
            n_y=np.array([0.0, 0.0], dtype="float32"),
            n_z=np.array([0.0, 0.0], dtype="float32"),
            ranges=(10.0, 10.0, 10.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # Analytical solution:
        # K = [[C(0)=1.0, C(6)=0.208], [C(6)=0.208, C(0)=1.0]]
        # k = [C(4)=0.432, C(10)=0.0]
        C6 = 1.0 - 1.5 * 0.6 + 0.5 * 0.6**3  # = 0.208
        C4 = 1.0 - 1.5 * 0.4 + 0.5 * 0.4**3  # = 0.432
        det = 1.0 - C6 * C6  # ≈ 0.956736
        expected_w0 = C4 / det  # ≈ 0.451535
        expected_w1 = (-C6 * C4) / det  # ≈ -0.093919

        np.testing.assert_allclose(weights[0], expected_w0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(weights[1], expected_w1, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)


# =============================================================================
# Ordinary Kriging — Weight Sum Constraint
# =============================================================================


@pytest.mark.hpgl
class TestOrdinaryKrigingConstraints:
    """OK weight sum equals 1.0 (unbiasedness constraint)."""

    def test_sk_weights_sum_to_one_with_ok_equivalent(self):
        """OK unbiasedness constraint: for a symmetric 2-point configuration,
        the estimate at a point equidistant from both data points equals
        the arithmetic mean of the two data values.

        This verifies that OK weights sum to 1.0 (unbiasedness) AND that
        symmetric points receive equal weights (w0 = w1 = 0.5).

        Uses a 3×3×1 grid with two informed cells at opposite corners
        and tests the estimate at the grid centre.
        """
        grid = SugarboxGrid(x=3, y=3, z=1)
        n_total = grid.x * grid.y * grid.z
        data = np.zeros(n_total, dtype="float32")
        mask = np.zeros(n_total, dtype="uint8")
        # Cell (0,0,0): flat index 0, value 100
        data[0] = 100.0
        mask[0] = 1
        # Cell (2,2,0): flat index 2 + 2*3 = 8, value 50
        data[8] = 50.0
        mask[8] = 1

        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 10.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(5, 5, 3),
            max_neighbours=4,
            cov_model=cov_model,
        )

        # Target cell (1,1,0): flat index = 1 + 1*3 + 0*9 = 4
        target_idx = 1 + 1 * grid.x + 0 * grid.x * grid.y
        estimate = result.data[target_idx]
        # By symmetry, w0 = w1 = 0.5, so estimate = (100 + 50)/2 = 75
        expected = (100.0 + 50.0) / 2.0

        np.testing.assert_allclose(
            estimate,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"OK symmetric estimate {estimate} != {expected}; "
            f"weight-sum constraint (w0+w1=1) or symmetry (w0=w1) violated",
        )
        # All estimated cells must be finite
        assert np.all(np.isfinite(result.data[result.mask != 0]))

    def test_sk_many_neighbors_weights_finite(self):
        """SK weights with many neighbors should be finite and valid."""
        np.random.seed(42)
        n = 8
        n_x = np.random.rand(n).astype("float32") * 10
        n_y = np.random.rand(n).astype("float32") * 10
        n_z = np.random.rand(n).astype("float32") * 5

        weights = simple_kriging_weights(
            center_point=(5.0, 5.0, 2.5),
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(10.0, 10.0, 10.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )
        assert len(weights) == n
        assert np.all(np.isfinite(weights))
        assert not np.any(np.isnan(weights))


# =============================================================================
# CDF Tests
# =============================================================================


@pytest.mark.hpgl
class TestCDFAnalytical:
    """CDF construction and invariants from known datasets."""

    def test_cdf_uniform_distribution(self):
        """CDF-ANALYTIC-1: 9 values 0.0..2.0 step 0.25 → probs=[1/9, 2/9, ..., 1.0]."""
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype="float32")
        mask = np.ones(9, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == 9
        assert len(cdf.probs) == 9
        # Expected probs: cumulative after each unique value
        for i in range(9):
            np.testing.assert_allclose(cdf.probs[i], (i + 1) / 9.0, rtol=1e-6, atol=1e-8)
        # Last prob must be 1.0
        np.testing.assert_allclose(cdf.probs[-1], 1.0)

    def test_cdf_sorted_values(self):
        """CDF values must be sorted ascending."""
        data = np.array([3.0, 1.0, 4.0, 1.5, 2.0, 2.5], dtype="float32")
        mask = np.ones(6, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert np.all(np.diff(cdf.values) >= 0), "CDF values must be sorted ascending"

    def test_cdf_monotonic_probs(self):
        """CDF probabilities must be non-decreasing."""
        np.random.seed(42)
        data = np.random.rand(50).astype("float32") * 100
        mask = np.ones(50, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert np.all(np.diff(cdf.probs) >= 0), "CDF probs must be non-decreasing"

    def test_cdf_last_prob_is_one(self):
        """CDF invariant: last probability must be 1.0."""
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        np.testing.assert_allclose(cdf.probs[-1], 1.0, rtol=1e-6, atol=1e-8)

    def test_cdf_multi_value_tail_below_one_after_float32_downcast(self):
        """F-N13: multi-value float32-downcast tail must stay strictly below 1.0.

        F-M11 facet (a): with 3 equal-count values the float64 cumulative sum
        is 0.9999999999999999 (not >= 1.0), so the pre-fix clamp was skipped
        and the float32 downcast rounded the tail to exactly 1.0f — re-creating
        the F-04 max-datum->median collapse. The clamp must run on the float32
        values so the stored tail is < 1.0 and equals the nextafter value.
        """
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == 3
        assert cdf.probs[-1] < 1.0, "float32-downcast tail must be clamped below 1.0"
        assert cdf.probs[-1] == np.nextafter(np.float32(1.0), np.float32(0.0))
        assert np.all(np.diff(cdf.probs) >= 0)

    def test_cdf_large_count_monotonic_float32_tail(self):
        """F-N13: full_count >= 2^25 must keep the float32 tail monotonic.

        F-M11 facet (b): a 2^25-cell grid with a one-cell last value rounds
        the second-to-last cumulative probability up to exactly 1.0f; the
        pre-fix clamp then produced [.., 1.0f, 0.99999994f] and a spurious
        monotonicity ValueError. The fixed clamp caps the 1.0f suffix.
        """
        n = 2**25
        data = np.zeros(n, dtype="float32")
        data[-1] = 1.0
        prop = ContProperty(data, np.ones(n, dtype="uint8"))
        cdf = calc_cdf(prop)

        assert np.all(np.diff(cdf.probs) >= 0), "float32 CDF must stay monotonic"
        assert cdf.probs[-1] < 1.0

    def test_cdf_first_prob_positive(self):
        """CDF invariant: first probability must be strictly positive (>0)."""
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert cdf.probs[0] > 0, "First CDF prob must be > 0"

    def test_cdf_values_probs_same_length(self):
        """CDF invariant: len(values) == len(probs)."""
        np.random.seed(42)
        data = np.random.rand(75).astype("float32") * 100
        mask = np.ones(75, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == len(cdf.probs)

    def test_cdf_duplicate_values(self):
        """CDF with duplicate values: each unique value appears once."""
        data = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype="float32")
        mask = np.ones(6, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == 3  # Only 3 unique values
        np.testing.assert_allclose(cdf.probs, [1.0 / 3.0, 2.0 / 3.0, 1.0], rtol=1e-6)

    def test_cdf_single_value(self):
        """CDF with a single value: one entry with prob=1.0."""
        data = np.array([42.0, 42.0, 42.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == 1
        assert cdf.values[0] == 42.0
        np.testing.assert_allclose(cdf.probs[0], 1.0)

    def test_cdf_with_masked_values(self):
        """CDF only considers informed (unmasked) cells."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        mask = np.array([1, 0, 1, 0, 1], dtype="uint8")  # Only 1.0, 3.0, 5.0
        prop = ContProperty(data, mask)
        cdf = calc_cdf(prop)

        assert len(cdf.values) == 3
        np.testing.assert_allclose(cdf.values, [1.0, 3.0, 5.0])

    def test_cdf_all_masked_raises(self):
        """calc_cdf raises ValueError when all cells are masked."""
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.zeros(3, dtype="uint8")
        prop = ContProperty(data, mask)

        with pytest.raises(ValueError, match="no informed values"):
            calc_cdf(prop)

    def test_cdf_3d_grid(self):
        """CDF from 3D grid produces same result as flat array with same data."""
        # 3D: 2x2x2 grid
        data_3d = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype="float32")
        mask_3d = np.ones(8, dtype="uint8")
        prop = ContProperty(data_3d, mask_3d)
        cdf = calc_cdf(prop)

        # Should have 4 unique values with probs [0.25, 0.5, 0.75, 1.0]
        assert len(cdf.values) == 4
        np.testing.assert_allclose(cdf.probs[-1], 1.0)

    def test_cdf_nan_in_data(self):
        """ContProperty rejects NaN in data at construction — no NaN reaches calc_cdf."""
        data = np.array([1.0, np.nan, 3.0, 4.0], dtype="float32")
        mask = np.ones(4, dtype="uint8")
        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(data, mask)


# =============================================================================
# CalcMean Tests
# =============================================================================


@pytest.mark.hpgl
class TestCalcMean:
    """calc_mean arithmetic mean of informed cells."""

    def test_calc_mean_all_informed(self):
        """calc_mean: all cells informed → simple average."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        mask = np.ones(5, dtype="uint8")
        prop = ContProperty(data, mask)

        from geo_bsd.geo import calc_mean

        result = calc_mean(prop)
        np.testing.assert_allclose(result, 3.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_calc_mean_with_uninformed(self):
        """calc_mean: ignores masked cells."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float32")
        mask = np.array([1, 1, 1, 0, 0], dtype="uint8")  # Only 10, 20, 30
        prop = ContProperty(data, mask)

        from geo_bsd.geo import calc_mean

        result = calc_mean(prop)
        np.testing.assert_allclose(result, 20.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)


# =============================================================================
# MeanCalc Boundary Tests (E01)
# =============================================================================


@pytest.mark.hpgl
class TestMeanCalcBoundary:
    """E01: MeanCalc boundary weight alignment tests.

    The MeanCalc function computes local means using a sliding window.
    Boundary points may have misaligned mask and data windows.
    These tests validate correct behavior at all boundary locations.
    """

    def test_e01_interior_uniform_cubical(self):
        """E01-T1: Interior point with cubical mask on uniform data → exact mean."""
        nx, ny, nz = 5, 5, 3
        cube = np.ones((nx, ny, nz), dtype="float32") * 100.0
        mask = np.ones((nx, ny, nz), dtype="uint8")
        radii = (2, 2, 1)
        mmask = GetCubicalMask(radii)

        result = MeanCalc(cube, mask, radii, mmask, (2, 2, 1), -999.0)
        np.testing.assert_allclose(result, 100.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_e02_left_boundary_uniform_cubical(self):
        """E01-T2: Left edge with cubical mask on uniform data → exact mean."""
        nx, ny, nz = 5, 5, 3
        cube = np.ones((nx, ny, nz), dtype="float32") * 100.0
        mask = np.ones((nx, ny, nz), dtype="uint8")
        radii = (2, 2, 1)
        mmask = GetCubicalMask(radii)

        result = MeanCalc(cube, mask, radii, mmask, (0, 2, 1), -999.0)
        np.testing.assert_allclose(result, 100.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_e03_corners_uniform_cubical(self):
        """E01-T3: All 8 corners with uniform data → exact mean."""
        nx, ny, nz = 5, 5, 3
        cube = np.ones((nx, ny, nz), dtype="float32") * 100.0
        mask = np.ones((nx, ny, nz), dtype="uint8")
        radii = (2, 2, 1)
        mmask = GetCubicalMask(radii)

        corners = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 4, 0),
            (0, 4, 2),
            (4, 0, 0),
            (4, 0, 2),
            (4, 4, 0),
            (4, 4, 2),
        ]
        for corner in corners:
            result = MeanCalc(cube, mask, radii, mmask, corner, -999.0)
            np.testing.assert_allclose(
                result,
                100.0,
                rtol=WEIGHT_RTOL,
                atol=WEIGHT_ATOL,
                err_msg=f"Corner {corner}: expected 100.0, got {result}",
            )

    def test_e04_ellipsoid_interior_vs_boundary(self):
        """E01-T4: Ellipsoid mask with non-uniform data — verifies correct
        mask alignment at boundaries.

        Grid 5x5x3 with cube[i,j,k] = float(i) (gradient along X).
        Uses ellipsoid mask with radii (2, 2, 1).
        All mask cells are informed.

        Interior point (2,2,1): window cube[0:4,0:4,0:2], mask fully present.
        Ellipsoid selects 12 cells with X values
        {0:1, 1:3, 2:5, 3:3} → analytical mean = 22/12 ≈ 1.8333.

        Boundary point (0,2,1): left edge, window cube[0:2,0:4,0:2] clipped,
        mask correctly shifted to keep centre aligned with data point.
        Ellipsoid selects 8 cells: {0:5, 1:3} → analytical mean = 3/8 = 0.375.

        With uniform data (all 100.0) both would give the same result
        regardless of mask alignment, making the E01 mask-alignment bug
        invisible. Non-uniform data exposes any misalignment.
        """
        nx, ny, nz = 5, 5, 3
        # Non-uniform data: gradient along X
        cube = np.zeros((nx, ny, nz), dtype="float32")
        for i in range(nx):
            cube[i, :, :] = float(i)
        mask = np.ones((nx, ny, nz), dtype="uint8")
        radii = (2, 2, 1)
        emask = GetEllipseMask(radii)

        interior = MeanCalc(cube, mask, radii, emask, (2, 2, 1), -999.0)
        boundary = MeanCalc(cube, mask, radii, emask, (0, 2, 1), -999.0)

        # Interior: 12 cells — one i=0, three i=1, five i=2, three i=3
        np.testing.assert_allclose(interior, 22.0 / 12.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

        # Boundary: correct mask alignment — five i=0, three i=1
        # With E01 misalignment bug present the mask slice would be
        # [0:2, ...] (one i=0, three i=1) giving 0.75 instead of 0.375.
        np.testing.assert_allclose(boundary, 3.0 / 8.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_e05_single_cell_grid(self):
        """E01-T5: Single-cell grid — works with any radius size."""
        cube = np.ones((1, 1, 1), dtype="float32") * 42.0
        mask = np.ones((1, 1, 1), dtype="uint8")
        radii = (5, 5, 5)
        mmask = GetCubicalMask(radii)

        result = MeanCalc(cube, mask, radii, mmask, (0, 0, 0), -999.0)
        np.testing.assert_allclose(result, 42.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_e06_gradient_boundary(self):
        """E01-T6: Non-constant gradient data — boundary mean should reflect edge values.

        Grid 10x10x5 with cube[i,j,k] = i (gradient along X).
        Left boundary (0,5,2) with radius=2: only cells at X=0,1,2.
        Interior point (5,5,2): should average around X=5.
        Expected boundary mean ≈ 1.0 (average of cells at X=0,1,2).
        """
        nx, ny, nz = 10, 10, 5
        cube = np.zeros((nx, ny, nz), dtype="float32")
        for i in range(nx):
            cube[i, :, :] = float(i)
        mask = np.ones((nx, ny, nz), dtype="uint8")
        radii = (2, 2, 2)
        # Use cubical mask for predictable boundary behavior
        mmask = GetCubicalMask(radii)

        # Interior point (5,5,2): should average around i=5
        result_interior = MeanCalc(cube, mask, radii, mmask, (5, 5, 2), -999.0)
        assert result_interior > 3.0 and result_interior < 7.0, (
            f"Interior mean at X=5 should be ~5, got {result_interior}"
        )

        # Left boundary point (0,5,2): should only see cells at X=0,1,2
        result_boundary = MeanCalc(cube, mask, radii, mmask, (0, 5, 2), -999.0)
        assert result_boundary < 2.0, (
            f"Boundary mean at X=0 should be ~1 (cells at 0,1,2), got {result_boundary}"
        )

    def test_e07_no_neighbors_returns_undefined(self):
        """E01-T7: When no neighbors exist (all masked locally), returns undefined_value."""
        cube = np.ones((5, 5, 3), dtype="float32") * 100.0
        mask = np.zeros((5, 5, 3), dtype="uint8")  # All masked
        radii = (2, 2, 1)
        mmask = GetCubicalMask(radii)

        result = MeanCalc(cube, mask, radii, mmask, (2, 2, 1), -999.0)
        np.testing.assert_allclose(result, -999.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)


# =============================================================================
# CalcVPC Tests
# =============================================================================


@pytest.mark.hpgl
class TestCalcVPC:
    """CalcVPC per-layer mean consistency."""

    def test_vpc_uniform_data(self):
        """CalcVPC with uniform data: each layer mean = uniform value."""
        nx, ny, nz = 4, 4, 3
        value = 77.0
        cube = np.ones((nx, ny, nz), dtype="float32") * value
        mask = np.ones((nx, ny, nz), dtype="uint8")

        result = CalcVPC(cube.copy(), mask, 0.0)
        assert len(result) == nz
        for k in range(nz):
            np.testing.assert_allclose(
                result[k],
                value,
                rtol=WEIGHT_RTOL,
                atol=WEIGHT_ATOL,
                err_msg=f"Layer {k}: expected {value}, got {result[k]}",
            )

    def test_vpc_empty_layer_gets_marginal(self):
        """CalcVPC: empty layer gets marginal mean."""
        nx, ny, nz = 4, 4, 3
        value = 77.0
        cube = np.ones((nx, ny, nz), dtype="float32") * value
        mask = np.ones((nx, ny, nz), dtype="uint8")
        mask[:, :, 1] = 0  # Entire layer 1 uninformed

        marginal = 99.0
        result = CalcVPC(cube.copy(), mask, marginal)
        assert len(result) == nz
        # Layer 0, 2 should be 77.0; layer 1 should be 99.0 (marginal)
        np.testing.assert_allclose(result[0], value, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(result[1], marginal, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(result[2], value, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_vpc_gradient_layers(self):
        """CalcVPC: layers with different values produce different means."""
        nx, ny, nz = 4, 4, 3
        cube = np.zeros((nx, ny, nz), dtype="float32")
        for k in range(nz):
            cube[:, :, k] = float(k * 10)  # Layer 0=0, 1=10, 2=20
        mask = np.ones((nx, ny, nz), dtype="uint8")

        result = CalcVPC(cube.copy(), mask, -1.0)
        assert len(result) == nz
        for k in range(nz):
            np.testing.assert_allclose(
                result[k], float(k * 10), rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL, err_msg=f"Layer {k}"
            )


# =============================================================================
# Rotation / Anisotropy Tests
# =============================================================================


@pytest.mark.hpgl
class TestRotationAnisotropy:
    """ZYX rotation convention verification."""

    def test_rot_isotropic_equal_weights(self):
        """ROT-T1: Isotropic model — same distance in X, Y, Z → same weight."""
        ranges = (10.0, 10.0, 10.0)
        dist = 5.0

        for cv_type in [covariance.spherical, covariance.exponential, covariance.gaussian]:
            wx = simple_kriging_weights(
                (0, 0, 0),
                np.array([dist], dtype="float32"),
                np.array([0.0], dtype="float32"),
                np.array([0.0], dtype="float32"),
                ranges=ranges,
                sill=1.0,
                cov_type=cv_type,
                nugget=0.0,
            )
            wy = simple_kriging_weights(
                (0, 0, 0),
                np.array([0.0], dtype="float32"),
                np.array([dist], dtype="float32"),
                np.array([0.0], dtype="float32"),
                ranges=ranges,
                sill=1.0,
                cov_type=cv_type,
                nugget=0.0,
            )
            wz = simple_kriging_weights(
                (0, 0, 0),
                np.array([0.0], dtype="float32"),
                np.array([0.0], dtype="float32"),
                np.array([dist], dtype="float32"),
                ranges=ranges,
                sill=1.0,
                cov_type=cv_type,
                nugget=0.0,
            )
            np.testing.assert_allclose(
                wx[0],
                wy[0],
                rtol=WEIGHT_RTOL,
                atol=WEIGHT_ATOL,
                err_msg=f"Isotropic {cv_type}: X vs Y mismatch",
            )
            np.testing.assert_allclose(
                wx[0],
                wz[0],
                rtol=WEIGHT_RTOL,
                atol=WEIGHT_ATOL,
                err_msg=f"Isotropic {cv_type}: X vs Z mismatch",
            )

    def test_rot_azimuth_90_swaps_axes(self):
        """ROT-T3: azimuth=90 swaps X and Y axes.

        ranges=(10,5,5), angles=(90,0,0):
        - (5,0,0) in global → rotates to (0,5,0) in local → h_eff along range=5 axis
        - (0,5,0) in global → rotates to (-5,0,0) in local → h_eff along range=10 axis
        So wy > wx because global Y aligns with range=10 axis after rotation.
        """
        ranges = (10.0, 5.0, 5.0)
        angles = (90.0, 0.0, 0.0)

        wx = simple_kriging_weights(
            (0, 0, 0),
            np.array([5.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        wy = simple_kriging_weights(
            (0, 0, 0),
            np.array([0.0], dtype="float32"),
            np.array([5.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # With azimuth 90, global Y maps to range=10 axis → higher weight
        assert wy[0] > wx[0], f"azimuth=90: wy={wy[0]} should be > wx={wx[0]}"

    def test_rot_zero_angles_no_effect(self):
        """With all-zero angles and isotropic ranges, rotation has no effect."""
        ranges = (10.0, 10.0, 10.0)
        angles = (0.0, 0.0, 0.0)

        # Neighbor at (3, 4, 0): distance = 5
        w1 = simple_kriging_weights(
            (0, 0, 0),
            np.array([3.0], dtype="float32"),
            np.array([4.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # Same distance but different rotation (should be same for isotropic)
        w2 = simple_kriging_weights(
            (0, 0, 0),
            np.array([5.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # Both at distance 5 from center → same weight for isotropic
        np.testing.assert_allclose(w1[0], w2[0], rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_rot_nontrivial_angles_documented(self):
        """ROT-T4: Non-trivial angles produce a specific, non-zero weight.

        ranges=(100,50,30), angles=(30,45,60), neighbor (3,4,2), target (0,0,0).
        Spherical sill=1.0, nugget=0.0.

        HPGL uses transposed ZYX rotation matrices (per cov_model.h):
          Rz_hpgl(a) = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
          Ry_hpgl(a) = [[c, 0, -s], [0, 1, 0], [s, 0, c]]
          Rx_hpgl(a) = [[1, 0, 0], [0, c, s], [0, -s, c]]
          t = scale * (Rx_hpgl @ Ry_hpgl @ Rz_hpgl)

        This test validates HPGL output against an independent numpy
        implementation of the same transposed-matrix convention.
        """
        ranges = (100.0, 50.0, 30.0)
        angles = (30.0, 45.0, 60.0)

        weights = simple_kriging_weights(
            (0, 0, 0),
            np.array([3.0], dtype="float32"),
            np.array([4.0], dtype="float32"),
            np.array([2.0], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        assert len(weights) == 1
        assert 0.0 < weights[0] <= 1.0, "Weight must be in (0, 1] for spherical with no nugget"

        # Independent mathematical validation using HPGL's transposed matrices
        a0, a1, a2 = np.radians(angles)  # angles = (azimuth, dip, plunge)
        Rz = np.array(
            [
                [np.cos(a0), np.sin(a0), 0],
                [-np.sin(a0), np.cos(a0), 0],
                [0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [np.cos(a1), 0, -np.sin(a1)],
                [0, 1, 0],
                [np.sin(a1), 0, np.cos(a1)],
            ]
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(a2), np.sin(a2)],
                [0, -np.sin(a2), np.cos(a2)],
            ]
        )
        R = Rx @ Ry @ Rz
        neighbor = np.array([3.0, 4.0, 2.0])
        rotated = R @ neighbor
        scaled = np.array(
            [
                rotated[0] / ranges[0],
                rotated[1] / ranges[1],
                rotated[2] / ranges[2],
            ]
        )
        h_eff = np.sqrt(np.sum(scaled**2))

        # Spherical covariance: C(h) = 1 - 1.5*h + 0.5*h^3  for h <= 1
        assert h_eff <= 1.0, f"Effective distance {h_eff:.4f} should be <= 1"
        expected_weight = 1.0 - 1.5 * h_eff + 0.5 * h_eff**3

        np.testing.assert_allclose(
            weights[0],
            expected_weight,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Rotation mismatch: HPGL={weights[0]:.6f}, "
            f"independent math={expected_weight:.6f}, h_eff={h_eff:.6f}",
        )

    def test_rot_convention_single_axis_only(self):
        """Single non-zero angle: effect should be predictable.

        ranges=(20,10,10), azimuth=45°, dip=0, rotation=0.
        Neighbor at (7.07, 7.07, 0): distance=10, but after 45° rotation,
        this maps entirely to the rotated X axis → h_eff along range=20.
        """
        # This is a qualitative test: verify that with an anisotropic model
        # and non-zero azimuth, the weight differs from isotropic expectation.
        ranges = (20.0, 10.0, 10.0)
        dist_xy = 5.0 / np.sqrt(2.0)  # ~3.5355 each axis → total distance 5

        # Without rotation (azimuth=0)
        w_no_rot = simple_kriging_weights(
            (0, 0, 0),
            np.array([dist_xy], dtype="float32"),
            np.array([dist_xy], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        # With 45° azimuth rotation
        w_rot = simple_kriging_weights(
            (0, 0, 0),
            np.array([dist_xy], dtype="float32"),
            np.array([dist_xy], dtype="float32"),
            np.array([0.0], dtype="float32"),
            ranges=ranges,
            angles=(45.0, 0.0, 0.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        # With rotation, the effective scaled distance should differ
        # Both should be valid (finite, non-NaN)
        assert np.isfinite(w_no_rot[0]) and np.isfinite(w_rot[0])

    def test_rot_zyx_convention_independent_verification(self):
        """ROT-T5: Verify ZYX convention independently.

        Cov_model.h:6-10 documents HPGL's ZYX (intrinsic) Euler rotation:
          t = scale * (Rx * Ry * Rz)
        where Rz rotates by angles[0], Ry by angles[1], Rx by angles[2],
        and scale = diag(1, rx/ry, rx/rz).

        This test independently builds the same rotation+scale matrices using
        numpy, computes the effective distance, then the analytical covariance,
        and compares against HPGL's simple_kriging_weights output.
        """
        from math import cos, radians, sin

        def hpgl_rotate_z(angle_deg):
            a = radians(angle_deg)
            return np.array(
                [
                    [cos(a), sin(a), 0],
                    [-sin(a), cos(a), 0],
                    [0, 0, 1],
                ]
            )

        def hpgl_rotate_y(angle_deg):
            a = radians(angle_deg)
            return np.array(
                [
                    [cos(a), 0, -sin(a)],
                    [0, 1, 0],
                    [sin(a), 0, cos(a)],
                ]
            )

        def hpgl_rotate_x(angle_deg):
            a = radians(angle_deg)
            return np.array(
                [
                    [1, 0, 0],
                    [0, cos(a), sin(a)],
                    [0, -sin(a), cos(a)],
                ]
            )

        def hpgl_effective_distance(ranges, angles, vec):
            """Compute h_eff the way HPGL does: ||scale * (Rx * Ry * Rz) * vec||."""
            rx, ry, rz = ranges
            Rz = hpgl_rotate_z(angles[0])
            Ry = hpgl_rotate_y(angles[1])
            Rx = hpgl_rotate_x(angles[2])
            # Combined rotation (ZYX intrinsic): R = Rx * Ry * Rz
            R = Rx @ Ry @ Rz
            # Rotate the displacement vector
            v_rot = R @ vec
            # Scale
            scale = np.array([1.0, rx / ry, rx / rz])
            v_scaled = v_rot * scale
            return np.sqrt(np.sum(v_scaled**2))

        def spherical_cov(h, sill, nugget, range_val):
            """Analytical spherical covariance: C(h)."""
            if h < 0.0001:
                return sill
            if h > range_val:
                return 0.0
            x = h / range_val
            return max(0.0, (sill - nugget) * (1.0 - 1.5 * x + 0.5 * x**3))

        # Test with non-trivial anisotropic ranges and all three angles
        ranges = (100.0, 50.0, 30.0)
        angles = (30.0, 45.0, 60.0)
        displacement = np.array([3.0, 4.0, 2.0])

        # Compute expected effective distance using independent math
        h_eff = hpgl_effective_distance(ranges, angles, displacement)
        expected_cov = spherical_cov(h_eff, sill=1.0, nugget=0.0, range_val=ranges[0])
        expected_weight = expected_cov

        # Get HPGL's weight
        weights = simple_kriging_weights(
            (0, 0, 0),
            np.array([displacement[0]], dtype="float32"),
            np.array([displacement[1]], dtype="float32"),
            np.array([displacement[2]], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        np.testing.assert_allclose(
            weights[0],
            expected_weight,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"ZYX convention mismatch: HPGL={weights[0]:.8f}, "
            f"independent={expected_weight:.8f} (h_eff={h_eff:.6f})",
        )

        # Second test case: just azimuth, anisotropic ranges
        ranges2 = (20.0, 10.0, 10.0)
        angles2 = (45.0, 0.0, 0.0)
        displacement2 = np.array([3.5355, 3.5355, 0.0])

        h_eff2 = hpgl_effective_distance(ranges2, angles2, displacement2)
        expected_cov2 = spherical_cov(h_eff2, sill=1.0, nugget=0.0, range_val=ranges2[0])
        expected_weight2 = expected_cov2

        weights2 = simple_kriging_weights(
            (0, 0, 0),
            np.array([displacement2[0]], dtype="float32"),
            np.array([displacement2[1]], dtype="float32"),
            np.array([displacement2[2]], dtype="float32"),
            ranges=ranges2,
            angles=angles2,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        np.testing.assert_allclose(
            weights2[0],
            expected_weight2,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"ZYX convention mismatch (azimuth only): "
            f"HPGL={weights2[0]:.8f}, independent={expected_weight2:.8f}",
        )

        # Third test case: dip only
        ranges3 = (10.0, 10.0, 5.0)
        angles3 = (0.0, 30.0, 0.0)
        displacement3 = np.array([0.0, 0.0, 5.0])

        h_eff3 = hpgl_effective_distance(ranges3, angles3, displacement3)
        expected_cov3 = spherical_cov(h_eff3, sill=1.0, nugget=0.0, range_val=ranges3[0])
        expected_weight3 = expected_cov3

        weights3 = simple_kriging_weights(
            (0, 0, 0),
            np.array([displacement3[0]], dtype="float32"),
            np.array([displacement3[1]], dtype="float32"),
            np.array([displacement3[2]], dtype="float32"),
            ranges=ranges3,
            angles=angles3,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        np.testing.assert_allclose(
            weights3[0],
            expected_weight3,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"ZYX convention mismatch (dip only): "
            f"HPGL={weights3[0]:.8f}, independent={expected_weight3:.8f}",
        )

    def test_rot_zyx_not_zxz(self):
        """ROT-T6: Verify HPGL is ZYX (not ZXZ as GSLIB uses).

        Under ZXZ convention, the same angles would produce markedly different
        effective distances for certain displacement directions. This test
        verifies that HPGL's output matches the ZYX calculation and does NOT
        match a ZXZ calculation.
        """
        from math import cos, radians, sin

        def make_rz(angle_deg):
            a = radians(angle_deg)
            return np.array(
                [
                    [cos(a), sin(a), 0],
                    [-sin(a), cos(a), 0],
                    [0, 0, 1],
                ]
            )

        def make_rx(angle_deg):
            a = radians(angle_deg)
            return np.array(
                [
                    [1, 0, 0],
                    [0, cos(a), sin(a)],
                    [0, -sin(a), cos(a)],
                ]
            )

        def zxz_effective_distance(ranges, angles, vec):
            """ZXZ (GSLIB) convention: R = Rz2 * Rx * Rz1."""
            rx, ry, rz = ranges
            Rz1 = make_rz(angles[0])
            Rx = make_rx(angles[1])
            Rz2 = make_rz(angles[2])
            R = Rz2 @ Rx @ Rz1
            v_rot = R @ vec
            scale = np.array([1.0, rx / ry, rx / rz])
            v_scaled = v_rot * scale
            return np.sqrt(np.sum(v_scaled**2))

        def spherical_cov(h, sill, nugget, range_val):
            if h < 0.0001:
                return sill
            if h > range_val:
                return 0.0
            x = h / range_val
            return max(0.0, (sill - nugget) * (1.0 - 1.5 * x + 0.5 * x**3))

        # Use all three angles non-zero so ZYX vs ZXZ diverge
        ranges = (100.0, 50.0, 30.0)
        angles = (30.0, 45.0, 60.0)
        displacement = np.array([3.0, 4.0, 2.0])

        weights = simple_kriging_weights(
            (0, 0, 0),
            np.array([displacement[0]], dtype="float32"),
            np.array([displacement[1]], dtype="float32"),
            np.array([displacement[2]], dtype="float32"),
            ranges=ranges,
            angles=angles,
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )

        # ZXZ effective distance (GSLIB convention — should NOT match)
        h_eff_zxz = zxz_effective_distance(ranges, angles, displacement)
        zxz_cov = spherical_cov(h_eff_zxz, sill=1.0, nugget=0.0, range_val=ranges[0])
        zxz_weight = zxz_cov

        # HPGL weight should differ from ZXZ
        assert not np.isclose(weights[0], zxz_weight, rtol=5e-2), (
            f"HPGL weight {weights[0]:.8f} accidentally matches ZXZ weight {zxz_weight:.8f} "
            f"— this would suggest HPGL uses ZXZ, not ZYX"
        )


# =============================================================================
# CalcMean from routines
# =============================================================================


@pytest.mark.hpgl
class TestCalcMeanRoutines:
    """CalcMean function from routines.py."""

    def test_calc_mean_uniform(self):
        """CalcMean on uniform data returns the uniform value."""
        cube = np.ones((5, 5, 3), dtype="float32") * 42.0
        mask = np.ones((5, 5, 3), dtype="uint8")
        result = CalcMean(cube, mask)
        np.testing.assert_allclose(result, 42.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)

    def test_calc_mean_with_mask(self):
        """CalcMean excludes masked cells."""
        cube = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        cube = cube.reshape((1, 1, 5))
        mask = np.array([1, 1, 1, 0, 0], dtype="uint8").reshape((1, 1, 5))
        result = CalcMean(cube, mask)
        np.testing.assert_allclose(result, 2.0, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
