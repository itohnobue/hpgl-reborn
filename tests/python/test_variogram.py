import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.variogram import (
        CubeScan,
        PointSetScanGridStyle,
        TVEllipsoid,
        TVVariogramSearchTemplate,
        _CalcLagDistances,
        _CalcSearchTemplateWindow,
        _IsInTunnel,
    )

    VARIOM_AVAILABLE = True
except ImportError:
    VARIOM_AVAILABLE = False


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestTVEllipsoid:
    def test_default_angles_identity(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        np.testing.assert_array_almost_equal(ell.Direction1, [1, 0, 0], decimal=10)
        np.testing.assert_array_almost_equal(ell.Direction2, [0, 1, 0], decimal=10)
        np.testing.assert_array_almost_equal(ell.Direction3, [0, 0, 1], decimal=10)

    def test_radii_stored(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        assert ell.R1 == 10
        assert ell.R2 == 5
        assert ell.R3 == 3

    def test_nonzero_azimuth(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=90)
        d1 = np.array(ell.Direction1)
        d2 = np.array(ell.Direction2)
        np.testing.assert_array_almost_equal(d1, [0, 1, 0], decimal=10)
        np.testing.assert_array_almost_equal(d2, [-1, 0, 0], decimal=10)

    def test_nonzero_dip(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Dip=90)
        d1 = np.array(ell.Direction1)
        d3 = np.array(ell.Direction3)
        np.testing.assert_array_almost_equal(d1, [0, 0, 1], decimal=10)
        np.testing.assert_array_almost_equal(d3, [-1, 0, 0], decimal=10)

    def test_nonzero_rotation(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Rotation=90)
        d2 = np.array(ell.Direction2)
        d3 = np.array(ell.Direction3)
        np.testing.assert_array_almost_equal(d2, [0, 0, 1], decimal=10)
        np.testing.assert_array_almost_equal(d3, [0, -1, 0], decimal=10)

    def test_directions_unit_vectors(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=37, Dip=23, Rotation=51)
        for d in [ell.Direction1, ell.Direction2, ell.Direction3]:
            norm = np.linalg.norm(d)
            assert abs(norm - 1.0) < 1e-10

    def test_directions_orthogonal(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=37, Dip=23, Rotation=51)
        d1 = np.array(ell.Direction1)
        d2 = np.array(ell.Direction2)
        d3 = np.array(ell.Direction3)
        assert abs(np.dot(d1, d2)) < 1e-10
        assert abs(np.dot(d1, d3)) < 1e-10
        assert abs(np.dot(d2, d3)) < 1e-10

    def test_180_degree_azimuth(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=180)
        d1 = np.array(ell.Direction1)
        np.testing.assert_array_almost_equal(d1, [-1, 0, 0], decimal=10)

    def test_360_degree_azimuth_same_as_zero(self):
        ell0 = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=0)
        ell360 = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=360)
        np.testing.assert_array_almost_equal(ell0.Direction1, ell360.Direction1, decimal=10)
        np.testing.assert_array_almost_equal(ell0.Direction2, ell360.Direction2, decimal=10)

    # ---- F-209: TVEllipsoid negative ranges validation ----

    def test_negative_ranges_raise_valueerror(self):
        """F-209 / H-06: TVEllipsoid raises ValueError for negative range values."""
        with pytest.raises(ValueError, match="ranges must be finite"):
            TVEllipsoid(R1=-1, R2=5, R3=3)
        with pytest.raises(ValueError, match="ranges must be finite"):
            TVEllipsoid(R1=10, R2=-5, R3=3)
        with pytest.raises(ValueError, match="ranges must be finite"):
            TVEllipsoid(R1=10, R2=5, R3=-3)

    def test_nan_inf_ranges_raise_valueerror(self):
        """H-06: NaN and Inf ranges must be rejected (not silently accepted)."""
        for bad in [float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match="ranges must be finite"):
                TVEllipsoid(bad, 1, 1)
            with pytest.raises(ValueError, match="ranges must be finite"):
                TVEllipsoid(1, bad, 1)
            with pytest.raises(ValueError, match="ranges must be finite"):
                TVEllipsoid(1, 1, bad)


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestTVVariogramSearchTemplate:
    def test_creation(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=10, Ellipsoid=ell
        )
        assert templ.LagWidth == 1.0
        assert templ.LagSeparation == 2.0
        assert templ.TolDistance == 1.0
        assert templ.NumLags == 10

    def test_num_lags_exceeds_max_raises(self):
        """Test that NumLags > MAX_NUM_LAGS raises ValueError."""
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        with pytest.raises(ValueError, match="NumLags .* exceeds maximum"):
            TVVariogramSearchTemplate(
                LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=20000, Ellipsoid=ell
            )

    def test_first_lag_distance_default(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=10, Ellipsoid=ell
        )
        assert templ.FirstLagDistance == 0

    def test_first_lag_distance_custom(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0,
            LagSeparation=2.0,
            TolDistance=1.0,
            NumLags=10,
            Ellipsoid=ell,
            FirstLagDistance=5.0,
        )
        assert templ.FirstLagDistance == 5.0

    def test_ellipsoid_reference(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=10, Ellipsoid=ell
        )
        assert templ.Ellipsoid is ell


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestIsInTunnel:
    def test_zero_r2_r3_raises(self):
        with pytest.raises(ValueError, match="ranges must be finite and positive"):
            TVEllipsoid(R1=10, R2=0, R3=0)

    def test_along_direction1_2d(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=5, Ellipsoid=ell
        )
        V = np.array([[5, 0, 0]])
        result = _IsInTunnel(templ, V)
        assert result[0]

    def test_perpendicular_outside_tunnel_2d(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=5, Ellipsoid=ell
        )
        V = np.array([[0, 10, 0]])
        result = _IsInTunnel(templ, V)
        assert not result[0]


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcSearchTemplateWindow:
    def test_larger_num_lags_increases_bounds(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ5 = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=5, Ellipsoid=ell
        )
        templ20 = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=20, Ellipsoid=ell
        )
        r5 = _CalcSearchTemplateWindow(templ5)
        r20 = _CalcSearchTemplateWindow(templ20)
        assert (r20[3] - r20[0]) >= (r5[3] - r5[0])


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcLagDistances:
    def test_lag_distances_correct(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=3.0, TolDistance=1.0, NumLags=4, Ellipsoid=ell
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        expected_distances = np.array([0, 3, 6, 9])
        np.testing.assert_array_equal(distances, expected_distances)

    def test_lag_start_end_surround_distance(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=4.0, TolDistance=1.0, NumLags=3, Ellipsoid=ell
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        for i in range(3):
            assert starts[i] < distances[i]
            assert ends[i] > distances[i]
            assert abs(ends[i] - starts[i] - 2.0) < 1e-10

    def test_first_lag_distance_offset(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0,
            LagSeparation=2.0,
            TolDistance=1.0,
            NumLags=3,
            Ellipsoid=ell,
            FirstLagDistance=5.0,
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(distances, expected)


# =============================================================================
# Core Computation Function Tests (Q2 fix — previously untested)
# =============================================================================


def _make_ellipsoid(r1=10, r2=5, r3=3):
    """Helper to create a TVEllipsoid with typical parameters."""
    return TVEllipsoid(R1=r1, R2=r2, R3=r3)


def _make_template(ell, num_lags=5, lag_width=1.0, lag_sep=2.0, tol_dist=1.0):
    """Helper to create a TVVariogramSearchTemplate."""
    return TVVariogramSearchTemplate(
        LagWidth=lag_width,
        LagSeparation=lag_sep,
        TolDistance=tol_dist,
        NumLags=num_lags,
        Ellipsoid=ell,
    )


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcLagsAreas:
    """Tests for _CalcLagsAreas function."""

    def test_lag_indexes_within_range(self):
        """Lag indexes are within [0, NumLags)."""
        from geo_bsd.variogram import _CalcLagsAreas

        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=4)
        i_arr, j_arr, k_arr, LagIndexes, LagDistance = _CalcLagsAreas(templ)
        assert LagIndexes.min() >= 0
        assert LagIndexes.max() <= 3  # 0-indexed, < num_lags

    def test_output_arrays_same_size(self):
        """I, J, K, LagIndexes have same length."""
        from geo_bsd.variogram import _CalcLagsAreas

        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        i_arr, j_arr, k_arr, LagIndexes, LagDistance = _CalcLagsAreas(templ)
        n = len(i_arr)
        assert n == len(j_arr) == len(k_arr) == len(LagIndexes)
        assert n > 0, "Should have at least one lag point"


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestVariogramCoreFunctions:
    """Tests for CalcVariogramFunction, CalcCovarianceFunction, CalcIndCorrelationFunction."""

    # ---- CalcVariogramFunction ----

    def test_calc_variogram_initializes_result(self):
        """CalcVariogramFunction initializes result on first call (Result=None)."""
        from geo_bsd.variogram import CalcVariogramFunction

        # Point1=None, Point2=None triggers initialization path (Result is None)
        params = {"HardData": [np.array([1.0, 2.0, 3.0], dtype="float32")]}
        result = CalcVariogramFunction(None, None, None, params)
        assert result is not None
        num_vals = len(params["HardData"])
        # Result shape: NumValues + NumValues + 1 = 2*n + 1
        assert len(result) == 2 * num_vals + 1

    # ---- CalcCovarianceFunction ----

    def test_calc_covariance_list_args(self):
        """CalcCovarianceFunction handles list-type point args (PointSetScanContStyle path)."""
        from geo_bsd.variogram import CalcCovarianceFunction

        values = [np.array([10.0, 20.0, 30.0], dtype="float32")]
        soft = [np.array([10.0, 20.0, 30.0], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        result = CalcCovarianceFunction(None, None, None, params)
        # Pass lists like PointSetScanContStyle does: [i], [j]
        result = CalcCovarianceFunction([0], [1], result, params)
        assert np.all(np.isfinite(result))

    # ---- CalcIndCorrelationFunction ----

    def test_calc_ind_correlation_list_args(self):
        """CalcIndCorrelationFunction handles list-type point args (PointSetScanContStyle path)."""
        from geo_bsd.variogram import CalcIndCorrelationFunction

        values = [np.array([10.0, 20.0, 30.0], dtype="float32")]
        soft = [np.array([10.0, 20.0, 30.0], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        # Pass lists like PointSetScanContStyle does: [i], [j]
        result = CalcIndCorrelationFunction([0], [1], result, params)
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestPointSetScanContStyle:
    """Tests for PointSetScanContStyle function.

    Note: Passing Function=None returns zeros((NumLags, 1)), LagDistance
    (variogram.py) — no UnboundLocalError exists. These tests use a trivial
    function to exercise the scan path.
    """

    def _trivial_fn(self, p1, p2, result, params):
        """A trivial accumulator function for scan tests."""
        return result if result is not None else np.zeros(3)

    def test_scan_with_empty_pointset_completes(self):
        """PointSetScanContStyle with empty point set doesn't crash."""
        from geo_bsd.variogram import PointSetScanContStyle

        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        point_set = {
            "X": np.array([], dtype="int32"),
            "Y": np.array([], dtype="int32"),
            "Z": np.array([], dtype="int32"),
        }
        result, lag_dist = PointSetScanContStyle(templ, point_set, self._trivial_fn, None)
        assert lag_dist is not None


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestPointSetScanGridStyle:
    """Tests for PointSetScanGridStyle function.

    Note: Passing Function=None returns zeros((NumLags, 1)), LagDistance
    (variogram.py) — no UnboundLocalError exists. These tests use a trivial
    function to exercise the scan path.
    """

    def _trivial_fn(self, p1, p2, result, params):
        return result if result is not None else np.zeros(3)

    def test_scan_with_empty_points_completes(self):
        """PointSetScanGridStyle with empty arrays doesn't crash."""
        from geo_bsd.variogram import PointSetScanGridStyle

        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        xyz = (
            np.array([], dtype="int32"),
            np.array([], dtype="int32"),
            np.array([], dtype="int32"),
        )
        result, lag_dist = PointSetScanGridStyle(templ, xyz, self._trivial_fn, None)
        assert lag_dist is not None


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCubeScan:
    """Tests for CubeScan function.

    Note: Passing Function=None returns zeros((NumLags, 1)), LagDistance
    (variogram.py) — no UnboundLocalError exists. These tests use a trivial
    function to exercise the scan path.
    """

    def _trivial_fn(self, p1, p2, result, params):
        return result if result is not None else np.zeros(3)

    def test_cube_scan_with_zero_mask_completes(self):
        """CubeScan with a mask of all zeros doesn't crash."""
        from geo_bsd.variogram import CubeScan

        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        mask = np.zeros((5, 5, 3), dtype="uint8")
        result, lag_dist = CubeScan(templ, mask, self._trivial_fn, None)
        assert result is not None
        assert len(lag_dist) == templ.NumLags


# =============================================================================
# NaN/Inf Input Tests for Variogram Core Functions (F-093)
# =============================================================================


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestVariogramNaNInfHandling:
    """Tests for NaN/Inf injection in variogram core functions."""

    # ---- CalcVariogramFunction NaN/Inf ----

    def test_calc_variogram_nan_in_harddata(self):
        """CalcVariogramFunction with NaN in HardData raises ValueError."""
        from geo_bsd.variogram import CalcVariogramFunction

        values = [np.array([np.nan, 2.0, 3.0], dtype="float32")]
        params = {"HardData": values}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcVariogramFunction(None, None, None, params)

    def test_calc_variogram_inf_in_harddata(self):
        """CalcVariogramFunction with Inf in HardData raises ValueError."""
        from geo_bsd.variogram import CalcVariogramFunction

        values = [np.array([np.inf, 2.0, 3.0], dtype="float32")]
        params = {"HardData": values}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcVariogramFunction(None, None, None, params)

    # ---- CalcCovarianceFunction NaN/Inf ----

    def test_calc_covariance_nan_in_data(self):
        """CalcCovarianceFunction with NaN in HardData or SoftData raises ValueError."""
        from geo_bsd.variogram import CalcCovarianceFunction

        values = [np.array([np.nan, 2.0, 3.0], dtype="float32")]
        soft = [np.array([1.0, 2.0, 3.0], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcCovarianceFunction(None, None, None, params)

    def test_calc_covariance_inf_in_data(self):
        """CalcCovarianceFunction with Inf in HardData raises ValueError."""
        from geo_bsd.variogram import CalcCovarianceFunction

        values = [np.array([np.inf, 2.0, 3.0], dtype="float32")]
        soft = [np.array([1.0, 2.0, 3.0], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcCovarianceFunction(None, None, None, params)

    # ---- CalcIndCorrelationFunction NaN/Inf ----

    def test_calc_ind_correlation_nan_in_data(self):
        """CalcIndCorrelationFunction with NaN in HardData raises ValueError."""
        from geo_bsd.variogram import CalcIndCorrelationFunction

        values = [np.array([np.nan, 1.0, 0.0], dtype="float32")]
        soft = [np.array([0.5, 0.5, 0.5], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcIndCorrelationFunction(None, None, None, params)

    def test_calc_ind_correlation_inf_in_data(self):
        """CalcIndCorrelationFunction with Inf in HardData raises ValueError."""
        from geo_bsd.variogram import CalcIndCorrelationFunction

        values = [np.array([np.inf, 1.0, 0.0], dtype="float32")]
        soft = [np.array([0.5, 0.5, 0.5], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            CalcIndCorrelationFunction(None, None, None, params)


# =============================================================================
# M-P-30: CubeScan tuple-point path tests
# =============================================================================


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCubeScanTotalWorkCap:
    """F-03: the pure-Python grid path must reject a legal input whose total
    work (len(LagIndexes) × grid volume) exceeds MAX_TOTAL_GRID_WORK before
    running the O(offsets × volume) loop. Pre-fix only MAX_WINDOW_VOLUME
    bounded the search-window offset list; a large template + large grid ran
    ~8e12 numpy ops (~2 h) and a 1000^3 grid allocated a 24 GB mgrid."""

    def test_cube_scan_below_cap_still_works(self):
        """Control: a small grid/template below the cap still computes."""
        ell = _make_ellipsoid(r1=5, r2=3, r3=2)
        templ = _make_template(ell, num_lags=2, lag_width=1.0, lag_sep=1.0)
        mask = np.ones((5, 5, 3), dtype="uint8")
        res, lag_dist = CubeScan(templ, mask, lambda *a: np.zeros(3), None)
        assert np.all(np.isfinite(res))
        assert len(lag_dist) == templ.NumLags

    def test_real_cap_defined(self):
        import geo_bsd.variogram as v

        assert v.MAX_TOTAL_GRID_WORK == 1e8


# =============================================================================
# F-27: GridStyle work cap must bound the real per-pair cost
# =============================================================================


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestGridStyleFractionalSpacing:
    """III-15: exact-integer offset matching silently dropped EVERY pair on
    fractional grid spacing (0.5/0.25 m). Continuous band binning (C++ /
    ContStyle parity) must count those pairs."""

    def _count_fn(self, p1, p2, result, params):
        if result is None:
            return np.zeros(3, dtype=np.float64)
        result[2] += 1
        return result

    def test_fractional_spacing_pairs_counted(self):
        ell = TVEllipsoid(R1=100, R2=100, R3=100)
        templ = TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell, FirstLagDistance=0,
        )
        # 6 points at 0, 0.5, 1.0, 1.5, 2.0, 2.5 on the X axis.
        xs = np.arange(6, dtype="float32") * 0.5
        ys = np.zeros(6, dtype="float32")
        zs = np.zeros(6, dtype="float32")
        res, _ = PointSetScanGridStyle(
            templ, (xs, ys, zs), self._count_fn, None
        )
        # 30 ordered pairs (6×5, self-pairs skipped) must be counted across
        # lag bands; the exact pre-fix integer matching counted ZERO
        # (repro: 0 of 30 ordered pairs).
        total = int(res[:, 2].sum())
        assert total > 0, "fractional-spacing pairs must be binned (III-15)"
        assert total == 30, f"expected 30 ordered pairs, got {total}"


# =============================================================================
# II-34: CalcIndCorrelationFunction must EXCLUDE soft-prob 0/1 pairs
# =============================================================================


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcIndCorrelationExcludesDegenerate:
    """II-34: soft-prob variance ≤ 0 (soft prob 0 or 1) must EXCLUDE the
    pair, not substitute denom=1.0. Substitution let the unnormalized raw
    covariance dominate, diluting/inflating the correlation outside [-1,1].
    Exclusion is the standard guard (C++ kernel parity)."""

    def test_soft_prob_01_pairs_excluded_from_count(self):
        from geo_bsd.variogram import CalcIndCorrelationFunction

        values = [np.array([0.0, 1.0, 0.0, 1.0], dtype="float32")]
        soft = [np.array([0.0, 0.0, 1.0, 1.0], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        # Pair (0,1): soft 0 and 0 -> product 0 -> excluded. Pair (2,3):
        # soft 1 and 1 -> product 0 -> excluded. Count slot (index 2) stays 0.
        result = CalcIndCorrelationFunction(np.array([0]), np.array([1]), result, params)
        result = CalcIndCorrelationFunction(np.array([2]), np.array([3]), result, params)
        assert result[2] == 0, (
            "II-34: degenerate soft-prob pairs must be excluded from the "
            f"count, got {result[2]} (pre-fix substitution counted them)"
        )
        assert np.all(np.isfinite(result))

    def test_valid_pairs_still_counted(self):
        """Control: soft probs strictly inside (0,1) are counted normally."""
        from geo_bsd.variogram import CalcIndCorrelationFunction

        values = [np.array([0.0, 1.0], dtype="float32")]
        soft = [np.array([0.5, 0.5], dtype="float32")]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        result = CalcIndCorrelationFunction(np.array([0]), np.array([1]), result, params)
        assert result[2] == 1, "valid soft-prob pairs must be counted"
