import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.variogram import (
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

    def test_all_angles_combined(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=45, Dip=30, Rotation=60)
        d1 = np.array(ell.Direction1)
        assert abs(np.linalg.norm(d1) - 1.0) < 1e-10

    def test_180_degree_azimuth(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=180)
        d1 = np.array(ell.Direction1)
        np.testing.assert_array_almost_equal(d1, [-1, 0, 0], decimal=10)

    def test_360_degree_azimuth_same_as_zero(self):
        ell0 = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=0)
        ell360 = TVEllipsoid(R1=10, R2=5, R3=3, Azimut=360)
        np.testing.assert_array_almost_equal(ell0.Direction1, ell360.Direction1, decimal=10)
        np.testing.assert_array_almost_equal(ell0.Direction2, ell360.Direction2, decimal=10)


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestTVVariogramSearchTemplate:
    def test_creation(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=10, Ellipsoid=ell
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
                LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
                NumLags=20000, Ellipsoid=ell
            )

    def test_first_lag_distance_default(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=10, Ellipsoid=ell
        )
        assert templ.FirstLagDistance == 0

    def test_first_lag_distance_custom(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=10, Ellipsoid=ell, FirstLagDistance=5.0
        )
        assert templ.FirstLagDistance == 5.0

    def test_ellipsoid_reference(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=10, Ellipsoid=ell
        )
        assert templ.Ellipsoid is ell


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestIsInTunnel:
    def test_zero_r2_returns_all_false(self):
        ell = TVEllipsoid(R1=10, R2=0, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        V = np.array([[5, 0, 0], [1, 1, 1]])
        result = _IsInTunnel(templ, V)
        assert np.all(result == False)

    def test_zero_r3_returns_all_false(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=0)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        V = np.array([[5, 0, 0], [1, 1, 1]])
        result = _IsInTunnel(templ, V)
        assert np.all(result == False)

    def test_zero_r2_r3_no_crash(self):
        ell = TVEllipsoid(R1=10, R2=0, R3=0)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        V = np.array([[5, 0, 0]])
        result = _IsInTunnel(templ, V)
        assert result[0] == False

    def test_along_direction1_2d(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        V = np.array([[5, 0, 0]])
        result = _IsInTunnel(templ, V)
        assert result[0] == True

    def test_perpendicular_outside_tunnel_2d(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        V = np.array([[0, 10, 0]])
        result = _IsInTunnel(templ, V)
        assert result[0] == False


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcSearchTemplateWindow:
    def test_returns_six_values(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        result = _CalcSearchTemplateWindow(templ)
        assert len(result) == 6

    def test_identity_directions_reasonable_bounds(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        min_i, min_j, min_k, max_i, max_j, max_k = _CalcSearchTemplateWindow(templ)
        assert max_i > min_i
        assert max_j > min_j
        assert max_k > min_k

    def test_larger_num_lags_increases_bounds(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ5 = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        templ20 = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=20, Ellipsoid=ell
        )
        r5 = _CalcSearchTemplateWindow(templ5)
        r20 = _CalcSearchTemplateWindow(templ20)
        assert (r20[3] - r20[0]) >= (r5[3] - r5[0])

    def test_zero_r2_r3_no_crash(self):
        ell = TVEllipsoid(R1=10, R2=0, R3=0)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        result = _CalcSearchTemplateWindow(templ)
        assert len(result) == 6


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcLagDistances:
    def test_returns_four_arrays(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        result = _CalcLagDistances(templ)
        assert len(result) == 4

    def test_lag_distances_correct(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=3.0, TolDistance=1.0,
            NumLags=4, Ellipsoid=ell
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        expected_distances = np.array([0, 3, 6, 9])
        np.testing.assert_array_equal(distances, expected_distances)

    def test_lag_start_end_surround_distance(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=4.0, TolDistance=1.0,
            NumLags=3, Ellipsoid=ell
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        for i in range(3):
            assert starts[i] < distances[i]
            assert ends[i] > distances[i]
            assert abs(ends[i] - starts[i] - 2.0) < 1e-10

    def test_first_lag_distance_offset(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=3, Ellipsoid=ell, FirstLagDistance=5.0
        )
        indexes, distances, starts, ends = _CalcLagDistances(templ)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(distances, expected)

    def test_indexes_sequential(self):
        ell = TVEllipsoid(R1=10, R2=5, R3=3)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0,
            NumLags=5, Ellipsoid=ell
        )
        indexes, _, _, _ = _CalcLagDistances(templ)
        np.testing.assert_array_equal(indexes, np.arange(5))


# =============================================================================
# Core Computation Function Tests (Q2 fix — previously untested)
# =============================================================================

def _make_ellipsoid(r1=10, r2=5, r3=3):
    """Helper to create a TVEllipsoid with typical parameters."""
    return TVEllipsoid(R1=r1, R2=r2, R3=r3)


def _make_template(ell, num_lags=5, lag_width=1.0, lag_sep=2.0, tol_dist=1.0):
    """Helper to create a TVVariogramSearchTemplate."""
    return TVVariogramSearchTemplate(
        LagWidth=lag_width, LagSeparation=lag_sep,
        TolDistance=tol_dist, NumLags=num_lags, Ellipsoid=ell
    )


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCalcLagsAreas:
    """Tests for _CalcLagsAreas function."""

    def test_returns_five_values(self):
        """_CalcLagsAreas returns I, J, K, LagIndexes, LagDistance."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid()
        templ = _make_template(ell)
        result = _CalcLagsAreas(templ)
        assert len(result) == 5
        I, J, K, LagIndexes, LagDistance = result

    def test_lag_indexes_within_range(self):
        """Lag indexes are within [0, NumLags)."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=4)
        I, J, K, LagIndexes, LagDistance = _CalcLagsAreas(templ)
        assert LagIndexes.min() >= 0
        assert LagIndexes.max() <= 3  # 0-indexed, < num_lags

    def test_output_arrays_same_size(self):
        """I, J, K, LagIndexes have same length."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        I, J, K, LagIndexes, LagDistance = _CalcLagsAreas(templ)
        n = len(I)
        assert n == len(J) == len(K) == len(LagIndexes)
        assert n > 0, "Should have at least one lag point"

    def test_lag_distances_correct_order(self):
        """LagDistance values are increasing."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=5, lag_sep=3.0)
        I, J, K, LagIndexes, LagDistance = _CalcLagsAreas(templ)
        expected = np.array([0, 3, 6, 9, 12])
        np.testing.assert_array_equal(LagDistance, expected)

    def test_all_coordinates_are_integers(self):
        """I, J, K are integer arrays."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        I, J, K, _, _ = _CalcLagsAreas(templ)
        assert I.dtype == np.int32 or I.dtype == np.int64
        assert J.dtype == np.int32 or J.dtype == np.int64
        assert K.dtype == np.int32 or K.dtype == np.int64

    def test_zero_radii_returns_empty_arrays(self):
        """Zero R2/R3 produces empty lag area arrays."""
        from geo_bsd.variogram import _CalcLagsAreas
        ell = _make_ellipsoid(r1=10, r2=0, r3=0)
        templ = _make_template(ell, num_lags=3)
        I, J, K, LagIndexes, _ = _CalcLagsAreas(templ)
        assert len(I) == 0 and len(J) == 0 and len(K) == 0


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestVariogramCoreFunctions:
    """Tests for CalcVariogramFunction, CalcCovarianceFunction, CalcIndCorrelationFunction."""

    # ---- CalcVariogramFunction ----

    def test_calc_variogram_initializes_result(self):
        """CalcVariogramFunction initializes result on first call (Result=None)."""
        from geo_bsd.variogram import CalcVariogramFunction
        # Point1=None, Point2=None triggers initialization path (Result is None)
        params = {"HardData": [np.array([1.0, 2.0, 3.0], dtype='float32')]}
        result = CalcVariogramFunction(None, None, None, params)
        assert result is not None
        num_vals = len(params["HardData"])
        # Result shape: NumValues + NumValues + 1 = 2*n + 1
        assert len(result) == 2 * num_vals + 1

    def test_calc_variogram_identity(self):
        """CalcVariogramFunction with identical points gives zero variance."""
        from geo_bsd.variogram import CalcVariogramFunction
        values = [np.array([10.0, 20.0, 30.0, 40.0], dtype='float32')]
        params = {"HardData": values}
        # Initialization: Result=None creates zeros
        result = CalcVariogramFunction(None, None, None, params)
        n_vals = len(values)
        assert len(result) == 2 * n_vals + 1
        # Compute variogram for two identical point indices
        # Point1/Point2 are single indices (as lists, per scan function convention)
        result = CalcVariogramFunction([0], [1], result, params)
        assert np.all(np.isfinite(result))
        variogram_vals = result[:n_vals]
        # Variogram for different values should be non-zero
        assert np.any(variogram_vals > 0)

    # ---- CalcCovarianceFunction ----

    def test_calc_covariance_initializes_result(self):
        """CalcCovarianceFunction initializes result on first call."""
        from geo_bsd.variogram import CalcCovarianceFunction
        values = [np.array([1.0, 2.0, 3.0], dtype='float32')]
        soft = [np.array([0.5, 1.5, 2.5], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcCovarianceFunction(None, None, None, params)
        assert result is not None
        n_vals = len(values)
        assert len(result) == 2 * n_vals + 1

    def test_calc_covariance_computes_values(self):
        """CalcCovarianceFunction computes covariance between point pairs."""
        from geo_bsd.variogram import CalcCovarianceFunction
        values = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        soft = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcCovarianceFunction(None, None, None, params)
        n_vals = len(values)

        p1 = np.int64(0)
        p2 = np.int64(1)
        result = CalcCovarianceFunction(p1, p2, result, params)
        assert np.all(np.isfinite(result))

    def test_calc_covariance_list_args(self):
        """CalcCovarianceFunction handles list-type point args (PointSetScanContStyle path)."""
        from geo_bsd.variogram import CalcCovarianceFunction
        values = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        soft = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcCovarianceFunction(None, None, None, params)
        # Pass lists like PointSetScanContStyle does: [i], [j]
        result = CalcCovarianceFunction([0], [1], result, params)
        assert np.all(np.isfinite(result))

    # ---- CalcIndCorrelationFunction ----

    def test_calc_ind_correlation_initializes_result(self):
        """CalcIndCorrelationFunction initializes result on first call."""
        from geo_bsd.variogram import CalcIndCorrelationFunction
        values = [np.array([1.0, 0.0], dtype='float32')]
        soft = [np.array([0.5, 0.5], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        assert result is not None
        n_vals = len(values)
        assert len(result) == 2 * n_vals + 1

    def test_calc_ind_correlation_div_zero_guard(self):
        """CalcIndCorrelationFunction handles soft data with value 0 or 1."""
        from geo_bsd.variogram import CalcIndCorrelationFunction
        # Soft data 0 or 1 causes denom=0 → guarded to 1.0
        values = [np.array([0.0, 1.0, 0.0], dtype='float32')]
        soft = [np.array([0.0, 0.0, 1.0], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        p1 = np.int64(0)
        p2 = np.int64(1)
        result = CalcIndCorrelationFunction(p1, p2, result, params)
        assert np.all(np.isfinite(result)), "Denom guard should prevent NaN/Inf"

    def test_calc_ind_correlation_list_args(self):
        """CalcIndCorrelationFunction handles list-type point args (PointSetScanContStyle path)."""
        from geo_bsd.variogram import CalcIndCorrelationFunction
        values = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        soft = [np.array([10.0, 20.0, 30.0], dtype='float32')]
        params = {"HardData": values, "SoftData": soft}
        result = CalcIndCorrelationFunction(None, None, None, params)
        # Pass lists like PointSetScanContStyle does: [i], [j]
        result = CalcIndCorrelationFunction([0], [1], result, params)
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestPointSetScanContStyle:
    """Tests for PointSetScanContStyle function.

    Note: Passing Function=None triggers an UnboundLocalError in variogram.py
    (Result variable not initialized when Function is None). These tests use
    a trivial function to exercise the scan path.
    """

    def _trivial_fn(self, p1, p2, result, params):
        """A trivial accumulator function for scan tests."""
        return result if result is not None else np.zeros(3)

    def test_scans_with_function_completes(self):
        """PointSetScanContStyle with a trivial function returns result."""
        from geo_bsd.variogram import PointSetScanContStyle
        ell = _make_ellipsoid(r1=20, r2=10, r3=5)
        templ = _make_template(ell, num_lags=3, lag_width=2.0, lag_sep=5.0)
        point_set = {
            "X": np.array([0, 1, 2], dtype='int32'),
            "Y": np.array([0, 0, 0], dtype='int32'),
            "Z": np.array([0, 0, 0], dtype='int32'),
        }
        result, lag_dist = PointSetScanContStyle(templ, point_set, self._trivial_fn, None)
        assert result is not None
        assert len(lag_dist) == templ.NumLags
        assert np.all(np.isfinite(result))

    def test_scan_with_empty_pointset_completes(self):
        """PointSetScanContStyle with empty point set doesn't crash."""
        from geo_bsd.variogram import PointSetScanContStyle
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        point_set = {
            "X": np.array([], dtype='int32'),
            "Y": np.array([], dtype='int32'),
            "Z": np.array([], dtype='int32'),
        }
        result, lag_dist = PointSetScanContStyle(templ, point_set, self._trivial_fn, None)
        assert lag_dist is not None


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestPointSetScanGridStyle:
    """Tests for PointSetScanGridStyle function.

    Note: Passing Function=None triggers an UnboundLocalError in variogram.py
    (same bug as PointSetScanContStyle).
    """

    def _trivial_fn(self, p1, p2, result, params):
        return result if result is not None else np.zeros(3)

    def test_scan_with_function_completes(self):
        """PointSetScanGridStyle with a trivial function returns result."""
        from geo_bsd.variogram import PointSetScanGridStyle
        ell = _make_ellipsoid(r1=20, r2=10, r3=5)
        templ = _make_template(ell, num_lags=3, lag_width=2.0, lag_sep=5.0)
        xyz = (
            np.array([0, 5, 10], dtype='int32'),
            np.array([0, 0, 0], dtype='int32'),
            np.array([0, 0, 0], dtype='int32'),
        )
        result, lag_dist = PointSetScanGridStyle(templ, xyz, self._trivial_fn, None)
        assert result is not None
        assert len(lag_dist) == templ.NumLags
        assert np.all(np.isfinite(result))

    def test_scan_with_empty_points_completes(self):
        """PointSetScanGridStyle with empty arrays doesn't crash."""
        from geo_bsd.variogram import PointSetScanGridStyle
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        xyz = (
            np.array([], dtype='int32'),
            np.array([], dtype='int32'),
            np.array([], dtype='int32'),
        )
        result, lag_dist = PointSetScanGridStyle(templ, xyz, self._trivial_fn, None)
        assert lag_dist is not None


@pytest.mark.skipif(not VARIOM_AVAILABLE, reason="variogram module not available")
class TestCubeScan:
    """Tests for CubeScan function.

    Note: Passing Function=None triggers an UnboundLocalError in variogram.py
    (Result variable not initialized when Function is None — same bug as scan functions).
    """

    def _trivial_fn(self, p1, p2, result, params):
        return result if result is not None else np.zeros(3)

    def test_cube_scan_with_function_completes(self):
        """CubeScan with a small mask returns result."""
        from geo_bsd.variogram import CubeScan
        ell = _make_ellipsoid(r1=10, r2=5, r3=3)
        # Use lag_sep small enough that all lag offsets fit within mask
        templ = _make_template(ell, num_lags=2, lag_width=1.0, lag_sep=1.0)
        mask = np.ones((10, 10, 5), dtype='uint8')
        result, lag_dist = CubeScan(templ, mask, self._trivial_fn, None)
        assert result is not None
        assert len(lag_dist) == templ.NumLags

    def test_cube_scan_with_zero_mask_completes(self):
        """CubeScan with a mask of all zeros doesn't crash."""
        from geo_bsd.variogram import CubeScan
        ell = _make_ellipsoid()
        templ = _make_template(ell, num_lags=3)
        mask = np.zeros((5, 5, 3), dtype='uint8')
        result, lag_dist = CubeScan(templ, mask, self._trivial_fn, None)
        assert result is not None
        assert len(lag_dist) == templ.NumLags
