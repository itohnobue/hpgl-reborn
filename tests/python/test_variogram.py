import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.variogram import (
        TVEllipsoid, TVVariogramSearchTemplate,
        _IsInTunnel, _CalcSearchTemplateWindow, _CalcLagDistances
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
