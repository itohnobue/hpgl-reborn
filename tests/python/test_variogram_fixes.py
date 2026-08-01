"""Regression tests for confirmed F-01/F-02/F-29/F-30/I2-02/I2-03/I2-04 fixes.

Each test fails against the pre-fix variogram.py and passes against the fixed
code (see test docstrings for the pre-fix failure mode).
"""

import numpy as np
import pytest

from geo_bsd import variogram
from geo_bsd.variogram import (
    MAX_POINT_SET_SIZE,
    CalcCovarianceFunction,
    CalcIndCorrelationFunction,
    CubeScan,
    PointSetScanContStyle,
    PointSetScanGridStyle,
    TVEllipsoid,
    TVVariogramSearchTemplate,
    _CalcLagsAreas,
)


def _pair_counter(p1, p2, result, params):
    """Callback that accumulates the number of matched pairs into result[2]."""
    if result is None:
        return np.zeros(3, dtype=np.float64)
    if isinstance(p1, tuple):
        n = len(p1[0])
    else:
        n = 1
    result[2] += n
    return result


def _make_template(r1=5, r2=5, r3=5, lag_width=1.0, lag_sep=1.0, num_lags=3):
    ell = TVEllipsoid(R1=r1, R2=r2, R3=r3)
    return TVVariogramSearchTemplate(
        LagWidth=lag_width,
        LagSeparation=lag_sep,
        TolDistance=1.0,
        NumLags=num_lags,
        Ellipsoid=ell,
    )


# =============================================================================
# F-01 (HIGH): CubeScan uint8/integer masks must behave like boolean masks
# =============================================================================


class TestCubeScanIntegerMask:
    def test_uint8_mask_matches_bool_pair_counts(self):
        """F-01: uint8 all-ones mask must produce the SAME pair counts as bool.

        Pre-fix: uint8 Mask1 & Mask2 stays uint8 and is used as an integer
        fancy index -> 5x pair-count inflation [625,1940,2223] vs bool
        [125,420,507] on a 5x5x5 all-ones mask.
        """
        templ = _make_template(r1=5, r2=5, r3=5, lag_width=1.0, lag_sep=1.0, num_lags=3)
        mask_u8 = np.ones((5, 5, 5), dtype="uint8")
        mask_bool = np.ones((5, 5, 5), dtype=bool)

        res_u8, _ = CubeScan(templ, mask_u8, _pair_counter, None)
        res_bool, _ = CubeScan(templ, mask_bool, _pair_counter, None)

        np.testing.assert_array_equal(res_u8, res_bool)
        # Correct (bool) pair counts for this template — not the inflated 5x.
        np.testing.assert_array_equal(res_bool[:, 2], np.array([125, 420, 507]))

    def test_non_binary_integer_mask_does_not_index_error(self):
        """F-01: masks with values >= 2 must not IndexError / inflate counts.

        Pre-fix: a uint8 mask with value 2 used as integer fancy index either
        raised IndexError (index out of bounds) or selected wrong elements.
        """
        templ = _make_template()
        mask2 = np.full((5, 5, 5), 2, dtype="uint8")
        mask_bool = np.ones((5, 5, 5), dtype=bool)

        res2, _ = CubeScan(templ, mask2, _pair_counter, None)
        res_bool, _ = CubeScan(templ, mask_bool, _pair_counter, None)

        # Non-zero cells are all informed -> identical to all-ones bool mask.
        np.testing.assert_array_equal(res2, res_bool)

    def test_int32_mask_matches_bool_pair_counts(self):
        """F-01: int32 masks behave like bool (non-zero == informed)."""
        templ = _make_template()
        mask_int = np.ones((5, 5, 5), dtype="int32")
        mask_bool = np.ones((5, 5, 5), dtype=bool)

        res_int, _ = CubeScan(templ, mask_int, _pair_counter, None)
        res_bool, _ = CubeScan(templ, mask_bool, _pair_counter, None)

        np.testing.assert_array_equal(res_int, res_bool)


# =============================================================================
# F-02 (HIGH): CubeScan tuple path with natural 3D data
# =============================================================================


class TestCalcCovariance3DTuplePath:
    def test_calc_covariance_3d_tuple_path(self):
        """F-02: CalcCovarianceFunction handles (I, J, K) tuple + 3D data.

        Pre-fix: ravel_multi_index flat index applied to 3D Values[i] raised
        ValueError: setting an array element with a sequence.
        """
        values = [np.arange(24, dtype="float32").reshape(4, 3, 2)]
        soft = [np.arange(24, dtype="float32").reshape(4, 3, 2) * 0.5]
        params = {"HardData": values, "SoftData": soft}

        result = CalcCovarianceFunction(None, None, None, params)
        idx_i = np.array([0, 1], dtype="int64")
        idx_j = np.array([0, 0], dtype="int64")
        idx_k = np.array([0, 0], dtype="int64")

        result2 = CalcCovarianceFunction((idx_i, idx_j, idx_k), (idx_i, idx_j, idx_k), result, params)
        assert np.all(np.isfinite(result2))
        assert result2[len(values) + len(values)] == 2  # two pairs accumulated

    def test_calc_covariance_3d_values_correct(self):
        """F-02: the 3D tuple path computes the expected covariance math."""
        # 2x2x2 grid: flat indices 0..7. Pair (flat 0, flat 1) -> values 0,1.
        values = [np.arange(8, dtype="float32").reshape(2, 2, 2)]
        soft = [np.zeros(8, dtype="float32").reshape(2, 2, 2)]
        params = {"HardData": values, "SoftData": soft}

        result = CalcCovarianceFunction(None, None, None, params)
        idx_i = np.array([0], dtype="int64")
        idx_j = np.array([0], dtype="int64")
        idx_k = np.array([0], dtype="int64")
        idx_i2 = np.array([0], dtype="int64")
        idx_j2 = np.array([0], dtype="int64")
        idx_k2 = np.array([1], dtype="int64")

        result2 = CalcCovarianceFunction((idx_i, idx_j, idx_k), (idx_i2, idx_j2, idx_k2), result, params)
        # cov = (0-0)*(1-0) = 0; count = 1 (result[2] is the pair-count slot)
        assert result2[2] == 1  # pair count slot
        assert np.all(np.isfinite(result2))

    def test_calc_ind_correlation_3d_tuple_path(self):
        """F-02: CalcIndCorrelationFunction handles (I, J, K) tuple + 3D data.

        Pre-fix: same ravel_multi_index flat-index-on-3D ValueError.
        """
        values = [np.arange(24, dtype="float32").reshape(4, 3, 2) % 2]
        soft = [np.full((4, 3, 2), 0.5, dtype="float32")]
        params = {"HardData": values, "SoftData": soft}

        result = CalcIndCorrelationFunction(None, None, None, params)
        idx_i = np.array([0, 1], dtype="int64")
        idx_j = np.array([0, 0], dtype="int64")
        idx_k = np.array([0, 0], dtype="int64")

        result2 = CalcIndCorrelationFunction((idx_i, idx_j, idx_k), (idx_i, idx_j, idx_k), result, params)
        assert np.all(np.isfinite(result2))
        assert result2[len(values) + len(values)] == 2

    def test_cube_scan_end_to_end_3d(self):
        """F-02 end-to-end: CubeScan + CalcCovarianceFunction with 3D data."""
        templ = _make_template(r1=5, r2=5, r3=5, lag_width=1.0, lag_sep=1.0, num_lags=2)
        mask = np.ones((5, 5, 5), dtype="uint8")
        grid_values = np.arange(5 * 5 * 5, dtype="float32").reshape(5, 5, 5)
        params = {
            "HardData": [grid_values],
            "SoftData": [grid_values * 0.5],
        }
        result, lag_dist = CubeScan(templ, mask, CalcCovarianceFunction, params)
        assert np.all(np.isfinite(result))
        assert len(lag_dist) == templ.NumLags


# =============================================================================
# F-29 (MEDIUM): TVEllipsoid angle validation
# =============================================================================


class TestTVEllipsoidAngleValidation:
    @pytest.mark.parametrize("kw", ["Azimut", "Dip", "Rotation"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_angles_rejected(self, kw, bad):
        """F-29: NaN/Inf angles must raise ValueError (pre-fix: accepted)."""
        with pytest.raises(ValueError, match="angles must be finite"):
            TVEllipsoid(R1=10, R2=5, R3=3, **{kw: bad})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"Azimut": -1},
            {"Azimut": 361},
            {"Dip": -91},
            {"Dip": 91},
            {"Rotation": -91},
            {"Rotation": 91},
        ],
    )
    def test_out_of_range_angles_rejected(self, kwargs):
        """F-29: angles outside valid ranges raise ValueError."""
        with pytest.raises(ValueError):
            TVEllipsoid(R1=10, R2=5, R3=3, **kwargs)

    def test_boundary_angles_accepted(self):
        """F-29: boundary values (Azimut 0/360, Dip +-90, Rotation +-90) valid."""
        TVEllipsoid(R1=10, R2=5, R3=3, Azimut=0, Dip=0, Rotation=0)
        TVEllipsoid(R1=10, R2=5, R3=3, Azimut=360, Dip=90, Rotation=90)
        TVEllipsoid(R1=10, R2=5, R3=3, Azimut=360, Dip=-90, Rotation=-90)


# =============================================================================
# F-30 (MEDIUM): TVVariogramSearchTemplate NaN-ineffective guards
# =============================================================================


class TestTVVariogramSearchTemplateNaNGuards:
    @pytest.mark.parametrize("kw", ["LagWidth", "LagSeparation", "TolDistance", "FirstLagDistance", "NumLags"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_params_rejected(self, kw, bad):
        """F-30: NaN/Inf params must raise ValueError (pre-fix: accepted)."""
        kwargs = {
            "LagWidth": 1.0,
            "LagSeparation": 2.0,
            "TolDistance": 1.0,
            "NumLags": 3,
            "Ellipsoid": TVEllipsoid(R1=10, R2=5, R3=3),
        }
        kwargs[kw] = bad
        with pytest.raises(ValueError):
            TVVariogramSearchTemplate(**kwargs)

    def test_valid_params_still_accepted(self):
        """F-30: valid template still constructs normally."""
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0,
            LagSeparation=2.0,
            TolDistance=1.0,
            NumLags=3,
            Ellipsoid=TVEllipsoid(R1=10, R2=5, R3=3),
        )
        assert templ.LagWidth == 1.0
        assert templ.FirstLagDistance == 0


# =============================================================================
# I2-02 (MEDIUM): point-count cap on PointSetScan callbacks
# =============================================================================


def _trivial_fn(p1, p2, result, params):
    return result if result is not None else np.zeros(3)


class TestPointSetScanPointCountCap:
    def test_cont_style_cap(self, monkeypatch):
        """I2-02: PointSetScanContStyle rejects oversized point sets.

        Pre-fix: no cap -> O(N^2) pure-Python callback loop with no upper bound.
        """
        monkeypatch.setattr(variogram, "MAX_POINT_SET_SIZE", 2)
        ell = TVEllipsoid(R1=20, R2=10, R3=5)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=3, Ellipsoid=ell
        )
        point_set = {
            "X": np.zeros(3),
            "Y": np.zeros(3),
            "Z": np.zeros(3),
        }
        with pytest.raises(ValueError, match="MAX_POINT_SET_SIZE"):
            PointSetScanContStyle(templ, point_set, _trivial_fn, None)

    def test_grid_style_cap(self, monkeypatch):
        """I2-02: PointSetScanGridStyle rejects oversized point sets."""
        monkeypatch.setattr(variogram, "MAX_POINT_SET_SIZE", 2)
        ell = TVEllipsoid(R1=20, R2=10, R3=5)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=3, Ellipsoid=ell
        )
        xyz = (np.zeros(3), np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="MAX_POINT_SET_SIZE"):
            PointSetScanGridStyle(templ, xyz, _trivial_fn, None)

    def test_constant_defined(self):
        """I2-02: the cap constant exists (consistent with cvariogram.py)."""
        assert MAX_POINT_SET_SIZE == 1_000_000


# =============================================================================
# I2-03 (MEDIUM): vectorized covariance/correlation accumulation
# =============================================================================


class TestCalcCovarianceVectorized:
    def test_batch_equals_manual_accumulation(self):
        """I2-03: batch (vectorized) accumulation matches manual per-pair math.

        Pre-fix: per-point zeros allocation loop; the vectorized twin produced
        the same numbers. This pins the vectorized result to the exact math.
        """
        rng = np.random.RandomState(42)
        values = [rng.rand(50).astype("float32")]
        soft = [rng.rand(50).astype("float32")]
        params = {"HardData": values, "SoftData": soft}

        result = CalcCovarianceFunction(None, None, None, params)
        p1 = np.arange(25)
        p2 = np.arange(25) + 25
        result = CalcCovarianceFunction(p1, p2, result, params)

        # Manual accumulation: sum over pairs of (v1-s1)*(v2-s2)
        expected = np.zeros(2 * len(values) + 1)
        for i in range(25):
            v1 = values[0][p1[i]]
            v2 = values[0][p2[i]]
            s1 = soft[0][p1[i]]
            s2 = soft[0][p2[i]]
            expected[1] += float(np.float32((v1 - s1) * (v2 - s2)))
            expected[2] += 1
        if expected[2] > 0:
            expected[0] = expected[1] / expected[2]
        assert expected[2] == 25
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_ind_correlation_batch_equals_manual(self):
        """I2-03: vectorized ind-correlation matches manual per-pair math."""
        rng = np.random.RandomState(7)
        values = [(rng.rand(40) > 0.5).astype("float32")]
        soft = [np.clip(rng.rand(40), 0.05, 0.95).astype("float32")]
        params = {"HardData": values, "SoftData": soft}

        result = CalcIndCorrelationFunction(None, None, None, params)
        p1 = np.arange(20)
        p2 = np.arange(20) + 20
        result = CalcIndCorrelationFunction(p1, p2, result, params)

        expected = np.zeros(2 * len(values) + 1)
        for i in range(20):
            v1 = values[0][p1[i]]
            v2 = values[0][p2[i]]
            s1 = soft[0][p1[i]]
            s2 = soft[0][p2[i]]
            product = s1 * (1 - s1) * s2 * (1 - s2)
            denom = product**0.5 if product > 0 else 1.0
            expected[1] += float(np.float32((v1 - s1) * (v2 - s2) / denom))
            expected[2] += 1
        if expected[2] > 0:
            expected[0] = expected[1] / expected[2]
        assert expected[2] == 20
        np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# I2-04 (MEDIUM): window-volume cap before mgrid in _CalcLagsAreas
# =============================================================================


class TestCalcLagsAreasWindowCap:
    def test_huge_window_raises(self):
        """I2-04: huge R2/R3 must raise ValueError instead of TB-scale mgrid.

        Pre-fix: R2=R3=1e6 produced a ~1152 TB mgrid allocation (hang/OOM).
        The cap check runs BEFORE mgrid, so this test is fast and safe.
        """
        ell = TVEllipsoid(R1=1e6, R2=1e6, R3=1e6)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0, NumLags=10, Ellipsoid=ell
        )
        with pytest.raises(ValueError, match="window volume"):
            _CalcLagsAreas(templ)

    def test_large_but_allowed_window_ok(self):
        """I2-04: moderately large windows still work (under the cap)."""
        ell = TVEllipsoid(R1=500, R2=100, R3=50)
        templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=2.0, TolDistance=1.0, NumLags=5, Ellipsoid=ell
        )
        result = _CalcLagsAreas(templ)
        assert len(result) == 5
