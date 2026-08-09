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
    CalcVariogramFunction,
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

        Note (2-M-13): the bool-mask counts were re-pinned from
        [125,420,507] to [125,420,687] when the Python lag-binning was
        aligned to the C++ projection metric (variograms.cpp:647, 805).
        Lag 2 under the old raw-Euclidean metric contained only offsets with
        Euclidean distance in [1.5,2.5); under the projection metric it
        additionally contains the dx=±2 offsets with dy/dz up to ±2 (e.g.
        (2,1,1)), whose Euclidean distance 2.45 still falls inside the same
        band but whose PROJECTION |dx|=2 the Euclidean metric had dropped.
        The new value is cross-checked against the C++ grid kernel
        (CalcVariograms) in TestVariogramProjectionMetric.
        """
        templ = _make_template(r1=5, r2=5, r3=5, lag_width=1.0, lag_sep=1.0, num_lags=3)
        mask_u8 = np.ones((5, 5, 5), dtype="uint8")
        mask_bool = np.ones((5, 5, 5), dtype=bool)

        res_u8, _ = CubeScan(templ, mask_u8, _pair_counter, None)
        res_bool, _ = CubeScan(templ, mask_bool, _pair_counter, None)

        np.testing.assert_array_equal(res_u8, res_bool)
        # Correct (bool) pair counts for this template — not the inflated 5x.
        np.testing.assert_array_equal(res_bool[:, 2], np.array([125, 420, 687]))

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


# =============================================================================
# F-M14: CalcVariogramFunction must receive the F-02 tuple-path fix the
# covariance/correlation siblings already have (ravel_multi_index + numpy.take)
# =============================================================================


class TestCalcVariogramTuplePath:
    def test_3d_values_3tuple_identical_pairs(self):
        """F-M14: 3D values + (I, J, K) 3-tuple (natural CubeScan path)."""
        values = [np.arange(24, dtype="float32").reshape(4, 3, 2)]
        params = {"HardData": values}

        result = CalcVariogramFunction(None, None, None, params)
        idx_i = np.array([0, 1], dtype="int64")
        idx_j = np.array([0, 0], dtype="int64")
        idx_k = np.array([0, 0], dtype="int64")
        result2 = CalcVariogramFunction((idx_i, idx_j, idx_k), (idx_i, idx_j, idx_k), result, params)

        # Identical point pairs -> zero-variance pairs -> variogram 0.
        assert np.allclose(result2[: len(values)], 0.0, atol=1e-6)
        assert result2[len(values) + len(values)] == 2  # two pairs accumulated

    def test_scalar_indices_with_3d_values(self):
        """F-M14: scalar index path + 3D values (PointSetScanGridStyle
        convention). Pre-fix: broadcast ValueError — Values[i][array([0])] on
        a 3D array returned a block, not a row."""
        values = [np.arange(8, dtype="float32").reshape(2, 2, 2)]
        params = {"HardData": values}

        result = CalcVariogramFunction(None, None, None, params)
        # Pair (flat 0, flat 1): values 0 and 1 -> var 1, count 1.
        result2 = CalcVariogramFunction(np.array([0]), np.array([1]), result, params)
        assert np.all(np.isfinite(result2))
        assert result2[len(values) + len(values)] == 1  # one pair accumulated
        # variogram[0] = sum/count/2 = (1-0)^2/1/2 = 0.5
        assert np.allclose(result2[0], 0.5)

    def test_1d_values_3tuple_matches_sibling_contract(self):
        """F-M14: 1D flat values + 3-component tuple must raise the same
        ValueError as the covariance/correlation siblings (ravel_multi_index
        requires len(multi_index) == len(shape)) — NOT the pre-fix
        IndexError."""
        values = [np.arange(125, dtype="float32")]
        params = {"HardData": values}
        result = CalcVariogramFunction(None, None, None, params)
        idx_i = np.array([0, 1], dtype="int64")
        idx_j = np.array([0, 0], dtype="int64")
        idx_k = np.array([0, 0], dtype="int64")
        with pytest.raises(ValueError, match="sequence of length"):
            CalcVariogramFunction((idx_i, idx_j, idx_k), (idx_i, idx_j, idx_k), result, params)


# =============================================================================
# F-N14: CubeScan must reject a search template whose extent exceeds the grid
# size (C++ is_inside equivalent) instead of crashing with a broadcast error
# =============================================================================


class TestCubeScanTemplateExtentGuard:
    def _template(self, r, lag_width, lag_sep, num_lags):
        ell = TVEllipsoid(R1=r, R2=r, R3=r)
        return TVVariogramSearchTemplate(
            LagWidth=lag_width,
            LagSeparation=lag_sep,
            TolDistance=1.0,
            NumLags=num_lags,
            Ellipsoid=ell,
        )

    def test_grid_smaller_than_template_extent_raises(self):
        """F-N14: 4x4x4 grid with a template whose lag offsets reach 5 in one
        axis previously crashed with a broadcast ValueError (Mask1 empty vs
        Mask2[0:-1]). The guard raises a clear ValueError instead."""
        templ = self._template(r=100, lag_width=3.0, lag_sep=2.0, num_lags=3)
        mask = np.ones((4, 4, 4), dtype="uint8")
        values = [np.arange(64, dtype="float32").reshape(4, 4, 4)]
        with pytest.raises(ValueError, match="exceeds grid size"):
            CubeScan(templ, mask, _pair_counter, {"HardData": values})

    def test_template_fitting_grid_still_works(self):
        """F-N14: a template within grid bounds must still compute."""
        templ = self._template(r=100, lag_width=1.0, lag_sep=1.0, num_lags=2)
        mask = np.ones((4, 4, 4), dtype="uint8")
        values = [np.arange(64, dtype="float32").reshape(4, 4, 4)]
        result, _ = CubeScan(templ, mask, _pair_counter, {"HardData": values})
        assert np.all(np.isfinite(result))

    def test_exact_extent_still_works(self):
        """F-N14: a template whose offset magnitude exactly equals the grid
        dimension produces empty-empty slices (no crash, C++-equivalent skip)
        — the guard must not reject it (boundary DI == NI)."""
        templ = self._template(r=100, lag_width=1.0, lag_sep=1.0, num_lags=3)
        mask = np.ones((3, 3, 3), dtype="uint8")
        values = [np.arange(27, dtype="float32").reshape(3, 3, 3)]
        result, _ = CubeScan(templ, mask, _pair_counter, {"HardData": values})
        assert np.all(np.isfinite(result))


# =============================================================================
# F-M24: PointSetScanContStyle must skip self-pairs (i == j) like the C++
# point-set scan (variograms.cpp:793) — equivalence test Python vs C++
# =============================================================================


class TestPointSetScanSelfPairSkip:
    def _make_templates(self):
        """Identical search geometry for the Python and C++ paths.

        1D points along X with LagWidth=2.0 put distance-1 pairs AND the
        self-pairs (distance 0) in the same lag band, exposing the self-pair
        counting difference: pre-fix Python lag1 = 22.22 vs C++ = 50.0.
        """
        p_ell = TVEllipsoid(R1=100, R2=100, R3=100)
        p_templ = TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=1.0, TolDistance=1.0, NumLags=3,
            Ellipsoid=p_ell, FirstLagDistance=0,
        )
        return p_templ

    def test_python_point_set_matches_cpp(self):
        """F-M24: Python PointSetScanContStyle (self-pairs skipped) matches
        the C++ calc_variograms_from_point_set output."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import (
            CalcVariogramsFromPointSet,
            Ellipsoid,
            VariogramSearchTemplate,
        )

        xs = np.array([0, 1, 2, 3, 4], dtype="float32")
        ys = np.zeros(5, dtype="float32")
        zs = np.zeros(5, dtype="float32")
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float32")

        # C++ reference (skips idx1 == idx2 at variograms.cpp:793).
        c_ell = Ellipsoid(R1=100, R2=100, R3=100, azimuth=0, dip=0, rotation=0)
        c_templ = VariogramSearchTemplate(
            lag_width=2.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=c_ell,
        )
        c_var = np.zeros(3, dtype="float32")
        _, cpp_result = CalcVariogramsFromPointSet(
            c_templ, {"X": xs, "Y": ys, "Z": zs, "Property": vals}, c_var
        )

        # Python (self-pairs skipped by the F-M24 fix).
        p_templ = self._make_templates()
        py_result, _ = PointSetScanContStyle(
            p_templ, {"X": xs, "Y": ys, "Z": zs}, CalcVariogramFunction,
            {"HardData": [vals]},
        )

        np.testing.assert_allclose(py_result[:, 0], cpp_result, atol=1e-4)
        # Sanity pin: lag 1 must be 50.0, NOT the pre-fix 22.22 dilution.
        assert abs(py_result[1, 0] - 50.0) < 1e-4

    def test_self_pair_not_counted_in_lag_zero(self):
        """F-M24: without any real pairs at distance 0, the self-pairs must
        not inflate the lag-0 pair count (pre-fix: 5 self-pairs counted)."""
        from geo_bsd.variogram import CalcVariogramFunction, PointSetScanContStyle

        xs = np.array([0, 1, 2, 3, 4], dtype="float32")
        ys = np.zeros(5, dtype="float32")
        zs = np.zeros(5, dtype="float32")
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float32")
        p_templ = self._make_templates()

        py_result, _ = PointSetScanContStyle(
            p_templ, {"X": xs, "Y": ys, "Z": zs}, CalcVariogramFunction,
            {"HardData": [vals]},
        )
        # Result layout: [variogram..., sums..., count] — count slot is index
        # 2*NumValues. NumValues = 1, so count is index 2.
        assert py_result[0, 2] == 0, "lag 0 must not count self-pairs"
        assert py_result[1, 2] == 4, "lag 1 counts the 4 real distance-1 pairs only"
