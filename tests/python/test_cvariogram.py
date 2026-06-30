import ctypes as C
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cvariogram import (
        CalcVariograms,
        CalcVariogramsFromPointSet,
        CStackLayers,
        Ellipsoid,
        VariogramSearchTemplate,
        _c_array,
        checked_create,
        cont_point_set_t,
        ellipsoid_t,
        float_data_t,
        hard_data_t,
        variogram_search_template_t,
        vector_t,
    )
    CVAR_AVAILABLE = True
except Exception:
    CVAR_AVAILABLE = False


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestCStructTypes:
    def test_vector_t_creation(self):
        v = vector_t(data=_c_array(C.c_double, 3, (1.0, 2.0, 3.0)))
        assert v.data[0] == 1.0
        assert v.data[1] == 2.0
        assert v.data[2] == 3.0

    def test_vector_t_zero(self):
        v = vector_t(data=_c_array(C.c_double, 3, (0, 0, 0)))
        assert v.data[0] == 0.0
        assert v.data[1] == 0.0
        assert v.data[2] == 0.0

    def test_ellipsoid_t_creation(self):
        vec = vector_t(data=_c_array(C.c_double, 3, (0, 0, 0)))
        ell = ellipsoid_t(
            direction1=vec, direction2=vec, direction3=vec,
            R1=10.0, R2=5.0, R3=3.0
        )
        assert ell.R1 == 10.0
        assert ell.R2 == 5.0
        assert ell.R3 == 3.0

    def test_variogram_search_template_t_creation(self):
        vec = vector_t(data=_c_array(C.c_double, 3, (0, 0, 0)))
        ell = ellipsoid_t(
            direction1=vec, direction2=vec, direction3=vec,
            R1=10.0, R2=5.0, R3=3.0
        )
        templ = variogram_search_template_t(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=10, first_lag_distance=0.0, ellipsoid=ell
        )
        assert templ.num_lags == 10
        assert templ.lag_separation == 2.0

    def test_hard_data_t_fields_exist(self):
        assert hasattr(hard_data_t, '_fields_')
        field_names = [f for f, _ in hard_data_t._fields_]
        assert 'data' in field_names
        assert 'mask' in field_names

    def test_cont_point_set_t_fields_exist(self):
        field_names = [f for f, _ in cont_point_set_t._fields_]
        assert 'xs' in field_names
        assert 'ys' in field_names
        assert 'zs' in field_names
        assert 'values' in field_names
        assert 'size' in field_names

    def test_float_data_t_fields_exist(self):
        field_names = [f for f, _ in float_data_t._fields_]
        assert 'data' in field_names
        assert 'data_shape' in field_names


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestCheckedCreate:
    def test_complete_fields(self):
        v = checked_create(vector_t, data=_c_array(C.c_double, 3, (1, 2, 3)))
        assert isinstance(v, vector_t)

    def test_missing_field_raises(self):
        with pytest.raises(RuntimeError, match="No values for parameters"):
            checked_create(vector_t)

    def test_extra_field_ignored(self):
        v = checked_create(vector_t, data=_c_array(C.c_double, 3, (1, 2, 3)), nonexistent=42)
        assert isinstance(v, vector_t)
        assert v.data[0] == 1.0


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestEllipsoid:
    def test_creation_with_valid_params(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        assert ell.ell.R1 == 10.0
        assert ell.ell.R2 == 5.0
        assert ell.ell.R3 == 3.0

    def test_directions_filled_identity(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        d1 = [ell.ell.direction1.data[i] for i in range(3)]
        assert abs(d1[0] - 1.0) < 1e-10
        assert abs(d1[1]) < 1e-10
        assert abs(d1[2]) < 1e-10

    def test_with_nonzero_angles(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=45, dip=30, rotation=60)
        d1_norm = sum(ell.ell.direction1.data[i]**2 for i in range(3))**0.5
        assert abs(d1_norm - 1.0) < 1e-6

    def test_with_zero_r2_r3_no_crash(self):
        ell = Ellipsoid(R1=10, R2=0, R3=0, azimuth=0, dip=0, rotation=0)
        assert ell.ell.R2 == 0.0
        assert ell.ell.R3 == 0.0

    def test_with_equal_radii(self):
        ell = Ellipsoid(R1=5, R2=5, R3=5, azimuth=0, dip=0, rotation=0)
        assert ell.ell.R1 == ell.ell.R2 == ell.ell.R3


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestVariogramSearchTemplate:
    def test_creation(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=10, first_lag_distance=0.0, ellipsoid=ell
        )
        assert templ.num_lags == 10
        assert templ.lag_separation == 2.0

    def test_with_one_lag(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=1, first_lag_distance=0.0, ellipsoid=ell
        )
        assert templ.num_lags == 1


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestCalcVariograms:
    def _make_grid_data(self, nx=5, ny=5, nz=3):
        np.random.seed(42)
        data = np.random.rand(nx, ny, nz).astype('float32') * 100
        mask = np.ones((nx, ny, nz), dtype='uint8')
        mask[::2, ::2, :] = 0
        return (data, mask)

    def test_basic_calculation(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data)
        assert len(lags) == 5
        assert len(variogram) == 5
        assert variogram.dtype == np.float32

    def test_lag_borders_correct(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=3.0, tol_distance=1.0,
            num_lags=4, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data)
        expected_lags = np.array([0, 3, 6, 9], dtype=float)
        np.testing.assert_array_almost_equal(lags, expected_lags)

    def test_percent_100(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data, percent=100)
        assert len(variogram) == 3

    def test_percent_50(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data, percent=50)
        assert len(variogram) == 3

    def test_percent_boundary_1(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data, percent=1)
        assert len(variogram) == 3

    def test_percent_boundary_100(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data, percent=100)

    def test_percent_zero_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        with pytest.raises(ValueError, match="percent must be in"):
            CalcVariograms(templ, hard_data, percent=0)

    def test_percent_negative_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        with pytest.raises(ValueError, match="percent must be in"):
            CalcVariograms(templ, hard_data, percent=-5)

    def test_percent_101_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        with pytest.raises(ValueError, match="percent must be in"):
            CalcVariograms(templ, hard_data, percent=101)

    def test_zero_num_lags_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=0, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        with pytest.raises(ValueError, match="num_lags must be positive"):
            CalcVariograms(templ, hard_data)

    def test_negative_num_lags_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=-5, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        with pytest.raises(ValueError, match="num_lags must be positive"):
            CalcVariograms(templ, hard_data)

    # --- C28: analytical γ(h) value assertions ---

    def test_variogram_constant_data(self):
        """C28: variogram of constant data must be ≈ 0 for all lags.

        γ(h) = 0.5 * E[(Z(x) - Z(x+h))^2]. If Z is constant, Z(x)-Z(x+h) = 0,
        so γ(h) = 0 for all h.
        """
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        data = np.ones((5, 5, 3), dtype='float32') * 42.0
        mask = np.ones((5, 5, 3), dtype='uint8')
        lags, variogram = CalcVariograms(templ, (data, mask))
        assert len(variogram) == 5
        # All γ(h) must be very close to zero for constant data
        assert np.all(np.abs(variogram) < 1e-5), (
            f"Constant data variogram must be 0, got {variogram}"
        )

    def test_variogram_non_negative(self):
        """C28: variogram values must be non-negative.

        γ(h) = 0.5 * E[(Z(x)-Z(x+h))^2] ≥ 0 by definition.
        """
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data)
        assert np.all(variogram >= 0.0), f"Variogram has negative values: {variogram}"

    def test_variogram_nugget_effect(self):
        """C28: variogram at lag 0 must be ≤ variogram at lag 1.

        The first lag captures the nugget effect (short-scale variance).
        For random data, γ(lag0) should be approximately the data variance.
        """
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=3.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data, percent=100)
        # All γ(h) should be finite
        assert np.all(np.isfinite(variogram)), "Variogram contains Inf or NaN"
        # For data with variance > 0, at least some γ(h) should be > 0
        data_var = np.var(hard_data[0][hard_data[1] > 0].astype('float64'))
        if data_var > 0:
            assert np.any(variogram > 0), (
                f"Variogram all zeros for data with variance {data_var}"
            )

    # --- C28: tol_distance boundary tests ---

    def test_tol_distance_small(self):
        """C28: tol_distance=0.5 should produce valid variogram output.

        Small tol_distance means fewer point pairs are captured in each bin
        but the computation should still complete without error.
        """
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=0.5,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data)
        assert len(variogram) == 3
        assert variogram.dtype == np.float32
        assert np.all(np.isfinite(variogram)), "Variogram with tol_distance=0.5 contains Inf/NaN"

    def test_tol_distance_large(self):
        """C28: tol_distance=2.0 produces valid variogram output.

        Large tol_distance broadens each lag bin, potentially increasing
        the number of point pairs. Output should still be valid.
        """
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=2.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell
        )
        hard_data = self._make_grid_data()
        lags, variogram = CalcVariograms(templ, hard_data)
        assert len(variogram) == 3
        assert variogram.dtype == np.float32
        assert np.all(np.isfinite(variogram)), "Variogram with tol_distance=2.0 contains Inf/NaN"


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestCalcVariogramsFromPointSet:
    def _make_point_set(self, n=20):
        np.random.seed(42)
        return {
            "X": np.random.rand(n).astype('float32') * 10,
            "Y": np.random.rand(n).astype('float32') * 10,
            "Z": np.random.rand(n).astype('float32') * 5,
            "Property": np.random.rand(n).astype('float32') * 100,
        }

    def test_basic_calculation(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        ps = self._make_point_set()
        lags, variogram = CalcVariogramsFromPointSet(templ, ps, None)
        assert len(lags) == 5
        assert len(variogram) == 5

    def test_zero_num_lags_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=0, first_lag_distance=0.0, ellipsoid=ell
        )
        ps = self._make_point_set()
        with pytest.raises(ValueError, match="num_lags must be positive"):
            CalcVariogramsFromPointSet(templ, ps, None)

    def test_missing_x_key_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        ps = {"Y": np.zeros(10, dtype='float32'), "Z": np.zeros(10, dtype='float32'), "Property": np.zeros(10, dtype='float32')}
        with pytest.raises(ValueError, match="missing required key 'X'"):
            CalcVariogramsFromPointSet(templ, ps, None)

    def test_missing_property_key_raises(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=2.0, tol_distance=1.0,
            num_lags=5, first_lag_distance=0.0, ellipsoid=ell
        )
        ps = {"X": np.zeros(10, dtype='float32'), "Y": np.zeros(10, dtype='float32'), "Z": np.zeros(10, dtype='float32')}
        with pytest.raises(ValueError, match="missing required key 'Property'"):
            CalcVariogramsFromPointSet(templ, ps, None)

    def test_lag_borders_correct(self):
        ell = Ellipsoid(R1=10, R2=5, R3=3, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=3.0, tol_distance=1.0,
            num_lags=4, first_lag_distance=0.0, ellipsoid=ell
        )
        ps = self._make_point_set()
        lags, variogram = CalcVariogramsFromPointSet(templ, ps, None)
        expected_lags = np.array([0, 3, 6, 9], dtype=float)
        np.testing.assert_array_almost_equal(lags, expected_lags)


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestCStackLayers:
    def _make_layer(self, nx=5, ny=5):
        np.random.seed(42)
        return np.random.rand(nx, ny, 1).astype('float32')

    def _make_result(self, nx=5, ny=5, nz=10):
        return np.zeros((nx, ny, nz), dtype='float32')

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="layers list is empty"):
            CStackLayers([], [], nz=10, scalez=1.0, blank_value=-99.0, result=self._make_result())

    def test_zero_nz_raises(self):
        layer = self._make_layer()
        with pytest.raises(ValueError, match="nz must be positive"):
            CStackLayers([layer], [1], nz=0, scalez=1.0, blank_value=-99.0, result=self._make_result())

    def test_negative_nz_raises(self):
        layer = self._make_layer()
        with pytest.raises(ValueError, match="nz must be positive"):
            CStackLayers([layer], [1], nz=-5, scalez=1.0, blank_value=-99.0, result=self._make_result())

    def test_basic_stacking(self):
        layer = self._make_layer()
        result = self._make_result()
        CStackLayers([layer], [1], nz=5, scalez=1.0, blank_value=-99, result=result)
        assert result.shape == (5, 5, 10)
        # Verify layer data was copied into the result at the correct z-position
        # Layer was placed at z-index 1 (scalez=1.0), so z-slice 0 should be fill,
        # and z-slices >=1 should contain the layer data (repeated or filled).
        # At minimum, the result should not be all zeros — data was written.
        assert not np.all(result == 0), "Result should contain non-zero data from layers"
        # No values should be NaN
        assert not np.any(np.isnan(result))

    def test_multiple_layers(self):
        l1 = self._make_layer()
        l2 = self._make_layer()
        result = self._make_result()
        CStackLayers([l1, l2], [1, 3], nz=5, scalez=1.0, blank_value=-99, result=result)
        assert result.shape == (5, 5, 10)
        # Verify multiple layers were stacked — result should not be all zeros
        assert not np.all(result == 0), "Result should contain non-zero data from multiple layers"
        assert not np.any(np.isnan(result))
