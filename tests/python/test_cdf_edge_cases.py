import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import ContProperty, SugarboxGrid
    from geo_bsd.cdf import CdfData, calc_cdf
    HPGL_AVAILABLE = True
except ImportError:
    HPGL_AVAILABLE = False


def _make_prop(values, mask=None, grid_shape=None):
    data = np.array(values, dtype='float32')
    if mask is None:
        mask = np.ones(len(data), dtype='uint8')
    else:
        mask = np.array(mask, dtype='uint8')
    prop = ContProperty(data, mask)
    if grid_shape is not None:
        grid = SugarboxGrid(*grid_shape)
        prop.fix_shape(grid)
    return prop


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCalcCdfDuplicateValues:
    def test_all_duplicate_values(self):
        prop = _make_prop([5.0] * 8, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 5.0
        assert cdf.probs[0] == 1.0

    def test_two_groups_of_duplicates(self):
        prop = _make_prop([1.0] * 4 + [2.0] * 4, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 1.0
        assert abs(cdf.probs[0] - 0.5) < 1e-5

    def test_three_groups_of_duplicates(self):
        prop = _make_prop([1.0] * 3 + [2.0] * 3 + [3.0] * 3, grid_shape=(3, 3, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([1.0, 2.0], dtype='float32')
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([1/3, 2/3], dtype='float32')
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCalcCdfSingleValue:
    def test_single_value_one_cell(self):
        prop = _make_prop([42.0], grid_shape=(1, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 42.0
        assert cdf.probs[0] == 1.0

    def test_single_value_many_cells(self):
        prop = _make_prop([7.0] * 27, grid_shape=(3, 3, 3))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 7.0
        assert cdf.probs[0] == 1.0


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCalcCdfAllSame:
    def test_all_same_positive(self):
        prop = _make_prop([100.0] * 12, grid_shape=(3, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 100.0

    def test_all_same_zero(self):
        prop = _make_prop([0.0] * 8, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 0.0

    def test_all_same_float(self):
        prop = _make_prop([3.14] * 8, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert abs(cdf.values[0] - 3.14) < 1e-5


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCalcCdfNegativeValues:
    def test_negative_values(self):
        prop = _make_prop([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, -2.0, 2.0], grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values[0] < 0
        assert np.all(np.diff(cdf.values) >= 0)

    def test_all_negative(self):
        prop = _make_prop([-10.0, -5.0, -1.0, -3.0], grid_shape=(2, 2, 1))
        cdf = calc_cdf(prop)
        assert np.all(cdf.values < 0)
        assert np.all(np.diff(cdf.values) >= 0)

    def test_mixed_positive_negative(self):
        prop = _make_prop([-100.0, -50.0, 0.0, 50.0, 100.0], grid_shape=(5, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values[0] == -100.0
        assert cdf.values[-1] == 50.0
        assert np.all(np.diff(cdf.values) >= 0)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCalcCdfEdgeCases:
    def test_all_masked_raises(self):
        prop = _make_prop([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          mask=[0, 0, 0, 0, 0, 0, 0, 0],
                          grid_shape=(2, 2, 2))
        with pytest.raises(ValueError, match="no informed values"):
            calc_cdf(prop)

    def test_partially_masked(self):
        prop = _make_prop([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          mask=[1, 1, 0, 0, 0, 0, 0, 0],
                          grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 1.0
        assert abs(cdf.probs[0] - 0.5) < 1e-5

    def test_two_unique_values_equal_counts(self):
        prop = _make_prop([10.0] * 6 + [20.0] * 6, grid_shape=(3, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 10.0
        assert abs(cdf.probs[0] - 0.5) < 1e-5

    def test_many_unique_sorted(self):
        np.random.seed(42)
        values = np.random.rand(30).astype('float32') * 100
        prop = _make_prop(list(values), grid_shape=(5, 3, 2))
        cdf = calc_cdf(prop)
        if cdf.values.size > 1:
            assert np.all(np.diff(cdf.values) > 0)

    def test_probs_monotonically_increasing(self):
        np.random.seed(42)
        values = np.random.rand(30).astype('float32') * 100
        prop = _make_prop(list(values), grid_shape=(5, 3, 2))
        cdf = calc_cdf(prop)
        if cdf.probs.size > 1:
            assert np.all(np.diff(cdf.probs) >= 0)

    def test_large_value_range(self):
        prop = _make_prop([1e-10, 1e10], grid_shape=(2, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 1e-10
        assert abs(cdf.probs[0] - 0.5) < 1e-5

    def test_near_equal_floats(self):
        prop = _make_prop([1.0, 1.0000001, 1.0000002], grid_shape=(3, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size >= 1
        assert np.all(np.diff(cdf.values) >= 0)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCdfDataCreation:
    def test_basic_creation(self):
        cdf = CdfData([1.0, 2.0, 3.0], [0.33, 0.66, 1.0])
        assert cdf.values.dtype == np.float32
        assert cdf.probs.dtype == np.float32
        assert len(cdf.values) == 3

    def test_empty_creation(self):
        cdf = CdfData([], [])
        assert cdf.values.size == 0
        assert cdf.probs.size == 0

    def test_single_point(self):
        cdf = CdfData([5.0], [1.0])
        assert cdf.values[0] == 5.0
        assert cdf.probs[0] == 1.0
