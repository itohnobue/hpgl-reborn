import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import CdfData, calc_cdf
    from geo_bsd.geo import ContProperty, SugarboxGrid
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


def _make_prop(values, mask=None, grid_shape=None):
    data = np.array(values, dtype="float32")
    if mask is None:
        mask = np.ones(len(data), dtype="uint8")
    else:
        mask = np.array(mask, dtype="uint8")
    prop = ContProperty(data, mask)
    if grid_shape is not None:
        grid = SugarboxGrid(*grid_shape)
        prop.fix_shape(grid)
    return prop


@pytest.mark.hpgl
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
        assert cdf.values.size == 2
        expected_values = np.array([1.0, 2.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([0.5, 1.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)

    def test_three_groups_of_duplicates(self):
        prop = _make_prop([1.0] * 3 + [2.0] * 3 + [3.0] * 3, grid_shape=(3, 3, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 3
        expected_values = np.array([1.0, 2.0, 3.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([1 / 3, 2 / 3, 1.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)


@pytest.mark.hpgl
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


@pytest.mark.hpgl
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


@pytest.mark.hpgl
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
        assert cdf.values[-1] == 100.0
        assert np.all(np.diff(cdf.values) >= 0)


@pytest.mark.hpgl
class TestCalcCdfEdgeCases:
    def test_all_masked_raises(self):
        prop = _make_prop(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            mask=[0, 0, 0, 0, 0, 0, 0, 0],
            grid_shape=(2, 2, 2),
        )
        with pytest.raises(ValueError, match="no informed values"):
            calc_cdf(prop)

    def test_partially_masked(self):
        prop = _make_prop(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            mask=[1, 1, 0, 0, 0, 0, 0, 0],
            grid_shape=(2, 2, 2),
        )
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([1.0, 2.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([0.5, 1.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)

    def test_two_unique_values_equal_counts(self):
        prop = _make_prop([10.0] * 6 + [20.0] * 6, grid_shape=(3, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([10.0, 20.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([0.5, 1.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)

    def test_many_unique_sorted(self):
        np.random.seed(42)
        values = np.random.rand(30).astype("float32") * 100
        prop = _make_prop(list(values), grid_shape=(5, 3, 2))
        cdf = calc_cdf(prop)
        if cdf.values.size > 1:
            assert np.all(np.diff(cdf.values) > 0)

    def test_probs_monotonically_increasing(self):
        np.random.seed(42)
        values = np.random.rand(30).astype("float32") * 100
        prop = _make_prop(list(values), grid_shape=(5, 3, 2))
        cdf = calc_cdf(prop)
        if cdf.probs.size > 1:
            assert np.all(np.diff(cdf.probs) >= 0)

    def test_large_value_range(self):
        prop = _make_prop([1e-10, 1e10], grid_shape=(2, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([1e-10, 1e10], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        expected_probs = np.array([0.5, 1.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs, expected_probs, decimal=5)

    def test_near_equal_floats(self):
        prop = _make_prop([1.0, 1.0000001, 1.0000002], grid_shape=(3, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size >= 1
        assert np.all(np.diff(cdf.values) >= 0)

    def test_all_nan_informed_raises(self):
        """ContProperty construction raises ValueError when all values are NaN."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            _make_prop([np.nan, np.nan, np.nan, np.nan], grid_shape=(2, 2, 1))

    def test_mixed_nan_and_finite(self):
        """ContProperty construction raises ValueError when data contains NaN mixed with finite."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            _make_prop([np.nan, 5.0, np.nan, 15.0], grid_shape=(2, 2, 1))


@pytest.mark.hpgl
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

    def test_length_mismatch_raises(self):
        """CdfData raises ValueError when values and probs have mismatched lengths."""
        with pytest.raises(ValueError, match="values length.*must match.*probs length"):
            CdfData([1.0, 2.0, 3.0], [0.5, 1.0])

    def test_length_mismatch_empty_vs_nonempty(self):
        """CdfData raises ValueError when one array is empty and the other is not."""
        with pytest.raises(ValueError):
            CdfData([1.0, 2.0], [])

    # ---- F-209: CdfData non-monotonic probability/value validation ----

    def test_non_monotonic_probs_raises(self):
        """F-209: CdfData raises ValueError for non-monotonically increasing probabilities."""
        with pytest.raises(ValueError, match="probabilities must be monotonically"):
            CdfData([1.0, 2.0, 3.0], [0.5, 0.3, 1.0])

    def test_non_monotonic_values_raises(self):
        """F-209: CdfData raises ValueError for non-monotonically increasing values."""
        with pytest.raises(ValueError, match="values must be monotonically"):
            CdfData([3.0, 2.0, 1.0], [0.33, 0.66, 1.0])
