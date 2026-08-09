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
        # F-04: last CDF probability must be strictly below 1.0 so the max
        # datum does not map to p=1.0 in the SGS back-transform.
        assert cdf.probs[0] < 1.0

    def test_two_groups_of_duplicates(self):
        prop = _make_prop([1.0] * 4 + [2.0] * 4, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([1.0, 2.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        assert cdf.probs[0] == 0.5
        # T-16: the clamp (cdf.py) pins the tail strictly below 1.0.
        assert cdf.probs[-1] < 1.0

    def test_three_groups_of_duplicates(self):
        prop = _make_prop([1.0] * 3 + [2.0] * 3 + [3.0] * 3, grid_shape=(3, 3, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 3
        expected_values = np.array([1.0, 2.0, 3.0], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        assert cdf.probs[0] == pytest.approx(1 / 3)
        # T-16: the clamp pins the tail strictly below 1.0.
        assert cdf.probs[-1] < 1.0


@pytest.mark.hpgl
class TestCalcCdfSingleValue:
    def test_single_value_one_cell(self):
        prop = _make_prop([42.0], grid_shape=(1, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 1
        assert cdf.values[0] == 42.0
        # F-04: last CDF probability must be strictly below 1.0.
        assert cdf.probs[0] < 1.0


@pytest.mark.hpgl
class TestCalcCdfNegativeValues:
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
        assert cdf.probs[0] == 0.5
        # T-16: the clamp pins the tail strictly below 1.0.
        assert cdf.probs[-1] < 1.0

    def test_many_unique_sorted(self):
        rng = np.random.RandomState(42)
        values = rng.rand(30).astype("float32") * 100
        prop = _make_prop(list(values), grid_shape=(5, 3, 2))
        cdf = calc_cdf(prop)
        # np.unique guarantees sorted output; assert unconditionally (N2-L15).
        assert np.all(np.diff(cdf.values) > 0)


# =============================================================================
# F-N13: multi-value CDF tail assertions (F-M11 verification)
# =============================================================================


@pytest.mark.hpgl
class TestCalcCdfMultiValueTail:
    def test_multi_value_float32_downcast_tail_below_one(self):
        """F-N13a: the float32-downcast tail must be clamped below 1.0.

        F-M11 facet (a): with 3 equal-count values the float64 cumulative sum
        is 0.9999999999999999 (not >= 1.0), which previously bypassed the
        `>= 1.0` clamp and rounded to exactly 1.0f in the float32 downcast —
        re-introducing the F-04 max-datum→median collapse. The clamp must
        fire AFTER the downcast so the stored tail is strictly below 1.0.
        """
        prop = _make_prop([1.0, 2.0, 3.0], grid_shape=(3, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 3
        assert cdf.probs[-1] < 1.0, "float32-downcast tail must be clamped below 1.0"
        assert cdf.probs[-1] == np.nextafter(np.float32(1.0), np.float32(0.0))

    @pytest.mark.slow
    def test_large_grid_monotonic_no_spurious_value_error(self):
        """F-N13b: full_count >= 2^25 must not raise a spurious ValueError.

        F-M11 facet (b): with 2^25 cells the last unique value's count is
        smaller than one float32 ulp, so the second-to-last cumulative
        probability rounds UP to exactly 1.0f; clamping probs[-1] to
        0.99999994f then produced the non-monotonic tail [1.0f, 0.99999994f]
        and a spurious monotonicity ValueError. The fixed clamp caps the
        1.0f suffix so the float32 output stays monotonic and probs[-1] < 1.
        calc_cdf supports flat (1D) property data — no grid needed, so the
        2^25-cell property is not subject to the 1e7 per-axis grid cap.

        T-31 (slow): 2^25 cells ≈ 134 MB float32 + 34 MB mask, peak
        ~168-400 MB during np.unique — machine-freeze risk in the default
        suite (AGENTS.md:28 slow-marker policy).
        """
        n = 2**25
        data = np.zeros(n, dtype="float32")
        data[-1] = 1.0
        prop = ContProperty(data, np.ones(n, dtype="uint8"))
        cdf = calc_cdf(prop)
        assert np.all(np.diff(cdf.probs) >= 0), "float32 CDF must stay monotonic"
        assert cdf.probs[-1] < 1.0

    def test_multi_value_earlier_probs_unchanged(self):
        """F-M11 must not alter earlier (non-tail) probabilities."""
        prop = _make_prop([1.0] * 4 + [2.0] * 4, grid_shape=(2, 2, 2))
        cdf = calc_cdf(prop)
        np.testing.assert_array_almost_equal(cdf.probs, [0.5, 1.0], decimal=5)
        assert cdf.probs[0] == 0.5
        assert cdf.probs[-1] < 1.0

    def test_large_value_range(self):
        prop = _make_prop([1e-10, 1e10], grid_shape=(2, 1, 1))
        cdf = calc_cdf(prop)
        assert cdf.values.size == 2
        expected_values = np.array([1e-10, 1e10], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.values, expected_values)
        assert cdf.probs[0] == 0.5
        # T-16: the clamp pins the tail strictly below 1.0.
        assert cdf.probs[-1] < 1.0

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
    def test_length_mismatch_raises(self):
        """CdfData raises ValueError when values and probs have mismatched lengths."""
        with pytest.raises(ValueError, match="values length.*must match.*probs length"):
            CdfData([1.0, 2.0, 3.0], [0.5, 1.0])

    # ---- F-209: CdfData non-monotonic probability/value validation ----

    def test_non_monotonic_probs_raises(self):
        """F-209: CdfData raises ValueError for non-monotonically increasing probabilities."""
        with pytest.raises(ValueError, match="probabilities must be monotonically"):
            CdfData([1.0, 2.0, 3.0], [0.5, 0.3, 1.0])

    def test_non_monotonic_values_raises(self):
        """F-209: CdfData raises ValueError for non-monotonically increasing values."""
        with pytest.raises(ValueError, match="values must be monotonically"):
            CdfData([3.0, 2.0, 1.0], [0.33, 0.66, 1.0])
