import numpy as np
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import ContProperty, SugarboxGrid
    from geo_bsd.gtsim import tk_calculation, pseudo_gaussian_transform
    HPGL_AVAILABLE = True
except ImportError:
    HPGL_AVAILABLE = False


def _make_cont_prop(size, values=None, mask=None):
    if values is None:
        np.random.seed(42)
        values = np.random.rand(size).astype('float32')
    else:
        values = np.array(values, dtype='float32')
    if mask is None:
        mask = np.ones(size, dtype='uint8')
    else:
        mask = np.array(mask, dtype='uint8')
    return ContProperty(values, mask)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestTkCalculation:
    def test_basic_calculation(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        result = tk_calculation(prop)
        assert result is prop
        assert result.data.size == 10
        assert np.all(np.isfinite(result.data))

    def test_default_params_gaussian_shape(self):
        prop = _make_cont_prop(100)
        result = tk_calculation(prop)
        values = result.data.flat[:]
        assert np.all(values >= 0)
        assert np.all(np.isfinite(values))

    def test_custom_mean(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        result = tk_calculation(prop, mean=0.5)
        assert np.all(np.isfinite(result.data))

    def test_custom_std_dev(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        result = tk_calculation(prop, std_dev=2.0)
        assert np.all(np.isfinite(result.data))

    def test_zero_std_dev_raises(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        with pytest.raises(ValueError, match="std_dev must be positive"):
            tk_calculation(prop, std_dev=0)

    def test_negative_std_dev_raises(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        with pytest.raises(ValueError, match="std_dev must be positive"):
            tk_calculation(prop, std_dev=-1.0)

    def test_gaussian_pdf_peak_at_mean(self):
        values_at_mean = np.array([0.0] * 5, dtype='float32')
        values_away = np.array([5.0] * 5, dtype='float32')
        prop_mean = _make_cont_prop(5, values=list(values_at_mean))
        prop_away = _make_cont_prop(5, values=list(values_away))
        result_mean = tk_calculation(prop_mean, mean=0.0, std_dev=1.0)
        result_away = tk_calculation(prop_away, mean=0.0, std_dev=1.0)
        assert result_mean.data.flat[0] > result_away.data.flat[0]

    def test_larger_std_dev_spreads_values(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype='float32')
        prop_narrow = _make_cont_prop(5, values=list(values))
        prop_wide = _make_cont_prop(5, values=list(values.copy()))
        result_narrow = tk_calculation(prop_narrow, mean=2.0, std_dev=0.5)
        result_wide = tk_calculation(prop_wide, mean=2.0, std_dev=5.0)
        range_narrow = float(np.max(result_narrow.data) - np.min(result_narrow.data))
        range_wide = float(np.max(result_wide.data) - np.min(result_wide.data))
        assert range_wide < range_narrow

    def test_mutates_input_property(self):
        prop = _make_cont_prop(5, values=[0.1, 0.3, 0.5, 0.7, 0.9])
        original_data = prop.data.copy()
        tk_calculation(prop)
        assert not np.array_equal(prop.data, original_data)

    def test_returns_same_property(self):
        prop = _make_cont_prop(5, values=[0.5] * 5)
        result = tk_calculation(prop)
        assert result is prop


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestPseudoGaussianTransform:
    def test_binary_zeros_transformed(self):
        np.random.seed(42)
        prop = _make_cont_prop(10, values=[0.0] * 10)
        pk_prop = _make_cont_prop(10, values=[0.5] * 10)
        result = pseudo_gaussian_transform(prop, pk_prop)
        assert result is prop
        for v in result.data.flat:
            assert 0.0 <= v < 0.5

    def test_binary_ones_transformed(self):
        np.random.seed(42)
        prop = _make_cont_prop(10, values=[1.0] * 10)
        pk_prop = _make_cont_prop(10, values=[0.5] * 10)
        result = pseudo_gaussian_transform(prop, pk_prop)
        assert result is prop
        for v in result.data.flat:
            assert 0.5 <= v <= 1.0

    def test_mixed_binary(self):
        np.random.seed(42)
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        prop = _make_cont_prop(8, values=values)
        pk_prop = _make_cont_prop(8, values=[0.5] * 8)
        result = pseudo_gaussian_transform(prop, pk_prop)
        zeros_mask = np.array(values) == 0.0
        ones_mask = np.array(values) == 1.0
        assert np.all(result.data.flat[zeros_mask] < 0.5)
        assert np.all(result.data.flat[ones_mask] >= 0.5)

    def test_non_binary_values_unchanged(self):
        np.random.seed(42)
        values = [0.3, 0.7, 0.4, 0.6]
        prop = _make_cont_prop(4, values=values)
        pk_prop = _make_cont_prop(4, values=[0.5] * 4)
        original = prop.data.copy()
        result = pseudo_gaussian_transform(prop, pk_prop)
        np.testing.assert_array_equal(result.data, original)

    def test_returns_same_property_object(self):
        prop = _make_cont_prop(5, values=[0.0, 1.0, 0.0, 1.0, 0.0])
        pk_prop = _make_cont_prop(5, values=[0.5] * 5)
        result = pseudo_gaussian_transform(prop, pk_prop)
        assert result is prop

    def test_different_pk_thresholds(self):
        np.random.seed(42)
        prop = _make_cont_prop(10, values=[0.0] * 10)
        pk_low = _make_cont_prop(10, values=[0.1] * 10)
        result = pseudo_gaussian_transform(prop, pk_low)
        for v in result.data.flat:
            assert 0.0 <= v < 0.1


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestGtsimNoFileWrites:
    def test_no_debug_files_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files_before = set(os.listdir(tmp_path))
        prop = _make_cont_prop(5, values=[0.5] * 5)
        tk_calculation(prop)
        files_after = set(os.listdir(tmp_path))
        new_files = files_after - files_before
        debug_files = [f for f in new_files if f.endswith(('.txt', '.dat', '.csv', '.log'))]
        assert len(debug_files) == 0
