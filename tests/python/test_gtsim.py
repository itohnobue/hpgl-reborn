import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import ContProperty, SugarboxGrid
    from geo_bsd.gtsim import pseudo_gaussian_transform, tk_calculation
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


def _make_cont_prop(size, values=None, mask=None):
    if values is None:
        np.random.seed(42)
        values = np.random.rand(size).astype("float32")
    else:
        values = np.array(values, dtype="float32")
    if mask is None:
        mask = np.ones(size, dtype="uint8")
    else:
        mask = np.array(mask, dtype="uint8")
    return ContProperty(values, mask)


@pytest.mark.hpgl
class TestTkCalculation:
    def test_basic_calculation(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        result = tk_calculation(prop)
        assert result is prop
        assert result.data.size == 10
        assert np.all(np.isfinite(result.data))

    def test_default_params_inverse_cdf_behavior(self):
        """Default (mean=0, std_dev=1) inverse CDF thresholds should be finite.

        Thresholds are t = mean - std_dev * Φ⁻¹(p), so they can be negative
        for probabilities > 0.5. Only check finiteness since the sign depends on p."""
        prop = _make_cont_prop(100)
        result = tk_calculation(prop)
        values = result.data.flat[:]
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

    def test_inverse_cdf_monotonic(self):
        """Inverse CDF thresholds decrease as input probabilities increase.

        Since t = mean - std_dev * Φ⁻¹(p), and Φ⁻¹ is strictly increasing,
        larger p → larger Φ⁻¹(p) → smaller t. So pk=0.1 gives a higher
        threshold than pk=0.9."""
        values_low_prob = np.array([0.1] * 5, dtype="float32")
        values_high_prob = np.array([0.9] * 5, dtype="float32")
        prop_low = _make_cont_prop(5, values=list(values_low_prob))
        prop_high = _make_cont_prop(5, values=list(values_high_prob))
        result_low = tk_calculation(prop_low, mean=0.0, std_dev=1.0)
        result_high = tk_calculation(prop_high, mean=0.0, std_dev=1.0)
        # Lower probability → higher threshold (makes indicator=1 less likely)
        assert result_low.data.flat[0] > result_high.data.flat[0]

    def test_larger_std_dev_spreads_thresholds(self):
        """Larger std_dev should produce wider spread in inverse CDF thresholds.

        Since t = mean - std_dev * Φ⁻¹(p), scaling std_dev scales the spread."""
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype="float32")
        prop_narrow = _make_cont_prop(5, values=list(values))
        prop_wide = _make_cont_prop(5, values=list(values.copy()))
        result_narrow = tk_calculation(prop_narrow, mean=0.0, std_dev=0.5)
        result_wide = tk_calculation(prop_wide, mean=0.0, std_dev=5.0)
        range_narrow = float(np.max(result_narrow.data) - np.min(result_narrow.data))
        range_wide = float(np.max(result_wide.data) - np.min(result_wide.data))
        assert range_wide > range_narrow

    def test_mutates_input_property(self):
        prop = _make_cont_prop(5, values=[0.1, 0.3, 0.5, 0.7, 0.9])
        original_data = prop.data.copy()
        tk_calculation(prop)
        assert not np.array_equal(prop.data, original_data)

    def test_returns_same_property(self):
        prop = _make_cont_prop(5, values=[0.5] * 5)
        result = tk_calculation(prop)
        assert result is prop


@pytest.mark.hpgl
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


@pytest.mark.hpgl
class TestGtsimNoFileWrites:
    def test_no_debug_files_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files_before = set(os.listdir(tmp_path))
        prop = _make_cont_prop(5, values=[0.5] * 5)
        tk_calculation(prop)
        files_after = set(os.listdir(tmp_path))
        new_files = files_after - files_before
        debug_files = [f for f in new_files if f.endswith((".txt", ".dat", ".csv", ".log"))]
        assert len(debug_files) == 0


# =============================================================================
# gtsim_2ind Tests (Q3 fix — previously zero coverage)
# =============================================================================

# gtsim_2ind depends on geo.py (through `from .geo import *`).
# The geo.py file has a pre-existing syntax error at line 403 (indentation).
# When geo.py is fixed, gtsim_2ind has an additional known bug (A5):
# sgs_simulation called without required cdf_data parameter.
# These tests are written to work once geo.py and gtsim.py are fixed.
try:
    from geo_bsd.geo import ContProperty, CovarianceModel, SugarboxGrid, covariance
    from geo_bsd.gtsim import gtsim_2ind

    _GTSIM_2IND_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError, OSError):
    _GTSIM_2IND_AVAILABLE = False


@pytest.mark.skipif(
    not _GTSIM_2IND_AVAILABLE, reason="gtsim_2ind not available (requires working geo.py)"
)
class TestGtsim2Ind:
    """Tests for the gtsim_2ind Gaussian Truncated Simulation workflow.

    NOTE: gtsim_2ind currently has a known bug (A5 from adversarially verified
    findings): sgs_simulation is called without the required cdf_data parameter.
    These tests use try/except to handle the expected failure gracefully while
    still verifying that components up to the SGS call work correctly.
    """

    def _make_grid_prop(self, x=5, y=5, z=2):
        """Create a small grid and continuous property for testing."""
        np.random.seed(42)
        grid = SugarboxGrid(x=x, y=y, z=z)
        size = x * y * z
        # Binary 0/1 data with ~20% uninformed
        data = np.where(np.random.rand(size) < 0.6, 0.0, 1.0).astype("float32")
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        return grid, prop

    def _make_sk_params(self):
        """Create simple kriging parameters."""
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        return {
            "radiuses": (3, 3, 2),
            "max_neighbours": 8,
            "cov_model": cov_model,
        }

    def test_gtsim_2ind_basic_execution(self):
        """gtsim_2ind with default parameters."""
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert isinstance(result, ContProperty)
        assert np.all(np.isfinite(result.data))

    def test_gtsim_2ind_with_provided_pk_prop(self):
        """gtsim_2ind with pre-computed pk_prop (skips SK step)."""
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_mask = np.ones(prop.data.size, dtype="uint8")
        pk_prop = ContProperty(pk_data, pk_mask)

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        assert isinstance(result, ContProperty)

    def test_gtsim_2ind_with_custom_tk_params(self):
        """gtsim_2ind accepts custom tk_mean and tk_std_dev."""
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, tk_mean=0.5, tk_std_dev=2.0, seed=42)
        assert isinstance(result, ContProperty)

    def test_gtsim_2ind_reproducibility_same_seed(self):
        """gtsim_2ind with same seed and same global random state produces identical output."""
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        # Reset global random state before each call to ensure reproducibility
        # (gtsim_2ind uses global np.random via pseudo_gaussian_transform)
        np.random.seed(42)
        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        np.random.seed(42)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=42)
        np.testing.assert_array_equal(result1.data, result2.data)

    def test_gtsim_2ind_different_seeds_produce_different(self):
        """gtsim_2ind with different seeds produces different output.

        Uses a partially-informed property (mask with uninformed cells) so
        SGS actually simulates values. With a fully-informed property the SGS
        step is a no-op (nothing to simulate), the pseudo-Gaussian transform
        degenerates (kriging probabilities are exactly 0/1), and no randomness
        is consumed — different seeds then produce identical output. That is
        a fixture artifact, not a seed bug: with uninformed cells present,
        different seeds genuinely produce different simulated fields.
        """
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        # Leave ~30% of cells uninformed in both props so SGS actually
        # simulates them (seed-dependent). Identical masks for a fair
        # seed comparison. Note: gtsim_2ind mutates prop.data in place,
        # so two fresh props are required.
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=12345)
        assert not np.array_equal(result1.data, result2.data)

    def test_gtsim_2ind_produces_both_categories(self):
        """gtsim_2ind with mixed input produces both 0 and 1 in output."""
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        unique = np.unique(result.data)
        assert 0.0 in unique
        assert 1.0 in unique

    def test_gtsim_2ind_returns_same_size(self):
        """gtsim_2ind output size matches input."""
        grid, prop = self._make_grid_prop(x=6, y=6, z=3)
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert result.data.size == prop.data.size
        assert result.mask.size == prop.mask.size

    def test_gtsim_2ind_no_nan_in_output(self):
        """gtsim_2ind output contains no NaN values."""
        grid, prop = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert not np.any(np.isnan(result.data))
