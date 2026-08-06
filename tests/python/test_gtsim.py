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


@pytest.mark.hpgl
class TestCdfDataInverse:
    """Unit tests for CdfData.inverse — the F-02 data-space threshold mapper.

    Mirrors the C++ non_parametric_cdf_2_t::inverse semantics
    (non_parametric_cdf.h:263-292): probabilities below the first cumulative
    probability map to the smallest value, above the last map to the largest
    value, and interior probabilities are linearly interpolated between the
    bracketing (value, prob) pairs (std::lower_bound over the probs array).
    """

    def _cdf(self):
        from geo_bsd.cdf import CdfData

        return CdfData(
            values=np.array([0.0, 10.0, 20.0], dtype="float32"),
            probs=np.array([0.2, 0.5, 1.0], dtype="float32"),
        )

    def test_below_first_prob_maps_to_smallest_value(self):
        c = self._cdf()
        assert c.inverse(0.1) == 0.0
        assert c.inverse(0.2) == 0.0  # first prob inclusive (lower_bound)

    def test_above_last_prob_maps_to_largest_value(self):
        c = self._cdf()
        assert c.inverse(1.0) == 20.0
        assert c.inverse(1.5) == 20.0

    def test_interior_linear_interpolation(self):
        c = self._cdf()
        # 0.35 between (0.0, 0.2) and (10.0, 0.5): x1 + (x2-x1)/(y2-y1)*(p-y1)
        # = 0 + 10/0.3 * 0.15 = 5.0
        assert abs(c.inverse(0.35) - 5.0) < 1e-6
        # 0.9 between (10.0, 0.5) and (20.0, 1.0): 10 + 10/0.5*0.4 = 18.0
        assert abs(c.inverse(0.9) - 18.0) < 1e-6

    def test_vectorized(self):
        c = self._cdf()
        out = c.inverse(np.array([0.1, 0.35, 0.9, 1.5]))
        np.testing.assert_allclose(out, [0.0, 5.0, 18.0, 20.0], atol=1e-6)

    def test_scalar_returns_scalar(self):
        c = self._cdf()
        assert np.isscalar(c.inverse(0.5))

    def test_empty_cdf_raises(self):
        from geo_bsd.cdf import CdfData

        c = CdfData(values=np.array([], dtype="float32"),
                    probs=np.array([], dtype="float32"))
        with pytest.raises(ValueError, match="empty CDF"):
            c.inverse(0.5)

    def test_roundtrip_with_calc_cdf(self):
        """F-02 end-to-end building block: inverse(prob(value)) == value for
        informed data (CDF built by calc_cdf from a property)."""
        from geo_bsd.cdf import calc_cdf

        rng = np.random.RandomState(7)
        values = (rng.rand(200) * 10).astype("float32")
        prop = ContProperty(values, np.ones(200, dtype="uint8"))
        c = calc_cdf(prop)
        # For each unique value the cumulative prob inverts back to ~value.
        for v in np.unique(values):
            p = float(np.mean(values <= v))
            recovered = c.inverse(p)
            assert abs(recovered - v) < 1.0


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
    """Tests for the gtsim_2ind Gaussian Truncated Simulation workflow."""

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

    def test_gtsim_2ind_clamps_overshoot_pk_prop_without_mutating_caller(self):
        """F-M15 regression: out-of-[0,1] pk probabilities are clamped (not
        rejected) and the caller's pk_prop.data array is never mutated.

        The 3ad77ee production-check commit changed the hard ValueError reject
        into a clamp but added no regression test (its commit message claims
        "gtsim out-of-[0,1] rejection" — no such test exists). The clamp was
        also applied through a ravel view (`np.clip(..., out=pk_flat)`), which
        permanently altered the caller's array in place (1D arrays are both
        C- and F-contiguous, so ContProperty's require("F") returns the same
        object). F-M15 clamps a copy: the caller's original array must keep
        its overshoot values.
        """
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_data[0] = -0.05
        pk_data[1] = 1.05
        orig = pk_data.copy()
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        assert isinstance(result, ContProperty)
        assert np.all(np.isfinite(result.data))
        # The caller's original array must not have been written in place.
        np.testing.assert_array_equal(pk_data, orig)

    def test_gtsim_2ind_with_custom_tk_params(self):
        """gtsim_2ind accepts custom tk_mean and tk_std_dev."""
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, tk_mean=0.5, tk_std_dev=2.0, seed=42)
        assert isinstance(result, ContProperty)

    def test_gtsim_2ind_reproducibility_same_seed(self):
        """gtsim_2ind with same seed and same global random state produces identical output.

        Partial masks (D-09/B-02): with fully-informed props the SGS step is a
        no-op and the comparison is trivially identical; uninformed cells make
        the reproducibility assertion meaningful.
        """
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

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
        # seed comparison. (2-M-12: gtsim_2ind no longer mutates the
        # caller's prop, but fresh props keep the comparison clean.)
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=12345)
        assert not np.array_equal(result1.data, result2.data)

    def test_gtsim_2ind_produces_both_categories(self):
        """gtsim_2ind with mixed input produces both 0 and 1 in output.

        Partial masks (D-09/B-02): uninformed cells are simulated, so the
        output's 0/1 mix genuinely comes from the simulation path, not just
        the hard-data copy.
        """
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        unique = np.unique(result.data)
        assert 0.0 in unique
        assert 1.0 in unique

    def test_gtsim_2ind_returns_same_size(self):
        """gtsim_2ind output size matches input (partial mask — D-09)."""
        grid, prop = self._make_grid_prop(x=6, y=6, z=3)
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert result.data.size == prop.data.size
        assert result.mask.size == prop.mask.size

    def test_gtsim_2ind_no_nan_in_output(self):
        """gtsim_2ind output contains no NaN values (partial mask — D-09)."""
        grid, prop = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert not np.any(np.isnan(result.data))

    # =========================================================================
    # F-02 (HIGH): truncation must compare in the SAME space as the SGS output
    # =========================================================================

    def test_gtsim_2ind_pk_05_proportion_roughly_half(self):
        """F-02: with pk=0.5 everywhere, the facies-1 proportion must be ~0.5.

        The C++ SGS kernel back-transforms the simulated standard-normal
        field into DATA space through the in-scope empirical CDF
        (sequential_gaussian_simulation.cpp back_transform →
        transform_cdf_p(output, gaussian_cdf_t(), ncdf)), so prop1 is a
        data-space value. Pre-fix the truncation compared that data-space
        output against the NORMAL-SCORE threshold tk = -Φ⁻¹(0.5) = 0, so
        every non-negative data-space cell classified as facies 1 →
        proportion 1.0 (live repro). Post-fix the threshold is mapped to
        data space via the same CDF (tk_data = F⁻¹(Φ(tk))), restoring
        monotone equivalence with the Gaussian-space comparison → ~0.5.
        """
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        # Partial masks (D-09/B-02): uninformed cells are simulated through
        # the pk threshold so the ~0.5 proportion is a real simulation
        # outcome, not a hard-data copy artifact.
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        frac1 = float(np.mean(result.data == 1.0))
        assert 0.2 < frac1 < 0.8, (
            f"F-02: pk=0.5 must give ~0.5 facies-1 proportion, got {frac1} "
            f"(pre-fix data-space vs normal-score comparison gave 1.0)"
        )

    def test_gtsim_2ind_non_default_tk_params_do_not_distort(self):
        """II-39: non-default tk_mean/tk_std_dev must not distort proportions.

        tk_calculation returns tk = tk_mean - tk_std_dev·Φ⁻¹(p); the engine
        simulates a STANDARD-normal field, so the effective threshold is
        (tk - tk_mean)/tk_std_dev = -Φ⁻¹(p) — the tk params must normalize
        away. Pre-fix, tk_mean=5.0/tk_std_dev=2.0 with pk=0.5 gave threshold
        tk=5.0 → every data-space cell < 5 → proportion 0.0. Post-fix the
        normalized threshold gives ~0.5, identical to the default params.
        """
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(
            grid, prop, sk_params, do_sk=False, pk_prop=pk_prop,
            seed=42, tk_mean=5.0, tk_std_dev=2.0,
        )
        frac1 = float(np.mean(result.data == 1.0))
        assert 0.2 < frac1 < 0.8, (
            f"II-39: non-default tk params must not distort the ~0.5 "
            f"proportion, got {frac1} (pre-fix tk_mean=5 gave 0.0)"
        )

    def test_gtsim_2ind_default_and_non_default_tk_agree(self):
        """II-39 control: same seed, default vs non-default tk params must
        produce the same facies field (params normalized away)."""
        grid, prop1 = self._make_grid_prop(x=8, y=8, z=4)
        _, prop2 = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        # Partial masks (D-09/B-02): with fully-informed props the SGS step
        # is a no-op and the two runs trivially agree.
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        pk_data = np.full(prop1.data.size, 0.5, dtype="float32")
        pk_prop1 = ContProperty(pk_data, np.ones(prop1.data.size, dtype="uint8"))
        pk_prop2 = ContProperty(pk_data.copy(), np.ones(prop2.data.size, dtype="uint8"))

        np.random.seed(42)
        r1 = gtsim_2ind(grid, prop1, sk_params, do_sk=False, pk_prop=pk_prop1, seed=42)
        np.random.seed(42)
        r2 = gtsim_2ind(
            grid, prop2, sk_params, do_sk=False, pk_prop=pk_prop2,
            seed=42, tk_mean=5.0, tk_std_dev=2.0,
        )
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_gtsim_2ind_low_pk_gives_few_facies1(self):
        """F-02 monotonicity: pk=0.1 everywhere must give far fewer facies-1
        cells than pk=0.9 everywhere (the data-space threshold mapping is
        monotone in pk).

        E2-32: the fixtures use PARTIAL masks so SGS actually simulates the
        uninformed cells — a fully-informed property degenerates (every cell
        is hard data, the pseudo-Gaussian transform is deterministic, and pk
        no longer drives the classification: 0.40234375 == 0.40234375 above).
        Uninformed cells get simulated from the pk threshold, so the
        monotonicity is measurable again.
        """
        grid, prop1 = self._make_grid_prop(x=8, y=8, z=4)
        _, prop2 = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        # Leave ~70% of cells uninformed so SGS simulates them (identical
        # masks for a fair pk comparison — 2-M-12 pattern, :389-392).
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        pk_low = ContProperty(np.full(prop1.data.size, 0.1, dtype="float32"),
                              np.ones(prop1.data.size, dtype="uint8"))
        pk_high = ContProperty(np.full(prop2.data.size, 0.9, dtype="float32"),
                               np.ones(prop2.data.size, dtype="uint8"))

        r_low = gtsim_2ind(grid, prop1, sk_params, do_sk=False, pk_prop=pk_low, seed=42)
        r_high = gtsim_2ind(grid, prop2, sk_params, do_sk=False, pk_prop=pk_high, seed=42)
        frac_low = float(np.mean(r_low.data == 1.0))
        frac_high = float(np.mean(r_high.data == 1.0))
        # Measured gap with this partial-mask fixture is ~0.23; assert a
        # margin safely below it so the monotonicity claim is meaningful but
        # not fragile to seed/rounding shifts.
        assert frac_high > frac_low + 0.15, (
            f"F-02: pk=0.9 must give more facies-1 than pk=0.1, "
            f"got {frac_low} vs {frac_high}"
        )
