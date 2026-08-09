"""
Comprehensive tests for HPGL simulation algorithms:
- Sequential Gaussian Simulation (SGS)
- Sequential Indicator Simulation (SIS)

Tests cover:
- Reproducibility with same seed (non-vacuous 10%-uninformed pins)
- Kriging type variations (SK vs OK for SGS)
- Statistical properties validation (CDF range, variogram structure,
  indicator proportions over the original uninformed cells)
- LVM support (both SGS and SIS)
- use_harddata parameter
- mask parameter (selective simulation with the mask overlapping
  originally-uninformed cells)
- Result validation (shape, indicator count, statistics)
- F-133 kriging-failure bounded-output contract

(T-15: use_correlogram tests deleted — the parameter is inert in the
non-LVM path, sis.py:370-376. T-33: vacuous small-grid tests deleted.
D-23/D-24: multiple-realizations pair deleted as redundant with the kept
different-seed tests.)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "geo_bsd"))

try:
    from geo_bsd.cdf import CdfData
    from geo_bsd.geo import ContProperty, CovarianceModel, IndProperty, SugarboxGrid, covariance
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
except (ImportError, OSError):
    # Dummy covariance for module-level parametrize decorators
    # Tests are skipped by @pytest.mark.hpgl when HPGL is unavailable
    class _DummyCovarianceTypes:
        spherical = 0
        exponential = 1
        gaussian = 2

    covariance = _DummyCovarianceTypes()


# =============================================================================
# Fixtures for Simulation Tests
# =============================================================================


@pytest.fixture
def sgs_cdf_data_2threshold():
    """CDF data with 2 thresholds (median IK case)"""
    values = np.array([25.0, 75.0], dtype="float32")
    probs = np.array([0.5, 1.0], dtype="float32")
    return CdfData(values, probs)


@pytest.fixture
def sgs_cdf_data_multi():
    """CDF data with multiple thresholds"""
    values = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype="float32")
    probs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype="float32")
    return CdfData(values, probs)


@pytest.fixture
def sgs_lvm_mean(sample_grid):
    """LVM mean array for SGS (spatially varying mean)"""
    # Create a gradient mean field
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mean = np.zeros((x, y, z), dtype="float32", order="F")
    for i in range(x):
        for j in range(y):
            for k in range(z):
                mean[i, j, k] = 30.0 + 10.0 * (i + j + k) / (x + y + z)
    return mean


@pytest.fixture
def sis_data_3indicator():
    """SIS data for 3-indicator case"""
    data = []
    for _i in range(3):
        data.append(
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
                "radiuses": (5, 5, 3),
                "max_neighbours": 12,
            }
        )
    return data


@pytest.fixture
def sis_data_5indicator():
    """SIS data for 5-indicator case"""
    data = []
    for _i in range(5):
        data.append(
            {
                "cov_model": CovarianceModel(
                    type=covariance.exponential,
                    ranges=(6.0, 6.0, 4.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.05,
                ),
                "radiuses": (6, 6, 4),
                "max_neighbours": 15,
            }
        )
    return data


@pytest.fixture
def sis_lvm_marginal_probs(sample_grid):
    """LVM marginal probabilities for SIS (spatially varying)

    Each cell's probabilities across all categories must sum to ~1.0
    (within PROBABILITY_SUM_TOLERANCE). We compute unnormalized values
    then normalize per-cell to ensure this constraint.
    """
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    # Create 3 spatially varying probability fields, then normalize per cell
    marginal_probs = []
    probs_sum = np.zeros((x, y, z), dtype="float32", order="F")
    for cat in range(3):
        probs = np.zeros((x, y, z), dtype="float32", order="F")
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    probs[i, j, k] = 0.2 + 0.1 * cat + 0.05 * (i / x)
        marginal_probs.append(probs)
        probs_sum += probs
    # Normalize per-cell so each cell sums to 1.0
    for cat in range(3):
        marginal_probs[cat] /= probs_sum
    return marginal_probs


@pytest.fixture
def simulation_mask(sample_grid):
    """Mask for selective simulation.

    Covers the central block (2 <= i < x-2, 2 <= j < y-2) AND the x=0 plane
    where the sample_property/sample_indicator_property uninformed cells live
    (mask[::10] = 0 -> flat indices 0,10,..,490 -> x=0 in F-order reshape).
    The x=0-plane overlap is needed so the mask intersects the originally
    uninformed cells (T-11/F-132): with only the central block, the mask is
    disjoint from every uninformed cell and no simulation runs. Half the
    x=0 plane (j < 5) is left outside the mask so tests can also assert that
    out-of-mask uninformed cells are preserved (mask honored).
    """
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mask = np.zeros((x, y, z), dtype="uint8", order="F")
    # Simulate central region
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if 2 <= i < x - 2 and 2 <= j < y - 2:
                    mask[i, j, k] = 1
    # Overlap with the uninformed x=0 plane (j < 5 half)
    for j in range(min(5, y)):
        for k in range(z):
            mask[0, j, k] = 1
    return mask


# =============================================================================
# SGS - Basic Execution Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationBasic:
    """Test basic SGS execution and parameter handling"""

    def test_sgs_with_2threshold_cdf(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_2threshold
    ):
        """Test SGS with 2-threshold CDF (median case)"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_2threshold,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        # Results should be within CDF value range
        assert np.all(result.data[result.mask > 0] >= 0)  # Non-negative

    def test_sgs_accepts_tuple_prop(self, sample_grid, sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")

        result = sgs_simulation(
            prop=(data, mask),  # Tuple input
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SGS - Reproducibility Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationReproducibility:
    """Test SGS reproducibility"""

    def test_sgs_same_seed_same_result(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS produces identical results with same seed"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    def test_sgs_different_seed_different_result(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS produces different results with different seeds"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)


# =============================================================================
# SGS - Kriging Type Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationKrigingType:
    """Test SGS kriging type parameter (SK vs OK)"""

    def test_sgs_sk_vs_ok_produce_different_results(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SK and OK produce different results"""
        result_sk = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        result_ok = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="ok",
        )

        # SK and OK should produce different results
        assert not np.array_equal(result_sk.data, result_ok.data)

    def test_sgs_invalid_kriging_type_raises_error(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test invalid kriging type raises appropriate error"""
        with pytest.raises(ValueError, match="invalid kriging_type"):
            sgs_simulation(
                prop=sample_property,
                grid=sample_grid,
                cdf_data=sgs_cdf_data_multi,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=sample_covariance_model,
                seed=42,
                kriging_type="invalid",
            )


# =============================================================================
# SGS - LVM (Locally Varying Mean) Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationLVM:
    """Test SGS with Locally Varying Mean"""

    def test_sgs_with_lvm_array(
        self,
        sample_property,
        sample_grid,
        sample_covariance_model,
        sgs_lvm_mean,
        sgs_cdf_data_multi,
    ):
        """Test SGS with LVM mean array"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=sgs_lvm_mean,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape
        assert not np.any(np.isnan(result.data.astype("float64")))
        # F-131: Verify LVM mean parameter influences the simulation.
        # With a spatially varying mean, the result should differ from the
        # same run with auto-computed mean — proving the mean parameter is used.
        result_no_lvm = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )
        # LVM and auto-mean results must differ (mean parameter IS used)
        assert not np.allclose(
            result.data.astype("float64"),
            result_no_lvm.data.astype("float64"),
            rtol=1e-4,
            atol=1e-4,
        ), "LVM mean should produce different result than auto-computed mean"

    def test_sgs_with_scalar_mean(
        self, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with scalar mean value.

        T-10: the original fixture (sample_property, mean ~50) made this test
        vacuous — the auto-computed mean also lands near 50, so the assertion
        passed whether or not the scalar mean was honored. Use a property
        whose data mean (~5) is far below the scalar mean (50.0) so the
        scalar-mean honoring is observable: the simulated-only mean with
        mean=50.0 must exceed the auto-mean run's simulated-only mean.
        """
        rng = np.random.RandomState(42)
        data = rng.rand(500).astype("float32") * 10  # mean ~ 5
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        orig_uninformed = (mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F") == 0)

        result = sgs_simulation(
            prop=prop,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=50.0,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        # F-131/T-10: scalar mean must influence the simulation. With data
        # mean ~5, mean=50.0 must pull the SIMULATED cells' mean above the
        # auto-mean run's (empirically ~16 vs ~13 on the 50 uninformed cells).
        sim_values = result.data.astype("float64")[orig_uninformed]
        auto_result = sgs_simulation(
            prop=prop,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )
        auto_values = auto_result.data.astype("float64")[orig_uninformed]
        assert np.mean(sim_values) > np.mean(auto_values), (
            f"SGS mean=50.0 must pull simulated cells above the auto-mean "
            f"run (got {np.mean(sim_values):.2f} vs {np.mean(auto_values):.2f})"
        )
        assert not np.all(result.data == 0), "SGS scalar mean: result should not be all-zeros"

    def test_sgs_with_mean_none(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with mean=None (auto-computed).

        N2-04 (LOW): this is a weak smoke, not a precise mean-reproduction
        pin. With mask-mutation result.mask is all-ones, so the simulated
        mean below is dominated by the 450 hard-data cells (mean ~50); the
        30% bound only fails for simulated-cell shifts > ~150 (0.1 * shift
        all-cell scaling on 50 simulated cells). Kept as a cheap smoke with
        the claim corrected accordingly.
        """
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        # Smoke claim: auto-computed mean should produce a result whose mean
        # is near the data mean (within 30%); tolerates simulated shifts up
        # to ~150 (hard-data dominance).
        data_mean = np.mean(sample_property.data[sample_property.mask == 1].astype("float64"))
        simulated_values = result.data[result.mask > 0].astype("float64")
        simulated_mean = np.mean(simulated_values)
        assert abs(simulated_mean - data_mean) < max(abs(data_mean) * 0.3, 15.0), (
            f"mean=None: result mean {simulated_mean:.2f} deviates from "
            f"data mean {data_mean:.2f} by more than 30%"
        )


# =============================================================================
# SGS - use_harddata Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationUseHarddata:
    """Test SGS use_harddata parameter"""

    def test_sgs_use_harddata_true(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with use_harddata=True preserves input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=True,
        )

        assert isinstance(result, ContProperty)
        # use_harddata=True must preserve hard data values at informed positions
        informed_mask = sample_property.mask == 1
        np.testing.assert_allclose(
            result.data[informed_mask].astype("float64"),
            sample_property.data[informed_mask].astype("float64"),
            rtol=1e-5,
            err_msg="Hard data values should be preserved with use_harddata=True",
        )

    def test_sgs_use_harddata_false(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with use_harddata=False ignores input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=False,
        )

        assert isinstance(result, ContProperty)
        # With use_harddata=False, output should not match input where informed
        # (output is simulated fresh)
        informed_mask = sample_property.mask == 1
        assert not np.allclose(
            result.data[informed_mask].astype("float64"),
            sample_property.data[informed_mask].astype("float64"),
            rtol=1e-5,
        ), "Result should differ from input with use_harddata=False"


# =============================================================================
# SGS - Mask Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationMask:
    """Test SGS mask parameter for selective simulation"""

    def test_sgs_with_mask(
        self,
        sample_property,
        sample_grid,
        sample_covariance_model,
        sgs_cdf_data_multi,
        simulation_mask,
    ):
        """Test SGS with mask for selective simulation.

        T-11/F-132: the original simulation_mask covered only the central
        block (x=2..7), disjoint from every uninformed cell (all at x=0) — so
        nothing was simulated and the std>0 assertion was armed by hard data.
        The fixture now also permits the x=0 plane (j<5 half), which overlaps
        the originally-uninformed cells. This test asserts:
          (a) in-mask originally-uninformed cells ARE simulated (finite,
              non-trivial spread);
          (b) out-of-mask originally-uninformed cells are byte-identical to
              the input (the mask is honored — a dropped mask fails this).
        """
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mask=simulation_mask,
        )

        assert isinstance(result, ContProperty)
        # Check that result shape matches input
        assert result.data.shape == sample_property.data.shape
        mask_3d = simulation_mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F")
        orig_mask_3d = sample_property.mask.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        orig_data_3d = sample_property.data.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        sim_data = result.data.astype("float64")

        # (a) in-mask originally-uninformed cells must be simulated
        in_mask_uninformed = (mask_3d == 1) & (orig_mask_3d == 0)
        assert np.sum(in_mask_uninformed) > 0, (
            "simulation_mask must overlap the originally-uninformed cells"
        )
        sim_cells = sim_data[in_mask_uninformed]
        assert np.all(np.isfinite(sim_cells)), "Simulated cells must be finite"
        assert np.std(sim_cells) > 0, "Simulated cells should not be uniform"

        # (b) out-of-mask originally-uninformed cells are preserved (mask honored)
        out_mask_uninformed = (mask_3d == 0) & (orig_mask_3d == 0)
        assert np.sum(out_mask_uninformed) > 0, "mask should leave some uninformed cells outside"
        np.testing.assert_array_equal(
            sim_data[out_mask_uninformed],
            orig_data_3d[out_mask_uninformed],
            err_msg="out-of-mask uninformed cells must be preserved (mask honored)",
        )


# =============================================================================
# SGS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationStatistics:
    """Test SGS statistical properties"""

    def test_sgs_result_within_cdf_range(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS results are within CDF value range"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        min_cdf = sgs_cdf_data_multi.values.min()
        max_cdf = sgs_cdf_data_multi.values.max()

        # Check that simulated values are within CDF range (with tolerance)
        simulated_values = result.data[result.mask > 0]
        assert np.all(simulated_values >= min_cdf - 1.0)
        assert np.all(simulated_values <= max_cdf + 1.0)

    def test_sgs_result_variogram_structure(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """C24: SGS output must have positive spatial correlation at short lags.

        T-13/N2-02: result.mask is ALL-ONES after the C++ kernel mutates the
        mask buffer (property_array.h set_at), so the original x-direction
        lag loop sampled the whole grid (mostly hard data) and the γ(1)<γ(5)
        gap was sampling noise. This hardened version samples the ORIGINAL
        uninformed cells (sample_property.mask==0 — the x=0 plane, 50 cells)
        and uses y-direction lags (the uninformed cells all lie in x=0, so
        x-lags have no pairs). Empirically: 45 near (y-lag 1) / 25 far
        (y-lag 5) pairs, γ(1)≈303 < γ(5)≈353 — non-empty and discriminating.
        """
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        sim = result.data.astype("float64")
        grid = sample_grid
        # Select the ORIGINAL uninformed cells (all at x=0 for this fixture).
        orig_uninformed = (sample_property.mask.reshape((grid.x, grid.y, grid.z), order="F") == 0)

        # y-direction lags: near = distance 1, far = distance 5 (≥ range).
        diffs_near = []
        diffs_far = []
        for x in range(grid.x):
            for y in range(grid.y):
                for z in range(grid.z):
                    if not orig_uninformed[x, y, z]:
                        continue
                    if y + 1 < grid.y and orig_uninformed[x, y + 1, z]:
                        diffs_near.append((sim[x, y, z] - sim[x, y + 1, z]) ** 2)
                    if y + 5 < grid.y and orig_uninformed[x, y + 5, z]:
                        diffs_far.append((sim[x, y, z] - sim[x, y + 5, z]) ** 2)

        assert diffs_near and diffs_far, (
            "T-13: expected non-empty lag pairs over the original-uninformed cells"
        )
        gamma_near = 0.5 * np.mean(diffs_near)  # Experimental γ(lag 1)
        gamma_far = 0.5 * np.mean(diffs_far)  # Experimental γ(lag 5)

        # The spherical model with sill=1.0, nugget=0.1, range=5.0 predicts:
        #   γ(h=1) ≈ 0.1 + 0.9 * (1.5*(1/5) - 0.5*(1/5)^3) ≈ 0.1 + 0.267 = 0.367
        #   γ(h=5) ≈ 0.1 + 0.9 * 1.0 = 1.0
        # Short-lag variance must be LESS than long-lag variance
        assert gamma_near < gamma_far, (
            f"γ(lag 1)={gamma_near:.3f} should be < γ(lag 5)={gamma_far:.3f} "
            f"for a positively correlated field"
        )


# =============================================================================
# SGS - Covariance Model Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationCovariance:
    """Test SGS with different covariance models"""

    @pytest.mark.parametrize(
        "cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian]
    )
    def test_sgs_different_covariance_types(
        self, sample_property, sample_grid, sgs_cdf_data_multi, cov_type
    ):
        """Test SGS with different covariance types"""
        cov_model = CovarianceModel(
            type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
        )

        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SIS - Basic Execution Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationBasic:
    """Test basic SIS execution and parameter handling"""

    def test_sis_basic_execution_5indicator(
        self, sample_grid, sis_data_5indicator
    ):
        """Test SIS with 5 indicators.

        Uses a 5-category property matching the 5-indicator data config
        (F-24: indicator_count must match len(data) — a mismatch now raises).

        N2-06: kept as the only non-vacuous 5-indicator SIS execution test
        (2/3-indicator execution is pinned by test_sis_fixes.py F1 arithmetic;
        the parametrized twin is fully-informed and vacuous).
        """
        data = np.random.RandomState(42).randint(0, 5, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = IndProperty(data, mask, 5)
        result = sis_simulation(
            prop=prop,
            grid=sample_grid,
            data=sis_data_5indicator,
            seed=42,
            marginal_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 5

    def test_sis_accepts_tuple_prop(self, sample_grid, sis_data_3indicator):
        """Test SIS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.randint(0, 3, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")

        result = sis_simulation(
            prop=(data, mask, 3),  # Tuple input
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape[0] == 500
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# SIS - Reproducibility Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationReproducibility:
    """Test SIS reproducibility"""

    def test_sis_same_seed_same_result(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces identical results with same seed"""
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    def test_sis_different_seed_different_result(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces different results with different seeds"""
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=999,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)


# =============================================================================
# SIS - LVM (Locally Varying Marginal Probabilities) Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationLVM:
    """Test SIS with Locally Varying Marginal Probabilities"""

    def test_sis_with_lvm_marginal_probs(
        self, sample_indicator_property, sample_grid, sis_data_3indicator, sis_lvm_marginal_probs
    ):
        """Test SIS with LVM marginal probabilities"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=sis_lvm_marginal_probs,
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

    def test_sis_with_scalar_marginal_probs(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with scalar marginal probabilities"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# SIS - use_harddata Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationUseHarddata:
    """Test SIS use_harddata parameter"""

    def test_sis_use_harddata_true(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_harddata=True preserves input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=True,
        )

        assert isinstance(result, IndProperty)
        # use_harddata=True must preserve hard data values at informed positions
        informed_mask = sample_indicator_property.mask == 1
        assert np.array_equal(
            result.data[informed_mask], sample_indicator_property.data[informed_mask]
        ), "Hard data should be preserved with use_harddata=True"

    def test_sis_use_harddata_false(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_harddata=False ignores input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=False,
        )

        assert isinstance(result, IndProperty)
        # With use_harddata=False, output should generally differ from input
        informed_mask = sample_indicator_property.mask == 1
        assert not np.array_equal(
            result.data[informed_mask], sample_indicator_property.data[informed_mask]
        ), "Result should differ from input with use_harddata=False"


# =============================================================================
# SIS - Mask Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationMask:
    """Test SIS mask parameter for selective simulation"""

    def test_sis_with_mask(
        self, sample_indicator_property, sample_grid, sis_data_3indicator, simulation_mask
    ):
        """Test SIS with mask for selective simulation.

        T-11/F-132: the original test was pure smoke (isinstance + shape).
        The simulation_mask fixture now overlaps the originally-uninformed
        cells (x=0 plane half), so this test asserts:
          (a) in-mask originally-uninformed cells ARE simulated (valid
              category values, non-trivial spread);
          (b) out-of-mask originally-uninformed cells are preserved
              (mask honored).
        """
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            mask=simulation_mask,
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == sample_indicator_property.data.shape
        mask_3d = simulation_mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F")
        orig_mask_3d = sample_indicator_property.mask.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        orig_data_3d = sample_indicator_property.data.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )

        # (a) in-mask originally-uninformed cells must be simulated (valid category)
        in_mask_uninformed = ((mask_3d == 1) & (orig_mask_3d == 0)).ravel(order="F")
        assert np.sum(in_mask_uninformed) > 0, (
            "simulation_mask must overlap the originally-uninformed cells"
        )
        sim_cells = result.data[in_mask_uninformed]
        assert np.all(sim_cells < result.indicator_count), (
            "SIS simulated cells must be valid categories"
        )
        assert len(np.unique(sim_cells)) >= 2, "SIS simulated cells should not be uniform"

        # (b) out-of-mask originally-uninformed cells are preserved (mask honored)
        out_mask_uninformed = ((mask_3d == 0) & (orig_mask_3d == 0)).ravel(order="F")
        assert np.sum(out_mask_uninformed) > 0, "mask should leave some uninformed cells outside"
        np.testing.assert_array_equal(
            result.data[out_mask_uninformed],
            orig_data_3d.ravel(order="F")[out_mask_uninformed],
            err_msg="out-of-mask uninformed cells must be preserved (mask honored)",
        )


# =============================================================================
# SIS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationStatistics:
    """Test SIS statistical properties"""

    def test_sis_result_within_indicator_range(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS results are within valid indicator range"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        # All values should be less than indicator_count
        assert np.all(result.data < result.indicator_count)

    def test_sis_indicator_proportions(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """C24: SIS indicator proportions should approximately match marginal_probs.

        T-12/N2-03: result.mask is ALL-ONES after the C++ kernel mutates the
        mask buffer, so the old `result.data[result.mask > 0]` sampled ALL 500
        cells (450 hard + 50 simulated) and passed by hard-data dominance.
        This hardened version samples the ORIGINAL uninformed cells
        (sample_indicator_property.mask == 0 — 50 cells) and asserts the
        simulated proportions follow marginal_probs.

        E-05 (post-fix TEST-UPDATE): seed restored to 42 and the docstring
        corrected to the verified committed-fixture reproduction:
        seed 42 → simulated-only proportions [0.36, 0.36, 0.28], max deviation
        0.06 vs the 0.15 threshold (0.09 headroom). The earlier seed-7 switch
        was made on a wrong empirical basis (the seed-7 reproduction is
        [0.26, 0.30, 0.44], maxdev 0.14 — only 0.01 headroom, statistically
        fragile); seed 42 gives the comfortable margin and the test is robust
        against C++ RNG-consumption or float drift.
        """
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        simulated = result.data[sample_indicator_property.mask == 0]
        n = len(simulated)
        assert n > 0, "original fixture must have uninformed cells to sample"
        target_probs = [0.3, 0.4, 0.3]

        for i, target in enumerate(target_probs):
            actual_prop = np.sum(simulated == i) / n
            assert abs(actual_prop - target) < 0.15, (
                f"Indicator {i}: proportion {actual_prop:.3f} deviates from "
                f"target {target:.3f} by more than 0.15"
            )


# =============================================================================
# SIS - Covariance Model Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationCovariance:
    """Test SIS with different covariance models"""

    @pytest.mark.parametrize(
        "cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian]
    )
    def test_sis_different_covariance_types(self, sample_indicator_property, sample_grid, cov_type):
        """Test SIS with different covariance types"""
        sis_data = []
        for _i in range(3):
            sis_data.append(
                {
                    "cov_model": CovarianceModel(
                        type=cov_type,
                        ranges=(5.0, 5.0, 3.0),
                        angles=(0.0, 0.0, 0.0),
                        sill=1.0,
                        nugget=0.1,
                    ),
                    "radiuses": (5, 5, 3),
                    "max_neighbours": 12,
                }
            )

        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.hpgl
class TestSimulationEdgeCases:
    """Test edge cases and error handling"""

    # F-133: SGS kriging failure test (variance ≤ 0 fallback)
    def test_sgs_kriging_failure_coincident_data(self):
        """F-133: SGS kriging-failure fallback produces bounded finite output.

        T-14: the original fixture was fully-informed (mask all ones), so the
        C++ kernel skipped every node before the kriging call — the fallback
        was never exercised and the assertions ran on the identity clone.
        This hardened version leaves ONE cell uninformed (mask [1,1,1,0]) so
        the kriging path is genuinely invoked (probe: points_calculated=1).
        The output must be finite, bounded (no 9e5-class extreme values), and
        non-zero. Known gap (ADV-M3/N2-03): the degenerate variance ≤ 0 case
        is NOT constructible from this fixture — the 4 data sit at distinct
        grid nodes, so the covariance matrix is non-singular; the test pins
        the invoked-kriging bounded-output contract rather than the
        degenerate-system branch.
        """
        data = np.array([5.0, 5.0, 5.0, 5.0], dtype="float32")
        mask = np.array([1, 1, 1, 0], dtype="uint8")  # one uninformed → kriging invoked
        grid = SugarboxGrid(x=2, y=2, z=1)

        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(2.0, 2.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        cdf_data = CdfData(
            values=np.array([0.0, 10.0], dtype="float32"),
            probs=np.array([0.0, 1.0], dtype="float32"),
        )

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(3, 3, 3),
            max_neighbours=4,
            cov_model=cov_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64"))), (
            "SGS kriging failure: result should not contain NaN"
        )
        assert not np.any(np.isinf(result.data.astype("float64"))), (
            "SGS kriging failure: result should not contain Inf"
        )
        # Fallback should produce values in the normal-score range, not explode
        result_flat = result.data.astype("float64").flatten()
        assert np.all(np.abs(result_flat) < 10.0), (
            f"SGS kriging failure: result values {result_flat} should be bounded "
            f"(expected N(0,1) fallback, not extreme values)"
        )
        # Result should not be all-zeros either
        assert not np.all(result.data == 0), "SGS kriging failure: should produce non-zero values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
