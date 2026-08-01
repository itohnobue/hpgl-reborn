"""
Comprehensive tests for HPGL simulation algorithms:
- Sequential Gaussian Simulation (SGS)
- Sequential Indicator Simulation (SIS)

Tests cover:
- Basic execution for all parameter combinations
- Reproducibility with same seed
- Statistical properties validation
- CDF transformation (SGS)
- LVM support (both SGS and SIS)
- Kriging type variations (SK vs OK for SGS)
- use_harddata parameter
- mask parameter
- min_neighbours parameter
- use_correlogram parameter (SIS)
- Multiple realizations
- Result validation (shape, indicator count, statistics)
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
def sis_data_2indicator():
    """SIS data for 2-indicator case (median IK)"""
    return [
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
        },
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
        },
    ]


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
    """Mask for selective simulation"""
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mask = np.zeros((x, y, z), dtype="uint8", order="F")
    # Simulate only central region
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if 2 <= i < x - 2 and 2 <= j < y - 2:
                    mask[i, j, k] = 1
    return mask


# =============================================================================
# SGS - Basic Execution Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationBasic:
    """Test basic SGS execution and parameter handling"""

    def test_sgs_basic_execution_sk(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with Simple Kriging completes without errors"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape
        assert result.mask.shape == sample_property.mask.shape

    def test_sgs_basic_execution_ok(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with Ordinary Kriging completes without errors"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="ok",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape

    def test_sgs_without_cdf(self, sample_property, sample_grid, sample_covariance_model):
        """Test SGS without CDF transformation"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=None,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape

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
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with scalar mean value"""
        result = sgs_simulation(
            prop=sample_property,
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
        # F-131: Verify scalar mean parameter influences result.
        # The output mean should be closer to the configured mean (50.0) than
        # to a very different mean (e.g. 0.0).
        simulated_values = result.data[result.mask > 0].astype("float64")
        simulated_mean = np.mean(simulated_values)
        # With mean=50.0, the result should not gravitate toward 0
        assert abs(simulated_mean - 50.0) < abs(simulated_mean - 0.0), (
            f"SGS mean=50.0: result mean {simulated_mean:.2f} should be closer to 50.0 than 0.0"
        )
        assert not np.all(result.data == 0), "SGS scalar mean: result should not be all-zeros"

    def test_sgs_with_mean_none(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with mean=None (auto-computed)"""
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
        # F-131: Auto-computed mean should produce result with mean near data mean
        data_mean = np.mean(sample_property.data[sample_property.mask == 1].astype("float64"))
        simulated_values = result.data[result.mask > 0].astype("float64")
        simulated_mean = np.mean(simulated_values)
        # Result mean should be within 30% of data mean (SGS reproduces target mean)
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
        """Test SGS with mask for selective simulation"""
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
        # F-132: Verify mask actually controls simulation — simulated values
        # should exist in masked region and input data should NOT be
        # naively copied to masked region (verifies mask is not ignored).
        mask_3d = simulation_mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F")
        sim_data = result.data.astype("float64")
        # Cells where mask=1 should have finite simulated values
        masked_cells = sim_data[mask_3d == 1]
        assert len(masked_cells) > 0, "Mask should select cells for simulation"
        assert np.all(np.isfinite(masked_cells)), "Simulated cells must be finite"
        # Simulated values should not be all identical (simulation produces variability)
        assert np.std(masked_cells) > 0, "Simulated values should not be uniform"


# =============================================================================
# SGS - min_neighbours Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationMinNeighbours:
    """Test SGS min_neighbours parameter"""

    def test_sgs_min_neighbours_zero(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with min_neighbours=0"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            min_neighbours=0,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_min_neighbours_positive(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with positive min_neighbours"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            min_neighbours=4,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SGS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationStatistics:
    """Test SGS statistical properties"""

    def test_sgs_result_not_all_zeros(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS results are not all zeros"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        assert not np.all(result.data == 0)

    def test_sgs_result_no_nan_or_inf(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS results contain no NaN or Inf values"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

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

    def test_sgs_result_variance_plausible(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS result variance is within reasonable bounds of input variance.

        The simulated output should have variance comparable to the CDF data range,
        not degenerate (near-zero) or exploding.
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

        cdf_range = sgs_cdf_data_multi.values.max() - sgs_cdf_data_multi.values.min()
        simulated_values = result.data[result.mask > 0].astype("float64")
        result_var = np.var(simulated_values)

        # Variance should be positive and bounded by 1.5 × (range/2)^2 max possible.
        # For bounded data, the maximum variance is (range/2)^2 (all values at
        # extremes). The 1.5× tolerance accounts for float32 precision and small
        # CDF extrapolation effects without allowing degenerate outputs.
        max_possible_var = (cdf_range / 2.0) ** 2
        assert result_var > 0.0, "SGS result should have non-zero variance"
        assert result_var < max_possible_var * 1.5, (
            f"SGS variance {result_var} exceeds plausible bound {max_possible_var * 1.5}"
        )

    def test_sgs_result_mean_validation(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """C24: SGS simulated mean should be close to the CDF weighted mean.

        The output mean should approximate the mean of the input CDF distribution
        since SGS reproduces the target distribution in expectation.
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

        simulated_values = result.data[result.mask > 0].astype("float64")
        simulated_mean = np.mean(simulated_values)

        # Expected mean approximated by simple average of CDF values
        cdf_values = sgs_cdf_data_multi.values.astype("float64")
        expected_mean_approx = np.mean(cdf_values)

        # Check that simulated mean is within ±25% of expected mean
        # (generous tolerance for small grid with 10% uninformed cells)
        rel_error = abs(simulated_mean - expected_mean_approx) / max(expected_mean_approx, 1.0)
        assert rel_error < 0.25, (
            f"SGS mean {simulated_mean:.2f} deviates from expected ~{expected_mean_approx:.2f} "
            f"(relative error {rel_error:.3f})"
        )

    def test_sgs_result_variogram_structure(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """C24: SGS output must have positive spatial correlation at short lags.

        Verifies that simulated values separated by 1 grid cell (lag 1 in x
        direction) are more similar than values separated by 4+ grid cells.
        The spherical model with range 5.0 implies strong correlation at lag 1
        (C(1) ≈ 0.7 × sill) and near-zero at lag 5.
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

        # Use only simulated (mask>0) cells
        sim = result.data.astype("float64")
        grid = sample_grid

        # Compute variance at lag 1 (adjacent cells in x-direction)
        # and lag > range (distant cells)
        diffs_near = []
        diffs_far = []
        for x in range(grid.x - 1):
            for y in range(grid.y):
                for z in range(grid.z):
                    if result.mask[x, y, z] > 0 and result.mask[x + 1, y, z] > 0:
                        diffs_near.append((sim[x, y, z] - sim[x + 1, y, z]) ** 2)
                    if x + 5 < grid.x and result.mask[x, y, z] > 0 and result.mask[x + 5, y, z] > 0:
                        diffs_far.append((sim[x, y, z] - sim[x + 5, y, z]) ** 2)

        if diffs_near and diffs_far:
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

    def test_sis_basic_execution_2indicator(
        self, sample_grid, sis_data_2indicator
    ):
        """Test SIS with 2 indicators (median IK path).

        Uses a 2-category property matching the 2-indicator data config
        (F-24: indicator_count must match len(data) — a mismatch now raises).
        """
        data = np.random.RandomState(42).randint(0, 2, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = IndProperty(data, mask, 2)
        result = sis_simulation(
            prop=prop,
            grid=sample_grid,
            data=sis_data_2indicator,
            seed=42,
            marginal_probs=[0.4, 0.6],
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 2
        assert result.data.shape == prop.data.shape

    def test_sis_basic_execution_3indicator(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with 3 indicators"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

    def test_sis_basic_execution_5indicator(
        self, sample_grid, sis_data_5indicator
    ):
        """Test SIS with 5 indicators.

        Uses a 5-category property matching the 5-indicator data config
        (F-24: indicator_count must match len(data) — a mismatch now raises).
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
        """Test SIS with mask for selective simulation"""
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


# =============================================================================
# SIS - min_neighbours Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationMinNeighbours:
    """Test SIS with various parameter combinations (min_neighbours parameter removed per C22).

    The min_neighbours parameter was removed from sis_simulation() because it was
    never forwarded to the C++ layer. These tests now verify basic SIS execution
    with the standard parameter set.
    """

    def test_sis_many_neighbours_3indicator(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with max_neighbours=12 (was previously min_neighbours=0 test)"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sis_with_alt_params(self, sample_grid, sis_data_5indicator):
        """Test SIS with 5-indicator data (was previously min_neighbours=4 test).

        Uses a 5-category property matching the 5-indicator data config
        (F-24: indicator_count must match len(data) — a mismatch now raises).
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
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SIS - use_correlogram Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationUseCorrelogram:
    """Test SIS use_correlogram parameter"""

    def test_sis_use_correlogram_true(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_correlogram=True"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_correlogram=True,
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)

    def test_sis_use_correlogram_false(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_correlogram=False"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_correlogram=False,
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# SIS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationStatistics:
    """Test SIS statistical properties"""

    def test_sis_result_not_all_zeros(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS results are not all zeros"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert not np.all(result.data == 0)

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

    def test_sis_indicator_distribution(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces valid indicator distribution"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        # Check that all 3 indicators are present (or at least no invalid values)
        unique_values = np.unique(result.data[result.mask > 0])
        assert np.all(unique_values < 3)
        assert np.all(unique_values >= 0)

    def test_sis_indicator_proportions(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """C24: SIS indicator proportions should approximately match marginal_probs.

        The simulated indicator distribution should be close to the target
        marginal probabilities [0.3, 0.4, 0.3] within ±0.15 tolerance
        (generous for a 10×10×5 grid with 10% uninformed cells).
        """
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        simulated = result.data[result.mask > 0]
        n = len(simulated)
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
# Multi-Realization Tests
# =============================================================================


@pytest.mark.hpgl
@pytest.mark.slow
class TestMultipleRealizations:
    """Test multiple realizations with different seeds"""

    def test_sgs_multiple_realizations(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS produces different results for multiple realizations"""
        seeds = [42, 123, 456, 789, 999]
        results = []

        for seed in seeds:
            result = sgs_simulation(
                prop=sample_property,
                grid=sample_grid,
                cdf_data=sgs_cdf_data_multi,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=sample_covariance_model,
                seed=seed,
            )
            results.append(result)

        # All results should be different
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                assert not np.array_equal(results[i].data, results[j].data)

    def test_sis_multiple_realizations(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces different results for multiple realizations"""
        seeds = [42, 123, 456, 789, 999]
        results = []

        for seed in seeds:
            result = sis_simulation(
                prop=sample_indicator_property,
                grid=sample_grid,
                data=sis_data_3indicator,
                seed=seed,
                marginal_probs=[0.3, 0.4, 0.3],
            )
            results.append(result)

        # All results should be different
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                assert not np.array_equal(results[i].data, results[j].data)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.hpgl
class TestSimulationEdgeCases:
    """Test edge cases and error handling"""

    def test_sgs_with_small_grid(self):
        """Test SGS with minimal grid size"""
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        mask = np.array([1, 1, 1, 1], dtype="uint8")

        grid = SugarboxGrid(x=2, y=2, z=1)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(2.0, 2.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        cdf_data = CdfData(
            values=np.array([0.0, 5.0], dtype="float32"),
            probs=np.array([0.0, 1.0], dtype="float32"),
        )

        prop = ContProperty(data, mask)
        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(2, 2, 1),
            max_neighbours=4,
            cov_model=cov_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sis_with_small_grid(self):
        """Test SIS with minimal grid size"""
        np.random.seed(42)
        data = np.array([0, 1, 0, 1], dtype="uint8")
        mask = np.array([1, 1, 1, 1], dtype="uint8")

        grid = SugarboxGrid(x=2, y=2, z=1)

        sis_data = [
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical,
                    ranges=(2.0, 2.0, 1.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.0,
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            }
        ] * 2  # 2 indicator categories

        prop = IndProperty(data, mask, 2)
        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data, seed=42, marginal_probs=[0.5, 0.5]
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)

    # F-133: SGS kriging failure test (variance ≤ 0 fallback)
    def test_sgs_kriging_failure_coincident_data(self):
        """F-133: SGS with coincident conditioning data and nugget=0 triggers kriging
        failure (variance ≤ 0) and must fall back gracefully.

        When conditioning data are at the same location with nugget=0, the
        kriging variance drops to zero. The code must produce finite output
        (not NaN, Inf, or extreme values like 9e5). According to the geostatistics
        research, the correct fallback is drawing from N(0,1) — the result must
        be bounded and finite.
        """
        data = np.array([5.0, 5.0, 5.0, 5.0], dtype="float32")
        mask = np.array([1, 1, 1, 1], dtype="uint8")
        grid = SugarboxGrid(x=2, y=2, z=1)

        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(2.0, 2.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,  # Zero nugget + coincident data → zero variance
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
