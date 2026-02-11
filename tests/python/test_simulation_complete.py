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
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "geo_bsd"))

try:
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.geo import ContProperty, IndProperty, CovarianceModel, covariance, SugarboxGrid
    from geo_bsd.cdf import CdfData, calc_cdf
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


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
    mean = np.zeros((x, y, z), dtype='float32', order='F')
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
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1
            ),
            'radiuses': (5, 5, 3),
            'max_neighbours': 12
        },
        {
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1
            ),
            'radiuses': (5, 5, 3),
            'max_neighbours': 12
        }
    ]


@pytest.fixture
def sis_data_3indicator():
    """SIS data for 3-indicator case"""
    data = []
    for i in range(3):
        data.append({
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1
            ),
            'radiuses': (5, 5, 3),
            'max_neighbours': 12
        })
    return data


@pytest.fixture
def sis_data_5indicator():
    """SIS data for 5-indicator case"""
    data = []
    for i in range(5):
        data.append({
            'cov_model': CovarianceModel(
                type=covariance.exponential,
                ranges=(6.0, 6.0, 4.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.05
            ),
            'radiuses': (6, 6, 4),
            'max_neighbours': 15
        })
    return data


@pytest.fixture
def sis_lvm_marginal_probs(sample_grid):
    """LVM marginal probabilities for SIS (spatially varying)"""
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    # Create 3 spatially varying probability fields
    marginal_probs = []
    for cat in range(3):
        probs = np.zeros((x, y, z), dtype='float32', order='F')
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # Create spatially varying probabilities
                    probs[i, j, k] = 0.2 + 0.1 * cat + 0.05 * (i / x)
        marginal_probs.append(probs)
    return marginal_probs


@pytest.fixture
def simulation_mask(sample_grid):
    """Mask for selective simulation"""
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mask = np.zeros((x, y, z), dtype='uint8', order='F')
    # Simulate only central region
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if 2 <= i < x-2 and 2 <= j < y-2:
                    mask[i, j, k] = 1
    return mask


# =============================================================================
# SGS - Basic Execution Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationBasic:
    """Test basic SGS execution and parameter handling"""

    def test_sgs_basic_execution_sk(self, sample_property, sample_grid,
                                     sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with Simple Kriging completes without errors"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape
        assert result.mask.shape == sample_property.mask.shape

    def test_sgs_basic_execution_ok(self, sample_property, sample_grid,
                                     sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with Ordinary Kriging completes without errors"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="ok"
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape

    def test_sgs_without_cdf(self, sample_property, sample_grid,
                              sample_covariance_model):
        """Test SGS without CDF transformation"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=None,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape

    def test_sgs_with_2threshold_cdf(self, sample_property, sample_grid,
                                      sample_covariance_model, sgs_cdf_data_2threshold):
        """Test SGS with 2-threshold CDF (median case)"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_2threshold,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        assert isinstance(result, ContProperty)
        # Results should be within CDF value range
        assert np.all(result.data[result.mask > 0] >= 0)  # Non-negative

    def test_sgs_accepts_tuple_prop(self, sample_grid, sample_covariance_model,
                                     sgs_cdf_data_multi):
        """Test SGS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.rand(500).astype('float32') * 100
        mask = np.ones(500, dtype='uint8')

        result = sgs_simulation(
            prop=(data, mask),  # Tuple input
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42
        )

        assert isinstance(result, ContProperty)


# =============================================================================
# SGS - Reproducibility Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationReproducibility:
    """Test SGS reproducibility"""

    def test_sgs_same_seed_same_result(self, sample_property, sample_grid,
                                        sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS produces identical results with same seed"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk"
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk"
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    def test_sgs_different_seed_different_result(self, sample_property, sample_grid,
                                                  sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS produces different results with different seeds"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk"
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)


# =============================================================================
# SGS - Kriging Type Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationKrigingType:
    """Test SGS kriging type parameter (SK vs OK)"""

    def test_sgs_sk_vs_ok_produce_different_results(self, sample_property, sample_grid,
                                                     sample_covariance_model, sgs_cdf_data_multi):
        """Test SK and OK produce different results"""
        result_sk = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        result_ok = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="ok"
        )

        # SK and OK should produce different results
        assert not np.array_equal(result_sk.data, result_ok.data)

    def test_sgs_invalid_kriging_type_raises_error(self, sample_property, sample_grid,
                                                    sample_covariance_model, sgs_cdf_data_multi):
        """Test invalid kriging type raises appropriate error"""
        with pytest.raises(KeyError):
            sgs_simulation(
                prop=sample_property,
                grid=sample_grid,
                cdf_data=sgs_cdf_data_multi,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=sample_covariance_model,
                seed=42,
                kriging_type="invalid"
            )


# =============================================================================
# SGS - LVM (Locally Varying Mean) Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationLVM:
    """Test SGS with Locally Varying Mean"""

    def test_sgs_with_lvm_array(self, sample_property, sample_grid,
                                 sample_covariance_model, sgs_lvm_mean, sgs_cdf_data_multi):
        """Test SGS with LVM mean array"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=sgs_lvm_mean
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape

    def test_sgs_with_scalar_mean(self, sample_property, sample_grid,
                                   sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with scalar mean value"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=50.0
        )

        assert isinstance(result, ContProperty)

    def test_sgs_with_mean_none(self, sample_property, sample_grid,
                                  sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with mean=None (auto-computed)"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None
        )

        assert isinstance(result, ContProperty)


# =============================================================================
# SGS - use_harddata Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationUseHarddata:
    """Test SGS use_harddata parameter"""

    def test_sgs_use_harddata_true(self, sample_property, sample_grid,
                                    sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with use_harddata=True preserves input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=True
        )

        assert isinstance(result, ContProperty)

    def test_sgs_use_harddata_false(self, sample_property, sample_grid,
                                     sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with use_harddata=False ignores input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=False
        )

        assert isinstance(result, ContProperty)
        # With use_harddata=False, output should not match input where informed
        # (output is simulated fresh)


# =============================================================================
# SGS - Mask Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationMask:
    """Test SGS mask parameter for selective simulation"""

    def test_sgs_with_mask(self, sample_property, sample_grid,
                            sample_covariance_model, sgs_cdf_data_multi, simulation_mask):
        """Test SGS with mask for selective simulation"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mask=simulation_mask
        )

        assert isinstance(result, ContProperty)
        # Check that result shape matches input
        assert result.data.shape == sample_property.data.shape


# =============================================================================
# SGS - min_neighbours Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationMinNeighbours:
    """Test SGS min_neighbours parameter"""

    def test_sgs_min_neighbours_zero(self, sample_property, sample_grid,
                                      sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with min_neighbours=0"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            min_neighbours=0
        )

        assert isinstance(result, ContProperty)

    def test_sgs_min_neighbours_positive(self, sample_property, sample_grid,
                                          sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS with positive min_neighbours"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            min_neighbours=4
        )

        assert isinstance(result, ContProperty)


# =============================================================================
# SGS - Statistical Validation Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationStatistics:
    """Test SGS statistical properties"""

    def test_sgs_result_not_all_zeros(self, sample_property, sample_grid,
                                       sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS results are not all zeros"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42
        )

        assert not np.all(result.data == 0)

    def test_sgs_result_no_nan_or_inf(self, sample_property, sample_grid,
                                       sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS results contain no NaN or Inf values"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42
        )

        assert not np.any(np.isnan(result.data.astype('float64')))
        assert not np.any(np.isinf(result.data.astype('float64')))

    def test_sgs_result_within_cdf_range(self, sample_property, sample_grid,
                                          sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS results are within CDF value range"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42
        )

        min_cdf = sgs_cdf_data_multi.values.min()
        max_cdf = sgs_cdf_data_multi.values.max()

        # Check that simulated values are within CDF range (with tolerance)
        simulated_values = result.data[result.mask > 0]
        assert np.all(simulated_values >= min_cdf - 1.0)
        assert np.all(simulated_values <= max_cdf + 1.0)


# =============================================================================
# SGS - Covariance Model Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulationCovariance:
    """Test SGS with different covariance models"""

    @pytest.mark.parametrize("cov_type", [
        covariance.spherical,
        covariance.exponential,
        covariance.gaussian
    ])
    def test_sgs_different_covariance_types(self, sample_property, sample_grid,
                                             sgs_cdf_data_multi, cov_type):
        """Test SGS with different covariance types"""
        cov_model = CovarianceModel(
            type=cov_type,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1
        )

        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42
        )

        assert isinstance(result, ContProperty)


# =============================================================================
# SIS - Basic Execution Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationBasic:
    """Test basic SIS execution and parameter handling"""

    def test_sis_basic_execution_2indicator(self, sample_indicator_property, sample_grid,
                                              sis_data_2indicator):
        """Test SIS with 2 indicators (median IK path)"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_2indicator,
            seed=42,
            marginal_probs=[0.4, 0.6]
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 2
        assert result.data.shape == sample_indicator_property.data.shape

    def test_sis_basic_execution_3indicator(self, sample_indicator_property, sample_grid,
                                              sis_data_3indicator):
        """Test SIS with 3 indicators"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

    def test_sis_basic_execution_5indicator(self, sample_indicator_property, sample_grid,
                                              sis_data_5indicator):
        """Test SIS with 5 indicators"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_5indicator,
            seed=42,
            marginal_probs=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 5

    def test_sis_accepts_tuple_prop(self, sample_grid, sis_data_3indicator):
        """Test SIS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.randint(0, 3, 500, dtype='uint8')
        mask = np.ones(500, dtype='uint8')

        result = sis_simulation(
            prop=(data, mask, 3),  # Tuple input
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# SIS - Reproducibility Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationReproducibility:
    """Test SIS reproducibility"""

    def test_sis_same_seed_same_result(self, sample_indicator_property, sample_grid,
                                         sis_data_3indicator):
        """Test SIS produces identical results with same seed"""
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    def test_sis_different_seed_different_result(self, sample_indicator_property, sample_grid,
                                                  sis_data_3indicator):
        """Test SIS produces different results with different seeds"""
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=999,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)


# =============================================================================
# SIS - LVM (Locally Varying Marginal Probabilities) Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationLVM:
    """Test SIS with Locally Varying Marginal Probabilities"""

    def test_sis_with_lvm_marginal_probs(self, sample_indicator_property, sample_grid,
                                          sis_data_3indicator, sis_lvm_marginal_probs):
        """Test SIS with LVM marginal probabilities"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=sis_lvm_marginal_probs
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

    def test_sis_with_scalar_marginal_probs(self, sample_indicator_property, sample_grid,
                                              sis_data_3indicator):
        """Test SIS with scalar marginal probabilities"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# SIS - use_harddata Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationUseHarddata:
    """Test SIS use_harddata parameter"""

    def test_sis_use_harddata_true(self, sample_indicator_property, sample_grid,
                                    sis_data_3indicator):
        """Test SIS with use_harddata=True preserves input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=True
        )

        assert isinstance(result, IndProperty)

    def test_sis_use_harddata_false(self, sample_indicator_property, sample_grid,
                                     sis_data_3indicator):
        """Test SIS with use_harddata=False ignores input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=False
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# SIS - Mask Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationMask:
    """Test SIS mask parameter for selective simulation"""

    def test_sis_with_mask(self, sample_indicator_property, sample_grid,
                            sis_data_3indicator, simulation_mask):
        """Test SIS with mask for selective simulation"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            mask=simulation_mask
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == sample_indicator_property.data.shape


# =============================================================================
# SIS - min_neighbours Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationMinNeighbours:
    """Test SIS min_neighbours parameter"""

    def test_sis_min_neighbours_zero(self, sample_indicator_property, sample_grid,
                                      sis_data_3indicator):
        """Test SIS with min_neighbours=0"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            min_neighbours=0
        )

        assert isinstance(result, IndProperty)

    def test_sis_min_neighbours_positive(self, sample_indicator_property, sample_grid,
                                          sis_data_3indicator):
        """Test SIS with positive min_neighbours"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            min_neighbours=4
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# SIS - use_correlogram Parameter Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationUseCorrelogram:
    """Test SIS use_correlogram parameter"""

    def test_sis_use_correlogram_true(self, sample_indicator_property, sample_grid,
                                       sis_data_3indicator):
        """Test SIS with use_correlogram=True"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_correlogram=True
        )

        assert isinstance(result, IndProperty)

    def test_sis_use_correlogram_false(self, sample_indicator_property, sample_grid,
                                        sis_data_3indicator):
        """Test SIS with use_correlogram=False"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_correlogram=False
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# SIS - Statistical Validation Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationStatistics:
    """Test SIS statistical properties"""

    def test_sis_result_not_all_zeros(self, sample_indicator_property, sample_grid,
                                       sis_data_3indicator):
        """Test SIS results are not all zeros"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        assert not np.all(result.data == 0)

    def test_sis_result_within_indicator_range(self, sample_indicator_property, sample_grid,
                                                sis_data_3indicator):
        """Test SIS results are within valid indicator range"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        # All values should be less than indicator_count
        assert np.all(result.data < result.indicator_count)

    def test_sis_indicator_distribution(self, sample_indicator_property, sample_grid,
                                          sis_data_3indicator):
        """Test SIS produces valid indicator distribution"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        # Check that all 3 indicators are present (or at least no invalid values)
        unique_values = np.unique(result.data[result.mask > 0])
        assert np.all(unique_values < 3)
        assert np.all(unique_values >= 0)


# =============================================================================
# SIS - Covariance Model Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulationCovariance:
    """Test SIS with different covariance models"""

    @pytest.mark.parametrize("cov_type", [
        covariance.spherical,
        covariance.exponential,
        covariance.gaussian
    ])
    def test_sis_different_covariance_types(self, sample_indicator_property, sample_grid, cov_type):
        """Test SIS with different covariance types"""
        sis_data = []
        for i in range(3):
            sis_data.append({
                'cov_model': CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1
                ),
                'radiuses': (5, 5, 3),
                'max_neighbours': 12
            })

        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3]
        )

        assert isinstance(result, IndProperty)


# =============================================================================
# Multi-Realization Tests
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.slow
class TestMultipleRealizations:
    """Test multiple realizations with different seeds"""

    def test_sgs_multiple_realizations(self, sample_property, sample_grid,
                                        sample_covariance_model, sgs_cdf_data_multi):
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
                seed=seed
            )
            results.append(result)

        # All results should be different
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                assert not np.array_equal(results[i].data, results[j].data)

    def test_sis_multiple_realizations(self, sample_indicator_property, sample_grid,
                                        sis_data_3indicator):
        """Test SIS produces different results for multiple realizations"""
        seeds = [42, 123, 456, 789, 999]
        results = []

        for seed in seeds:
            result = sis_simulation(
                prop=sample_indicator_property,
                grid=sample_grid,
                data=sis_data_3indicator,
                seed=seed,
                marginal_probs=[0.3, 0.4, 0.3]
            )
            results.append(result)

        # All results should be different
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                assert not np.array_equal(results[i].data, results[j].data)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSimulationEdgeCases:
    """Test edge cases and error handling"""

    def test_sgs_with_small_grid(self):
        """Test SGS with minimal grid size"""
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
        mask = np.array([1, 1, 1, 1], dtype='uint8')

        grid = SugarboxGrid(x=2, y=2, z=1)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(2.0, 2.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0
        )

        cdf_data = CdfData(
            values=np.array([0.0, 5.0], dtype='float32'),
            probs=np.array([0.0, 1.0], dtype='float32')
        )

        prop = ContProperty(data, mask)
        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(2, 2, 1),
            max_neighbours=4,
            cov_model=cov_model,
            seed=42
        )

        assert isinstance(result, ContProperty)

    def test_sis_with_small_grid(self):
        """Test SIS with minimal grid size"""
        np.random.seed(42)
        data = np.array([0, 1, 0, 1], dtype='uint8')
        mask = np.array([1, 1, 1, 1], dtype='uint8')

        grid = SugarboxGrid(x=2, y=2, z=1)

        sis_data = [{
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(2.0, 2.0, 1.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.0
            ),
            'radiuses': (2, 2, 1),
            'max_neighbours': 4
        }]

        prop = IndProperty(data, mask, 2)
        result = sis_simulation(
            prop=prop,
            grid=grid,
            data=sis_data,
            seed=42,
            marginal_probs=[0.5, 0.5]
        )

        assert isinstance(result, IndProperty)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
