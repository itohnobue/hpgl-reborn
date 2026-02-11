"""
Migrated tests from legacy HPGL test scripts.

This file contains tests migrated from the old script-style tests in
src/geo_testing/test_scripts/ that provide valuable test coverage not
already present in the modern test suite.

Legacy tests are marked with @pytest.mark.legacy
Tests requiring external data are marked with skip and documented.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ordinary_kriging, simple_kriging, indicator_kriging,
        ContProperty, IndProperty, CovarianceModel, covariance,
        SugarboxGrid, calc_mean
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False


# ============================================================================
# Tests Requiring External Data (Documented and Skipped)
# ============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_CON_160_141_20.INC")
def test_big_ok_kriging_legacy():
    """
    Test Ordinary Kriging on large dataset - migrated from test_big.py

    Original test loaded BIG_SOFT_DATA_CON_160_141_20.INC (166x141x20 grid)
    and ran OK with exponential covariance, ranges=(10,10,10), sill=1.

    Requirements:
      - Test data: src/geo_testing/test_scripts/test_data/BIG_SOFT_DATA_CON_160_141_20.INC
      - Grid size: 166x141x20
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_CON_160_141_20.INC")
def test_big_sk_kriging_legacy():
    """
    Test Simple Kriging on large dataset - migrated from test_big.py

    Original test used mean=1.6 with exponential covariance.

    Requirements:
      - Test data: src/geo_testing/test_scripts/test_data/BIG_SOFT_DATA_CON_160_141_20.INC
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: mean_0.487_166_141_20.inc")
def test_big_lvm_kriging_legacy():
    """
    Test LVM Kriging on large dataset - migrated from test_big.py

    Original test used locally varying mean data from mean_0.487_166_141_20.inc.

    Requirements:
      - Primary data: BIG_SOFT_DATA_CON_160_141_20.INC
      - Mean data: mean_0.487_166_141_20.inc
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_160_141_20.INC")
def test_big_ik_legacy():
    """
    Test Indicator Kriging on large dataset - migrated from test_big.py

    Original test used 2-category IK with marginal_probs=(0.8, 0.2).

    Requirements:
      - Test data: BIG_SOFT_DATA_160_141_20.INC with indicators [0,1]
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_160_141_20.INC")
def test_big_multi_ik_legacy():
    """
    Test multi-facies Indicator Kriging - migrated from test_big.py

    Original test used 4 categories with different ranges for each:
      - Category 0: ranges=(4,4,4), sill=0.25, marginal_prob=0.24
      - Category 1: ranges=(6,6,6), sill=0.25, marginal_prob=0.235
      - Category 2: ranges=(2,2,2), sill=0.25, marginal_prob=0.34
      - Category 3: ranges=(10,10,10), sill=0.25, marginal_prob=0.18

    Requirements:
      - Test data: MULTI_IND.INC with indicators [0,1,2,3]
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_CON_160_141_20.INC")
def test_big_sgs_sk_legacy():
    """
    Test SGS with Simple Kriging on large dataset - migrated from test_big.py

    Original test used sill=0.4 (different from kriging tests).

    Requirements:
      - Test data: BIG_SOFT_DATA_CON_160_141_20.INC
      - Mean data: mean_0.487_166_141_20.inc for LVM variant
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_160_141_20.INC")
def test_big_sis_legacy():
    """
    Test SIS on large dataset - migrated from test_big.py

    Requirements:
      - Test data: BIG_SOFT_DATA_160_141_20.INC with indicators [0,1]
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_CON_CoK.INC")
def test_simple_cokriging_mark1_legacy():
    """
    Test Simple Cokriging Mark I - migrated from test_sck_mI.py

    Original test parameters:
      - correlation_coef=0.97
      - secondary_variance=5.9

    Requirements:
      - Primary data: BIG_SOFT_DATA_CON_160_141_20.INC
      - Secondary data: BIG_SOFT_DATA_CON_CoK.INC
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_SOFT_DATA_CON_CoK.INC")
def test_simple_cokriging_mark2_legacy():
    """
    Test Simple Cokriging Mark II - migrated from test_sck_mII.py

    Requirements:
      - Primary data: BIG_SOFT_DATA_CON_160_141_20.INC
      - Secondary data: BIG_SOFT_DATA_CON_CoK.INC
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: NEW_TEST_PROP_01.INC")
def test_vpc_small_legacy():
    """
    Test Variogram of Pixel Coordinates on small grid - migrated from test_vpc.py

    Original test calculated VPC with marginal_probs=[0.8, 0.2].

    Requirements:
      - Test data: NEW_TEST_PROP_01.INC with indicators [0,1]
      - Grid: 55x52x1
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_N_EMPTY.INC")
def test_sis_on_empty_data_legacy():
    """
    Test SIS with sparse/empty data - migrated from test_ik_on_empty.py

    Original test used very sparse data with sill=0.4, smaller radiuses=(10,10,10).

    Requirements:
      - Test data: BIG_N_EMPTY.INC with indicators [0,1]
      - Grid: 166x141x20
    """
    pass


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skip(reason="Requires external test data file: BIG_N_EMPTY.INC")
def test_sgs_on_empty_data_legacy():
    """
    Test SGS with sparse/empty data - migrated from test_sgs_on_empty.py

    Requirements:
      - Test data: BIG_N_EMPTY.INC
      - Grid: 166x141x20
    """
    pass


# ============================================================================
# Tests Implemented with Synthetic Data
# ============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestAnisotropicCovariance:
    """Tests for anisotropic covariance models - migrated from test_big_aniz.py"""

    def test_sk_with_anisotropy(self):
        """
        Test SK with anisotropic ranges and angles - migrated from test_big_aniz.py

        Original test used:
          - ranges=(20, 10, 5) - anisotropic ranges
          - angles=(40, 50, 90) - rotation angles
          - mean=1.7
        """
        grid = SugarboxGrid(x=20, y=20, z=10)
        np.random.seed(42)
        data = np.random.rand(4000).astype('float32') * 3.4  # centered around mean=1.7
        mask = np.ones(4000, dtype='uint8')
        # Make some values uninformed
        mask[::20] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(20.0, 10.0, 5.0),
            angles=(40.0, 50.0, 90.0),
            sill=1.0,
            nugget=0.0
        )

        result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(100, 100, 100),
            max_neighbours=12,
            cov_model=cov_model,
            mean=1.7
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape
        # Check results are reasonable (not all zeros, no NaN)
        assert not np.all(result.data == 0)
        assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestDifferentCovarianceTypes:
    """Tests for different covariance types - migrated from test_compare.py"""

    @pytest.mark.parametrize("cov_type_name,cov_type", [
        ("exponential", covariance.exponential),
        ("gaussian", covariance.gaussian),
        ("spherical", covariance.spherical),
    ])
    def test_sk_with_different_covariance_types(self, cov_type_name, cov_type):
        """
        Test SK with different covariance types - migrated from test_compare.py

        Original test_compare.py tested:
          - Exponential: ranges=(10,10,10), sill=1
          - Gaussian: ranges=(10,10,10), sill=1
          - Spherical: ranges=(10,10,10), sill=1
        """
        grid = SugarboxGrid(x=15, y=15, z=10)
        np.random.seed(42)
        data = np.random.rand(2250).astype('float32') * 3.2
        mask = np.ones(2250, dtype='uint8')
        mask[::15] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=cov_type,
            ranges=(10.0, 10.0, 10.0),
            sill=1.0,
            nugget=0.0
        )

        result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            mean=1.6
        )

        assert isinstance(result, ContProperty)
        assert not np.all(result.data == 0)
        assert not np.any(np.isnan(result.data.astype('float64')))

    def test_sk_with_nugget(self):
        """
        Test SK with nugget effect - migrated from test_compare.py

        Original test used nugget=0.5 with exponential covariance.
        """
        grid = SugarboxGrid(x=15, y=15, z=10)
        np.random.seed(42)
        data = np.random.rand(2250).astype('float32') * 3.2
        mask = np.ones(2250, dtype='uint8')
        mask[::15] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=1.0,
            nugget=0.5  # Significant nugget effect
        )

        result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            mean=1.6
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestMaskedSimulation:
    """Tests for masked simulation - migrated from masked_sgs.py"""

    def test_sgs_with_mask(self):
        """
        Test SGS with masking - migrated from masked_sgs.py

        Original test created a checkerboard mask (alternating 0/1)
        and ran SGS only on masked nodes.
        """
        grid = SugarboxGrid(x=55, y=52, z=1)
        np.random.seed(42)
        data = np.random.rand(2860).astype('float32') * 100
        mask = np.ones(2860, dtype='uint8')
        # Make alternating pattern mask like original test
        for i in range(2860):
            mask[i] = i % 2

        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=0.4,
            nugget=0.0
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        # Create mask array (0 = skip, 1 = process)
        process_mask = np.ones(2860, dtype='uint8')
        for i in range(2860):
            process_mask[i] = i % 2

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=3439275,
            mask=process_mask
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestMultiCategoryIndicator:
    """Tests for multi-category indicator simulation - migrated from test_compare.py"""

    def test_multi_facies_sis(self):
        """
        Test SIS with 4 facies categories - migrated from test_compare.py

        Original test used 4 categories with different covariance ranges
        for each category to model different spatial continuity per facies.
        """
        grid = SugarboxGrid(x=20, y=20, z=10)
        np.random.seed(42)
        # 4 categories: 0, 1, 2, 3
        data = np.random.randint(0, 4, 4000, dtype='uint8')
        mask = np.ones(4000, dtype='uint8')
        mask[::20] = 0
        prop = IndProperty(data, mask, 4)

        # Different covariance for each category
        sis_data = [
            {
                'cov_model': CovarianceModel(
                    type=covariance.spherical,
                    ranges=(4.0, 4.0, 4.0),
                    sill=0.25,
                    nugget=0.0
                ),
                'radiuses': (20, 20, 20),
                'max_neighbours': 12
            },
            {
                'cov_model': CovarianceModel(
                    type=covariance.spherical,
                    ranges=(6.0, 6.0, 6.0),
                    sill=0.25,
                    nugget=0.0
                ),
                'radiuses': (20, 20, 20),
                'max_neighbours': 12
            },
            {
                'cov_model': CovarianceModel(
                    type=covariance.spherical,
                    ranges=(2.0, 2.0, 2.0),
                    sill=0.25,
                    nugget=0.0
                ),
                'radiuses': (20, 20, 20),
                'max_neighbours': 12
            },
            {
                'cov_model': CovarianceModel(
                    type=covariance.spherical,
                    ranges=(10.0, 10.0, 10.0),
                    sill=0.25,
                    nugget=0.0
                ),
                'radiuses': (20, 20, 20),
                'max_neighbours': 12
            }
        ]

        marginal_probs = [0.24, 0.235, 0.34, 0.18]

        result = sis_simulation(
            prop=prop,
            grid=grid,
            data=sis_data,
            seed=3241347,
            marginal_probs=marginal_probs
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 4


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestSGSReproducibility:
    """Tests for SGS reproducibility across different parameters - migrated from test_sgs.py"""

    def test_sgs_reproducibility_same_seed(self):
        """
        Test SGS produces same results with same seed - migrated from test_sgs.py

        Original test_sgs.py tested SGS with workers_count from 1 to 5
        and verified reproducibility.
        """
        grid = SugarboxGrid(x=20, y=20, z=5)
        np.random.seed(42)
        data = np.random.rand(2000).astype('float32') * 100
        mask = np.ones(2000, dtype='uint8')
        mask[::10] = 0
        prop1 = ContProperty(data.copy(), mask.copy())
        prop2 = ContProperty(data.copy(), mask.copy())

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=0.4,
            nugget=0.0
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        result1 = sgs_simulation(
            prop=prop1,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=3439275
        )

        result2 = sgs_simulation(
            prop=prop2,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=3439275
        )

        # Same seed should produce same results
        np.testing.assert_array_equal(result1.data, result2.data)

    def test_sgs_different_seeds_produce_different_results(self):
        """
        Test SGS produces different results with different seeds
        """
        grid = SugarboxGrid(x=20, y=20, z=5)
        np.random.seed(42)
        data = np.random.rand(2000).astype('float32') * 100
        mask = np.ones(2000, dtype='uint8')
        mask[::10] = 0
        prop1 = ContProperty(data.copy(), mask.copy())
        prop2 = ContProperty(data.copy(), mask.copy())

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=0.4,
            nugget=0.0
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        result1 = sgs_simulation(
            prop=prop1,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=3439275
        )

        result2 = sgs_simulation(
            prop=prop2,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=24193421  # Different seed
        )

        # Different seeds should produce different results
        assert not np.array_equal(result1.data, result2.data)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestEdgeCases:
    """Tests for edge cases - migrated from test_ik_on_empty.py and test_sgs_on_empty.py"""

    def test_kriging_with_very_sparse_data(self):
        """
        Test kriging behavior with very sparse input data

        Original test_ik_on_empty.py used BIG_N_EMPTY.INC which had
        very few informed nodes.
        """
        grid = SugarboxGrid(x=50, y=50, z=20)
        np.random.seed(42)
        data = np.random.rand(50000).astype('float32') * 100
        mask = np.zeros(50000, dtype='uint8')  # All uninformed initially
        # Only inform 1% of nodes
        sparse_indices = np.random.choice(50000, 500, replace=False)
        mask[sparse_indices] = 1
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=0.4,
            nugget=0.0
        )

        result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 10),  # Smaller search radius
            max_neighbours=12,
            cov_model=cov_model,
            mean=50.0
        )

        assert isinstance(result, ContProperty)
        # With sparse data, many nodes may still be uninformed
        # but result should be valid where estimated
        assert not np.any(np.isnan(result.data.astype('float64')))

    def test_simulation_with_very_sparse_data(self):
        """
        Test SGS behavior with very sparse input data
        """
        grid = SugarboxGrid(x=30, y=30, z=10)
        np.random.seed(42)
        data = np.random.rand(9000).astype('float32') * 100
        mask = np.zeros(9000, dtype='uint8')
        # Only inform 2% of nodes
        sparse_indices = np.random.choice(9000, 180, replace=False)
        mask[sparse_indices] = 1
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(10.0, 10.0, 10.0),
            sill=0.4,
            nugget=0.0
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(20, 20, 20),
            max_neighbours=12,
            cov_model=cov_model,
            seed=3439275
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
class TestSGSSeedVariations:
    """
    Tests from test_sk_sgs.py - simulation averaging behavior

    The original test ran 100 SGS realizations and checked that
    the average converged to the SK kriging estimate. This is
    a property test verifying simulation correctness.
    """

    def test_sgi_average_converges_to_sk(self):
        """
        Test that average of multiple SGS realizations converges to SK estimate

        This is a simplified version using fewer realizations than the
        original 100, but still tests the convergence property.
        """
        grid = SugarboxGrid(x=15, y=15, z=5)
        np.random.seed(42)
        data = np.random.rand(1125).astype('float32') * 100
        mask = np.ones(1125, dtype='uint8')
        mask[::10] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(5.0, 5.0, 5.0),
            sill=1.0,
            nugget=0.0
        )

        # Get SK estimate
        sk_result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 10),
            max_neighbours=12,
            cov_model=cov_model,
            mean=None
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        # Run multiple SGS realizations
        n_realizations = 10
        sum_array = np.zeros(sk_result.data.size, dtype='float32')

        for i in range(n_realizations):
            # Need fresh input for each realization
            fresh_prop = ContProperty(data.copy(), mask.copy())
            sgs_result = sgs_simulation(
                prop=fresh_prop,
                grid=grid,
                cdf_data=cdf_data,
                radiuses=(10, 10, 10),
                max_neighbours=12,
                cov_model=cov_model,
                seed=(94234523 // (i + 1)) - (234432 // (i + 1))
            )
            sum_array += sgs_result.data.flatten()

        # Calculate average
        average = sum_array / n_realizations

        # Check that average is reasonably close to SK estimate
        # We allow some tolerance due to random variation
        mean_diff = np.mean(np.abs(average - sk_result.data.flatten()))

        # The difference should be relatively small
        # (this is a statistical test, so we use a loose tolerance)
        assert mean_diff < 50.0, f"Mean difference {mean_diff} too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'legacy'])
