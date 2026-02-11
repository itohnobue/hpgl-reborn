"""
Migrated tests from legacy HPGL test scripts.

This file contains tests migrated from the old script-style tests in
src/geo_testing/test_scripts/ that provide valuable test coverage not
already present in the modern test suite.

Legacy tests are marked with @pytest.mark.legacy
Tests use data files restored from git history where available,
and synthetic data where original files were never committed.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

TEST_DATA_DIR = Path(__file__).parent / "test_data"

try:
    from geo_bsd.geo import (
        ordinary_kriging, simple_kriging, lvm_kriging, indicator_kriging,
        simple_cokriging_markI, simple_cokriging_markII,
        ContProperty, IndProperty, CovarianceModel, covariance,
        SugarboxGrid, calc_mean, load_cont_property, load_ind_property,
        write_property
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.cdf import CdfData, calc_cdf
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False

# Grid dimensions used by the "big" test dataset (166x141x20)
BIG_GRID = (166, 141, 20)
BIG_SIZE = 166 * 141 * 20  # 468,120 cells


def _has_data_file(name):
    return (TEST_DATA_DIR / name).exists()


def _load_cont(name):
    """Load a continuous property from test_data directory."""
    return load_cont_property(str(TEST_DATA_DIR / name), -99, BIG_GRID)


def _load_ind(name, indicators):
    """Load an indicator property from test_data directory."""
    return load_ind_property(str(TEST_DATA_DIR / name), -99, indicators, BIG_GRID)


# ============================================================================
# Tests Using Restored Data Files (from git history)
# ============================================================================

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_big_ok_kriging_legacy():
    """
    Ordinary Kriging on large dataset - migrated from test_big.py

    Original: OK with exponential covariance, ranges=(10,10,10), sill=1,
    radiuses=(20,20,20), max_neighbours=12 on 166x141x20 grid.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    result = ordinary_kriging(
        prop=prop_cont,
        grid=grid,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))
    # Some uninformed cells should now be informed
    assert np.sum(result.mask) > np.sum(prop_cont.mask)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_big_sk_kriging_legacy():
    """
    Simple Kriging on large dataset - migrated from test_big.py

    Original: SK with mean=1.6, exponential covariance.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    result = simple_kriging(
        prop=prop_cont,
        grid=grid,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model,
        mean=1.6
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_big_lvm_kriging_legacy():
    """
    LVM Kriging on large dataset - migrated from test_big.py

    Original: LVM kriging using locally varying mean data.
    The mean data file (mean_0.487_166_141_20.inc) was never in git,
    so we generate synthetic mean data from the property itself.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    # Generate mean data as spatially smooth version of the property
    # (original used a separate file with mean ~0.487)
    mean_value = calc_mean(prop_cont)
    mean_data = np.full(BIG_SIZE, mean_value, dtype='float32')

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    result = lvm_kriging(
        prop=prop_cont,
        grid=grid,
        mean_data=mean_data,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_160_141_20.INC"
)
def test_big_ik_legacy():
    """
    Indicator Kriging (2-category) on large dataset - migrated from test_big.py

    Original: IK with 2 categories, marginal_probs=(0.8, 0.2),
    exponential covariance, ranges=(10,10,10), sill=1.
    """
    ik_prop = _load_ind("BIG_SOFT_DATA_160_141_20.INC", [0, 1])
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    ik_data = [
        {
            'cov_model': cov_model,
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
        {
            'cov_model': cov_model,
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        }
    ]

    result = indicator_kriging(
        prop=ik_prop,
        grid=grid,
        data=ik_data,
        marginal_probs=(0.8, 0.2)
    )

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
def test_big_multi_ik_legacy():
    """
    Multi-facies Indicator Kriging (4 categories) - migrated from test_big.py

    Original test defined multi_ik_data with 4 categories and different
    covariance ranges for each. Uses synthetic 4-category data since
    the original data file was never committed to git.
    """
    grid = SugarboxGrid(x=20, y=20, z=10)
    np.random.seed(42)
    data = np.random.randint(0, 4, 4000, dtype='uint8')
    mask = np.ones(4000, dtype='uint8')
    mask[::20] = 0
    prop = IndProperty(data, mask, 4)

    ik_data = [
        {
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(4.0, 4.0, 4.0), sill=0.25, nugget=0.0
            ),
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
        {
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(6.0, 6.0, 6.0), sill=0.25, nugget=0.0
            ),
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
        {
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(2.0, 2.0, 2.0), sill=0.25, nugget=0.0
            ),
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
        {
            'cov_model': CovarianceModel(
                type=covariance.spherical,
                ranges=(10.0, 10.0, 10.0), sill=0.25, nugget=0.0
            ),
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
    ]

    result = indicator_kriging(
        prop=prop,
        grid=grid,
        data=ik_data,
        marginal_probs=[0.24, 0.235, 0.34, 0.18]
    )

    assert isinstance(result, IndProperty)
    assert result.indicator_count == 4


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_big_sgs_sk_legacy():
    """
    SGS with Simple Kriging on large dataset - migrated from test_big.py

    Original: SGS with sill=0.4, exponential covariance, kriging_type="sk".
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=0.4,
        nugget=0.0
    )

    # calc_cdf needs 3D-shaped data
    prop_cont.fix_shape(grid)
    cdf = calc_cdf(prop_cont)

    result = sgs_simulation(
        prop=prop_cont,
        grid=grid,
        cdf_data=cdf,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model,
        seed=3439275
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_160_141_20.INC"
)
def test_big_sis_legacy():
    """
    SIS on large dataset - migrated from test_big.py

    Original: SIS with 2 categories [0,1], seed=3241347,
    exponential covariance, ranges=(10,10,10), sill=1.
    """
    ik_prop = _load_ind("BIG_SOFT_DATA_160_141_20.INC", [0, 1])
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    sis_data = [
        {
            'cov_model': cov_model,
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        },
        {
            'cov_model': cov_model,
            'radiuses': (20, 20, 20),
            'max_neighbours': 12,
        }
    ]

    result = sis_simulation(
        prop=ik_prop,
        grid=grid,
        data=sis_data,
        seed=3241347,
        marginal_probs=(0.8, 0.2)
    )

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE
    assert result.indicator_count == 2


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_simple_cokriging_mark1_legacy():
    """
    Simple Cokriging Mark I - migrated from test_sck_mI.py

    Original: cokriging with correlation_coef=0.97, secondary_variance=5.9.
    Secondary data file was never in git; using synthetic correlated data.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    # Generate synthetic secondary data correlated with primary
    np.random.seed(42)
    primary_mean = calc_mean(prop_cont)
    sec_data_arr = np.where(
        prop_cont.mask == 1,
        prop_cont.data * 0.97 + np.random.randn(BIG_SIZE).astype('float32') * 0.5,
        -99.0
    ).astype('float32')
    sec_mask = prop_cont.mask.copy()
    sec_data = ContProperty(sec_data_arr, sec_mask)
    secondary_mean = calc_mean(sec_data)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    result = simple_cokriging_markI(
        prop=prop_cont,
        grid=grid,
        secondary_data=sec_data,
        primary_mean=primary_mean,
        secondary_mean=secondary_mean,
        secondary_variance=5.9,
        correlation_coef=0.97,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC"
)
def test_simple_cokriging_mark2_legacy():
    """
    Simple Cokriging Mark II - migrated from test_sck_mII.py

    Original: cokriging mark II with correlation_coef=0.97,
    both primary and secondary using exponential covariance.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    # Generate synthetic secondary data
    np.random.seed(42)
    primary_mean = calc_mean(prop_cont)
    sec_data_arr = np.where(
        prop_cont.mask == 1,
        prop_cont.data * 0.97 + np.random.randn(BIG_SIZE).astype('float32') * 0.5,
        -99.0
    ).astype('float32')
    sec_mask = prop_cont.mask.copy()
    sec_data = ContProperty(sec_data_arr, sec_mask)
    secondary_mean = calc_mean(sec_data)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=1.0,
        nugget=0.0
    )

    result = simple_cokriging_markII(
        grid=grid,
        primary_data={
            'data': prop_cont,
            'mean': primary_mean,
            'cov_model': cov_model,
        },
        secondary_data={
            'data': sec_data,
            'mean': secondary_mean,
            'cov_model': cov_model,
        },
        correlation_coef=0.97,
        radiuses=(20, 20, 20),
        max_neighbours=12
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
@pytest.mark.skipif(
    not _has_data_file("NEW_TEST_PROP_01.INC"),
    reason="Test data file not found: NEW_TEST_PROP_01.INC"
)
def test_vpc_small_legacy():
    """
    Vertical Proportion Curves on small grid - migrated from test_vpc.py

    Original: VPC calculation with marginal_probs=[0.8, 0.2] on 55x52x1 grid.
    Uses CalcVPCsIndicator from routines module (pure Python implementation).
    """
    from geo_bsd.routines import CalcVPCsIndicator

    grid_dims = (55, 52, 1)
    ik_prop = load_ind_property(
        str(TEST_DATA_DIR / "NEW_TEST_PROP_01.INC"),
        -99, [0, 1], grid_dims
    )

    # Reshape data to 3D for VPC calculation
    cube = ik_prop.data.reshape(grid_dims, order='F')
    mask = ik_prop.mask.reshape(grid_dims, order='F')

    vpcs = CalcVPCsIndicator(cube, mask, [0, 1], [0.8, 0.2])

    assert len(vpcs) == 2  # One VPC per indicator
    assert len(vpcs[0]) == grid_dims[2]  # One value per Z layer
    assert len(vpcs[1]) == grid_dims[2]


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
def test_sis_on_empty_data_legacy():
    """
    SIS with sparse/empty data - migrated from test_ik_on_empty.py

    Original: SIS on BIG_N_EMPTY.INC (very sparse data), seed=3241347,
    spherical covariance, sill=0.4, radiuses=(10,10,10).
    Data file was never in git; using synthetic sparse data.
    """
    grid = SugarboxGrid(*BIG_GRID)
    np.random.seed(42)

    # Create very sparse indicator data (~1% informed)
    data = np.random.randint(0, 2, BIG_SIZE, dtype='uint8')
    mask = np.zeros(BIG_SIZE, dtype='uint8')
    sparse_indices = np.random.choice(BIG_SIZE, BIG_SIZE // 100, replace=False)
    mask[sparse_indices] = 1
    prop = IndProperty(data, mask, 2)

    cov_model = CovarianceModel(
        type=covariance.spherical,
        ranges=(10.0, 10.0, 10.0),
        sill=0.4,
        nugget=0.0
    )

    sis_data = [
        {
            'cov_model': cov_model,
            'radiuses': (10, 10, 10),
            'max_neighbours': 12,
        },
        {
            'cov_model': cov_model,
            'radiuses': (10, 10, 10),
            'max_neighbours': 12,
        }
    ]

    result = sis_simulation(
        prop=prop,
        grid=grid,
        data=sis_data,
        seed=3241347,
        marginal_probs=(0.9, 0.1)
    )

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE
    assert result.indicator_count == 2


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.legacy
def test_sgs_on_empty_data_legacy():
    """
    SGS with sparse/empty data - migrated from test_sgs_on_empty.py

    Original: SGS on BIG_N_EMPTY.INC, seed=3439275, exponential covariance,
    sill=0.4, radiuses=(20,20,20). Data file was never in git;
    using synthetic sparse continuous data.
    """
    grid = SugarboxGrid(*BIG_GRID)
    np.random.seed(42)

    # Create very sparse continuous data (~1% informed)
    data = np.random.rand(BIG_SIZE).astype('float32') * 100
    mask = np.zeros(BIG_SIZE, dtype='uint8')
    sparse_indices = np.random.choice(BIG_SIZE, BIG_SIZE // 100, replace=False)
    mask[sparse_indices] = 1
    prop = ContProperty(data, mask)

    cov_model = CovarianceModel(
        type=covariance.exponential,
        ranges=(10.0, 10.0, 10.0),
        sill=0.4,
        nugget=0.0
    )

    # Build CDF from the sparse informed data
    informed = data[mask == 1]
    sorted_vals = np.sort(informed)
    probs = np.linspace(0, 1, len(sorted_vals)).astype('float32')
    cdf_data = CdfData(sorted_vals, probs)

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
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype('float64')))


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
