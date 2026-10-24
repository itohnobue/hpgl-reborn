"""
Migrated tests from legacy HPGL test scripts.

This file contains tests migrated from the old script-style tests in
src/geo_testing/test_scripts/ that provide valuable test coverage not
already present in the modern test suite.

Legacy tests are marked with @pytest.mark.legacy
Tests use data files restored from git history where available,
and synthetic data where original files were never committed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

TEST_DATA_DIR = Path(__file__).parent / "test_data"

try:
    from geo_bsd.cdf import CdfData, calc_cdf
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        calc_mean,
        covariance,
        indicator_kriging,
        load_cont_property,
        load_ind_property,
        lvm_kriging,
        ordinary_kriging,
        simple_cokriging_markI,
        simple_cokriging_markII,
        simple_kriging,
    )
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


def _assert_written(result, prop):
    """F-557: the result must have WRITTEN the previously-uninformed cells.

    Every test in this file previously asserted only isinstance/size/NaN, so
    two mutations shipped green: "return the input unchanged" (a pass-through)
    and "fill with a constant". Both assertions below are shape-agnostic —
    the loaded fixtures are 3D while the synthetic ones are flat, so compare
    in F-order (the property layout).
    """
    uninformed = np.asarray(prop.mask).ravel(order="F") == 0
    before = np.asarray(prop.data).ravel(order="F")[uninformed]
    after = np.asarray(result.data).ravel(order="F")[uninformed]
    assert np.any(after != before), (
        "no previously-uninformed cell was written — result is a pass-through of the input"
    )
    assert float(np.std(after.astype("float64"))) > 0.0, (
        "previously-uninformed cells were filled with a constant value"
    )


def _assert_valid_categories(result):
    """F-557: indicator results must hold valid category codes.

    IndProperty.data is uint8, so the existing NaN/Inf checks are structurally
    unsatisfiable; every assertion in the file passed for an all-zero or
    out-of-range output.
    """
    count = int(result.indicator_count)
    values = np.asarray(result.data)
    assert values.size > 0
    assert np.all(values < count), (
        f"indicator result holds categories outside [0, {count})"
    )


# ============================================================================
# Tests Using Restored Data Files (from git history)
# ============================================================================


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell kriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
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
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    result = ordinary_kriging(
        prop=prop_cont, grid=grid, radiuses=(20, 20, 20), max_neighbours=12, cov_model=cov_model
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    # Some uninformed cells should now be informed
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell kriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
)
def test_big_sk_kriging_legacy():
    """
    Simple Kriging on large dataset - migrated from test_big.py

    Original: SK with mean=1.6, exponential covariance.
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    result = simple_kriging(
        prop=prop_cont,
        grid=grid,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model,
        mean=1.6,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell kriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
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
    mean_data = np.full(BIG_SIZE, mean_value, dtype="float32")

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    result = lvm_kriging(
        prop=prop_cont,
        grid=grid,
        mean_data=mean_data,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell kriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_160_141_20.INC",
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
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    ik_data = [
        {
            "cov_model": cov_model,
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
        {
            "cov_model": cov_model,
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
    ]

    result = indicator_kriging(prop=ik_prop, grid=grid, data=ik_data, marginal_probs=(0.8, 0.2))

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE
    assert np.sum(result.mask) > np.sum(ik_prop.mask)
    _assert_valid_categories(result)


@pytest.mark.hpgl
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
    data = np.random.randint(0, 4, 4000, dtype="uint8")
    mask = np.ones(4000, dtype="uint8")
    mask[::20] = 0
    prop = IndProperty(data, mask, 4)

    ik_data = [
        {
            "cov_model": CovarianceModel(
                type=covariance.spherical, ranges=(4.0, 4.0, 4.0), sill=0.25, nugget=0.0
            ),
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
        {
            "cov_model": CovarianceModel(
                type=covariance.spherical, ranges=(6.0, 6.0, 6.0), sill=0.25, nugget=0.0
            ),
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
        {
            "cov_model": CovarianceModel(
                type=covariance.spherical, ranges=(2.0, 2.0, 2.0), sill=0.25, nugget=0.0
            ),
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
        {
            "cov_model": CovarianceModel(
                type=covariance.spherical, ranges=(10.0, 10.0, 10.0), sill=0.25, nugget=0.0
            ),
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
    ]

    result = indicator_kriging(
        prop=prop, grid=grid, data=ik_data, marginal_probs=[0.24, 0.235, 0.34, 0.18]
    )

    assert isinstance(result, IndProperty)
    assert result.indicator_count == 4
    assert np.sum(result.mask) > np.sum(prop.mask)
    _assert_valid_categories(result)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # H-02/B-05: 468K-cell SGS — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
)
def test_big_sgs_sk_legacy():
    """
    SGS with Simple Kriging on large dataset - migrated from test_big.py

    Original: SGS with sill=0.4, exponential covariance, kriging_type="sk".
    """
    prop_cont = _load_cont("BIG_SOFT_DATA_CON_160_141_20.INC")
    grid = SugarboxGrid(*BIG_GRID)

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=0.4, nugget=0.0
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
        seed=3439275,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # H-02/B-05: 468K-cell SIS — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_160_141_20.INC",
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
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    sis_data = [
        {
            "cov_model": cov_model,
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
        {
            "cov_model": cov_model,
            "radiuses": (20, 20, 20),
            "max_neighbours": 12,
        },
    ]

    result = sis_simulation(
        prop=ik_prop, grid=grid, data=sis_data, seed=3241347, marginal_probs=(0.8, 0.2)
    )

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE
    assert result.indicator_count == 2
    assert np.sum(result.mask) > np.sum(ik_prop.mask)
    _assert_valid_categories(result)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # H-02/B-05: 468K-cell cokriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
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
    # F-55: load_cont_property with a 3-tuple size now returns 3D arrays on
    # both parser paths; ravel for the flat-array arithmetic below.
    prim_data_flat = prop_cont.data.ravel(order="F")
    prim_mask_flat = prop_cont.mask.ravel(order="F")
    sec_data_arr = np.where(
        prim_mask_flat == 1,
        prim_data_flat * 0.97 + np.random.randn(BIG_SIZE).astype("float32") * 0.5,
        -99.0,
    ).astype("float32")
    sec_mask = prim_mask_flat.copy()
    sec_data = ContProperty(sec_data_arr, sec_mask)
    secondary_mean = calc_mean(sec_data)

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
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
        cov_model=cov_model,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # H-02/B-05: 468K-cell cokriging — top machine-freeze risk, run last/alone
@pytest.mark.skipif(
    not _has_data_file("BIG_SOFT_DATA_CON_160_141_20.INC"),
    reason="Test data file not found: BIG_SOFT_DATA_CON_160_141_20.INC",
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
    # F-55: load_cont_property with a 3-tuple size now returns 3D arrays on
    # both parser paths; ravel for the flat-array arithmetic below.
    prim_data_flat = prop_cont.data.ravel(order="F")
    prim_mask_flat = prop_cont.mask.ravel(order="F")
    sec_data_arr = np.where(
        prim_mask_flat == 1,
        prim_data_flat * 0.97 + np.random.randn(BIG_SIZE).astype("float32") * 0.5,
        -99.0,
    ).astype("float32")
    sec_mask = prim_mask_flat.copy()
    sec_data = ContProperty(sec_data_arr, sec_mask)
    secondary_mean = calc_mean(sec_data)

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=1.0, nugget=0.0
    )

    result = simple_cokriging_markII(
        grid=grid,
        primary_data={
            "data": prop_cont,
            "mean": primary_mean,
            "cov_model": cov_model,
        },
        secondary_data={
            "data": sec_data,
            "mean": secondary_mean,
            "cov_model": cov_model,
        },
        correlation_coef=0.97,
        radiuses=(20, 20, 20),
        max_neighbours=12,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop_cont.mask)
    _assert_written(result, prop_cont)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell SIS — top machine-freeze risk, run last/alone
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
    data = np.random.randint(0, 2, BIG_SIZE, dtype="uint8")
    mask = np.zeros(BIG_SIZE, dtype="uint8")
    sparse_indices = np.random.choice(BIG_SIZE, BIG_SIZE // 100, replace=False)
    mask[sparse_indices] = 1
    prop = IndProperty(data, mask, 2)

    cov_model = CovarianceModel(
        type=covariance.spherical, ranges=(10.0, 10.0, 10.0), sill=0.4, nugget=0.0
    )

    sis_data = [
        {
            "cov_model": cov_model,
            "radiuses": (10, 10, 10),
            "max_neighbours": 12,
        },
        {
            "cov_model": cov_model,
            "radiuses": (10, 10, 10),
            "max_neighbours": 12,
        },
    ]

    result = sis_simulation(
        prop=prop, grid=grid, data=sis_data, seed=3241347, marginal_probs=(0.9, 0.1)
    )

    assert isinstance(result, IndProperty)
    assert result.data.size == BIG_SIZE
    assert result.indicator_count == 2
    assert np.sum(result.mask) > np.sum(prop.mask)
    _assert_valid_categories(result)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # B-05: 468K-cell SGS — top machine-freeze risk, run last/alone
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
    data = np.random.rand(BIG_SIZE).astype("float32") * 100
    mask = np.zeros(BIG_SIZE, dtype="uint8")
    sparse_indices = np.random.choice(BIG_SIZE, BIG_SIZE // 100, replace=False)
    mask[sparse_indices] = 1
    prop = ContProperty(data, mask)

    cov_model = CovarianceModel(
        type=covariance.exponential, ranges=(10.0, 10.0, 10.0), sill=0.4, nugget=0.0
    )

    # Build CDF from the sparse informed data
    informed = data[mask == 1]
    sorted_vals = np.sort(informed)
    probs = np.linspace(0, 1, len(sorted_vals)).astype("float32")
    cdf_data = CdfData(sorted_vals, probs)

    result = sgs_simulation(
        prop=prop,
        grid=grid,
        cdf_data=cdf_data,
        radiuses=(20, 20, 20),
        max_neighbours=12,
        cov_model=cov_model,
        seed=3439275,
    )

    assert isinstance(result, ContProperty)
    assert result.data.size == BIG_SIZE
    assert not np.any(np.isnan(result.data.astype("float64")))
    assert np.sum(result.mask) > np.sum(prop.mask)
    _assert_written(result, prop)


# ============================================================================
# Tests Implemented with Synthetic Data
# ============================================================================


@pytest.mark.hpgl
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
        data = np.random.rand(4000).astype("float32") * 3.4  # centered around mean=1.7
        mask = np.ones(4000, dtype="uint8")
        # Make some values uninformed
        mask[::20] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(20.0, 10.0, 5.0),
            angles=(40.0, 50.0, 90.0),
            sill=1.0,
            nugget=0.0,
        )

        result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(100, 100, 100),
            max_neighbours=12,
            cov_model=cov_model,
            mean=1.7,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape
        # Check results are reasonable (not all zeros, no NaN)
        assert not np.all(result.data == 0)
        assert not np.any(np.isnan(result.data.astype("float64")))

        # F-557: the test passed Angles=(40,50,90) but never asserted that the
        # rotation contributed anything — a regression dropping the angles
        # (rotating the ellipsoid to identity) was invisible. Identical
        # ranges, mean and data with Angles=(0,0,0) must produce a DIFFERENT
        # field; measured difference on the untouched kernel: max |Δ| ≈ 1.74.
        isotropic_model = CovarianceModel(
            type=covariance.exponential,
            ranges=(20.0, 10.0, 5.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )
        isotropic = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(100, 100, 100),
            max_neighbours=12,
            cov_model=isotropic_model,
            mean=1.7,
        )
        assert not np.allclose(result.data, isotropic.data, rtol=0, atol=1e-6), (
            "covariance Angles had no effect on the SK estimates"
        )


@pytest.mark.hpgl
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
        data = np.random.randint(0, 4, 4000, dtype="uint8")
        mask = np.ones(4000, dtype="uint8")
        mask[::20] = 0
        prop = IndProperty(data, mask, 4)

        # Different covariance for each category
        sis_data = [
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical, ranges=(4.0, 4.0, 4.0), sill=0.25, nugget=0.0
                ),
                "radiuses": (20, 20, 20),
                "max_neighbours": 12,
            },
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical, ranges=(6.0, 6.0, 6.0), sill=0.25, nugget=0.0
                ),
                "radiuses": (20, 20, 20),
                "max_neighbours": 12,
            },
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical, ranges=(2.0, 2.0, 2.0), sill=0.25, nugget=0.0
                ),
                "radiuses": (20, 20, 20),
                "max_neighbours": 12,
            },
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical, ranges=(10.0, 10.0, 10.0), sill=0.25, nugget=0.0
                ),
                "radiuses": (20, 20, 20),
                "max_neighbours": 12,
            },
        ]

        marginal_probs = [0.24, 0.235, 0.34, 0.18]

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data, seed=3241347, marginal_probs=marginal_probs
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 4
        assert np.sum(result.mask) > np.sum(prop.mask)
        _assert_valid_categories(result)


@pytest.mark.hpgl
@pytest.mark.legacy
def test_sis_unconditional_follows_marginals():
    """F-557: the requested marginals must drive the simulated categories.

    None of the SIS tests in this file could detect a regression that ignored
    `marginal_probs`: the multi-facies fixture requests [0.24, 0.235, 0.34,
    0.18] against a uniform input prior, i.e. the no-op and the real kernel
    produce indistinguishable proportions. With `max_neighbours=0` (the
    documented "unconditional simulation" mode) every node takes the
    marginal-probability draw, so the realized proportions follow the
    request; flipping the request flips the result. Measured on the untouched
    kernel: (0.9, 0.1) -> [0.908, 0.092], (0.1, 0.9) -> [0.178, 0.822]
    (the 10% hard-data cells are all category 0 and stay category 0).
    """
    grid = SugarboxGrid(x=10, y=10, z=5)
    np.random.seed(42)
    data = np.zeros(500, dtype="uint8")
    mask = np.zeros(500, dtype="uint8")
    mask[::10] = 1  # 10% hard data, all category 0
    prop = IndProperty(data, mask, 2)

    ik_data = [
        {
            "cov_model": CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=0.5, nugget=0.0
            ),
            "radiuses": (5, 5, 3),
            "max_neighbours": 0,  # unconditional: every node uses the marginal
        }
        for _ in range(2)
    ]

    heavy_zero = sis_simulation(
        prop=prop, grid=grid, data=ik_data, seed=3241347, marginal_probs=(0.9, 0.1)
    )
    heavy_one = sis_simulation(
        prop=prop, grid=grid, data=ik_data, seed=3241347, marginal_probs=(0.1, 0.9)
    )

    zero_share = float(np.mean(np.asarray(heavy_zero.data) == 0))
    one_share = float(np.mean(np.asarray(heavy_one.data) == 1))
    assert zero_share > 0.8, f"marginal (0.9, 0.1) produced category 0 only {zero_share:.3f}"
    assert one_share > 0.7, f"marginal (0.1, 0.9) produced category 1 only {one_share:.3f}"
    _assert_valid_categories(heavy_zero)
    _assert_valid_categories(heavy_one)


@pytest.mark.hpgl
@pytest.mark.legacy
@pytest.mark.slow  # N2-L08: 100 SGS realizations on a 15×15×5 grid — heavy, slow-suite only
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

        Runs 100 SGS realizations (matching the original test) and checks
        the convergence property.
        """
        grid = SugarboxGrid(x=15, y=15, z=5)
        np.random.seed(42)
        data = np.random.rand(1125).astype("float32") * 100
        mask = np.ones(1125, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.exponential, ranges=(5.0, 5.0, 5.0), sill=1.0, nugget=0.0
        )

        # Get SK estimate
        sk_result = simple_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 10),
            max_neighbours=12,
            cov_model=cov_model,
            mean=None,
        )

        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype="float32"),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32"),
        )

        # Run multiple SGS realizations (100 for statistical convergence)
        n_realizations = 100
        sum_array = np.zeros(sk_result.data.size, dtype="float32")

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
                seed=int(100000 + i * 99991),
            )
            sum_array += sgs_result.data.flatten(order="F")

        # Calculate average
        average = sum_array / n_realizations

        # Check that average is reasonably close to SK estimate
        # With 100 realizations, convergence should be tighter
        mean_diff = np.mean(np.abs(average - sk_result.data.flatten(order="F")))

        # The difference should be small with 100 realizations (tighter than original 50.0)
        assert mean_diff < 1.0, f"Mean difference {mean_diff} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "legacy"])
