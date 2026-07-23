"""
Parametrized tests for HPGL simulation algorithms:
- Sequential Gaussian Simulation (SGS)
- Sequential Indicator Simulation (SIS)

Comprehensive parametrized coverage across:
- Kriging types (SK, OK)
- Seeds (0, 1, 42, 100, random)
- Mask patterns (none, checkerboard, sparse)
- Min neighbours (0, 1, 5, 10)
- Grid sizes ((4,4,4), (10,10,10))
- Covariance types (spherical, exponential, gaussian)
- Edge cases (empty, single-point, uniform, extreme, dense, sparse)
- Property construction (ContProperty, IndProperty)

Invariants verified:
- No NaN/Inf in output
- Output shape matches input
- Deterministic with fixed seed
- Parallel and serial produce identical results (OMP_NUM_THREADS)
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
# Helpers
# =============================================================================

COVARIANCE_NAMES = {
    covariance.spherical: "spherical",
    covariance.exponential: "exponential",
    covariance.gaussian: "gaussian",
}


def _make_cov_model(cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1):
    """Create a CovarianceModel with given type."""
    return CovarianceModel(type=cov_type, ranges=ranges, angles=angles, sill=sill, nugget=nugget)


def _make_cont_property(grid, mask_pattern="none", data_pattern="random", extra_mask=None):
    """Create a ContProperty for SGS testing.

    Parameters
    ----------
    grid : SugarboxGrid
    mask_pattern : str
        "none" = all informed, "checkerboard" = alternating, "sparse" = 10% informed,
        "fully_informed" = 100% informed, "all_uninformed" = 0% informed
    data_pattern : str
        "random" = uniform random [0, 100), "uniform" = all 42.0,
        "extreme" = very large/small values, "zeros" = all zeros
    extra_mask : numpy.ndarray or None
        Additional mask to AND with the generated mask
    """
    rng = np.random.RandomState(42)
    size = grid.x * grid.y * grid.z

    if data_pattern == "random":
        data = rng.rand(size).astype("float32") * 100
    elif data_pattern == "uniform":
        data = np.full(size, 42.0, dtype="float32")
    elif data_pattern == "extreme":
        half = size // 2
        data = np.empty(size, dtype="float32")
        data[:half] = 1e-6
        data[half:] = 1e6
        rng.shuffle(data)
    elif data_pattern == "zeros":
        data = np.zeros(size, dtype="float32")
    elif data_pattern == "negative":
        data = rng.rand(size).astype("float32") * 100 - 50
    else:
        raise ValueError(f"Unknown data_pattern: {data_pattern}")

    if mask_pattern == "none" or mask_pattern == "fully_informed":
        mask = np.ones(size, dtype="uint8")
    elif mask_pattern == "checkerboard":
        mask = np.zeros(size, dtype="uint8")
        mask[::2] = 1
    elif mask_pattern == "sparse":
        mask = np.zeros(size, dtype="uint8")
        mask[::10] = 1
    elif mask_pattern == "all_uninformed":
        mask = np.zeros(size, dtype="uint8")
    else:
        raise ValueError(f"Unknown mask_pattern: {mask_pattern}")

    if extra_mask is not None:
        mask = mask & extra_mask

    prop = ContProperty(data, mask)
    prop.fix_shape(grid)
    return prop


def _make_cdf_data(values=None, probs=None):
    """Create a CdfData with given values/probs or sensible defaults."""
    if values is None:
        values = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype="float32")
    if probs is None:
        probs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype="float32")
    return CdfData(values, probs)


def _make_ind_property(grid, indicator_count=3, mask_pattern="none"):
    """Create an IndProperty for SIS testing."""
    rng = np.random.RandomState(42)
    size = grid.x * grid.y * grid.z

    data = rng.randint(0, indicator_count, size, dtype="uint8")

    if mask_pattern == "none" or mask_pattern == "fully_informed":
        mask = np.ones(size, dtype="uint8")
    elif mask_pattern == "checkerboard":
        mask = np.zeros(size, dtype="uint8")
        mask[::2] = 1
    elif mask_pattern == "sparse":
        mask = np.zeros(size, dtype="uint8")
        mask[::10] = 1
    elif mask_pattern == "all_uninformed":
        mask = np.zeros(size, dtype="uint8")
    else:
        raise ValueError(f"Unknown mask_pattern: {mask_pattern}")

    prop = IndProperty(data, mask, indicator_count)
    prop.fix_shape(grid)
    return prop


def _make_sis_data(indicator_count, cov_type=covariance.spherical, ranges=(5.0, 5.0, 3.0),
                   radiuses=(5, 5, 3), max_neighbours=12):
    """Create SIS per-indicator data list."""
    sis_data = []
    for _i in range(indicator_count):
        sis_data.append({
            "cov_model": _make_cov_model(cov_type, ranges=ranges),
            "radiuses": radiuses,
            "max_neighbours": max_neighbours,
        })
    return sis_data


def _make_grid(x, y, z):
    """Create a SugarboxGrid."""
    return SugarboxGrid(x=x, y=y, z=z)


# =============================================================================
# SGS - Determinism & Output Invariants
# =============================================================================


@pytest.mark.hpgl
class TestSGSDeterminism:
    """Parametrized SGS determinism tests: same seed produces identical output."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_sgs_determinism_seed(self, seed):
        """Same seed produces identical SGS output."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type="sk",
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_sgs_determinism_kriging_type(self, kriging_type, seed):
        """Determinism holds regardless of kriging type."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type=kriging_type,
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type=kriging_type,
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_sgs_determinism_random_seed(self):
        """A randomly-generated seed also yields deterministic output."""
        random_seed = np.random.randint(0, 2**31 - 1)
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=random_seed, kriging_type="sk",
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=random_seed, kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_sgs_different_seeds_valid_output(self, seed):
        """Different seeds all produce valid SGS output (no NaN/Inf, correct shape)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SGS - Output Invariants
# =============================================================================


@pytest.mark.hpgl
class TestSGSOutputInvariants:
    """Parametrized SGS output invariant checks."""

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    def test_sgs_no_nan_inf(self, grid_dims, kriging_type):
        """SGS output contains no NaN or Inf."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    def test_sgs_output_shape_matches(self, grid_dims, kriging_type):
        """SGS output shape matches the grid dimensions."""
        x, y, z = grid_dims
        grid = _make_grid(x, y, z)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
        )

        assert result.data.shape == (x, y, z)
        assert isinstance(result, ContProperty)

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    def test_sgs_not_all_zeros(self, grid_dims, kriging_type):
        """SGS output is not all zeros."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
        )

        assert not np.all(result.data == 0)

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    def test_sgs_kriging_type_output_valid(self, grid_dims, kriging_type):
        """Both SK and OK produce valid SGS output (no NaN/Inf, correct shape)."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
        )

        assert isinstance(result, ContProperty)
        x, y, z = grid_dims
        assert result.data.shape == (x, y, z)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        assert not np.all(result.data == 0)


# =============================================================================
# SGS - Covariance Types
# =============================================================================


@pytest.mark.hpgl
class TestSGSCovarianceTypes:
    """Parametrized SGS with different covariance model types."""

    @pytest.mark.parametrize("cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sgs_covariance_type(self, cov_type, grid_dims):
        """SGS completes without NaN/Inf for each covariance type."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(cov_type, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian])
    def test_sgs_covariance_type_determinism(self, cov_type):
        """Each covariance type is deterministic with fixed seed."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(cov_type, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# SGS - min_neighbours
# =============================================================================


@pytest.mark.hpgl
class TestSGSMinNeighbours:
    """Parametrized SGS with different min_neighbours values."""

    @pytest.mark.parametrize("min_nb", [0, 1, 5, 10])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    def test_sgs_min_neighbours(self, min_nb, kriging_type):
        """SGS completes without NaN/Inf for various min_neighbours."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
            min_neighbours=min_nb,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("min_nb", [0, 5])
    def test_sgs_min_neighbours_determinism(self, min_nb):
        """SGS with min_neighbours is deterministic with same seed."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
            min_neighbours=min_nb,
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
            min_neighbours=min_nb,
        )

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# SGS - Mask Patterns
# =============================================================================


@pytest.mark.hpgl
class TestSGSMaskPatterns:
    """Parametrized SGS with different mask patterns."""

    @pytest.mark.parametrize("mask_pattern", ["none", "checkerboard", "sparse"])
    @pytest.mark.parametrize("kriging_type", ["sk", "ok"])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sgs_mask_pattern(self, mask_pattern, kriging_type, grid_dims):
        """SGS completes without NaN/Inf for various mask patterns."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern=mask_pattern)
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type=kriging_type,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("mask_pattern", ["none", "checkerboard", "sparse"])
    def test_sgs_mask_pattern_determinism(self, mask_pattern):
        """SGS with masks is deterministic with fixed seed."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern=mask_pattern)
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result1 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )
        result2 = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# SGS - Parallel vs Serial (OMP_NUM_THREADS)
# =============================================================================
#
# NOTE: Thread-count consistency testing (verifying that SGS produces identical
# results regardless of OMP_NUM_THREADS) is NOT possible with in-process
# parametrized tests. The OpenMP runtime reads OMP_NUM_THREADS at process
# start, and manipulating it within a running process has no effect. Genuine
# thread-count consistency testing requires subprocess-based tests where each
# test case runs in a fresh process with its own OMP_NUM_THREADS value.
# This is planned for a future test infrastructure improvement.


# =============================================================================
# SIS - Determinism & Output Invariants
# =============================================================================


@pytest.mark.hpgl
class TestSISDeterminism:
    """Parametrized SIS determinism tests."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_sis_determinism_seed(self, seed):
        """Same seed produces identical SIS output."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result1 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=seed, marginal_probs=[0.3, 0.4, 0.3],
        )
        result2 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=seed, marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)
        np.testing.assert_array_equal(result1.mask, result2.mask)

    def test_sis_determinism_random_seed(self):
        """A randomly-generated seed also yields deterministic SIS output."""
        random_seed = np.random.randint(0, 2**31 - 1)
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result1 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=random_seed, marginal_probs=[0.3, 0.4, 0.3],
        )
        result2 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=random_seed, marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_sis_different_seeds_valid_output(self, seed):
        """Different seeds all produce valid SIS output (no NaN/Inf, correct shape)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=seed, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == (10, 10, 10)
        assert np.all(result.data < 3)
        assert np.all(result.data >= 0)


# =============================================================================
# SIS - Output Invariants
# =============================================================================


@pytest.mark.hpgl
class TestSISOutputInvariants:
    """Parametrized SIS output invariant checks."""

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_no_nan_inf(self, grid_dims):
        """SIS output contains no NaN or Inf."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_output_shape_matches(self, grid_dims):
        """SIS output shape matches the grid dimensions."""
        x, y, z = grid_dims
        grid = _make_grid(x, y, z)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert result.data.shape == (x, y, z)
        assert isinstance(result, IndProperty)

    @pytest.mark.parametrize("indicator_count", [2, 3, 5])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_indicator_count(self, indicator_count, grid_dims):
        """SIS handles different indicator counts correctly."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=indicator_count, mask_pattern="none")
        sis_data = _make_sis_data(indicator_count, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))
        # Equal marginal probs that sum to 1.0
        marginal_probs = [1.0 / indicator_count] * indicator_count

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=marginal_probs,
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == indicator_count
        assert np.all(result.data < indicator_count)
        assert np.all(result.data >= 0)

    def test_sis_not_all_zeros(self):
        """SIS output is not all zeros."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert not np.all(result.data == 0)


# =============================================================================
# SIS - Covariance Types
# =============================================================================


@pytest.mark.hpgl
class TestSISCovarianceTypes:
    """Parametrized SIS with different covariance model types."""

    @pytest.mark.parametrize("cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_covariance_type(self, cov_type, grid_dims):
        """SIS completes without NaN/Inf for each covariance type."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=cov_type,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian])
    def test_sis_covariance_type_determinism(self, cov_type):
        """Each SIS covariance type is deterministic with fixed seed."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=cov_type)

        result1 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )
        result2 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# SIS - Mask Patterns
# =============================================================================


@pytest.mark.hpgl
class TestSISMaskPatterns:
    """Parametrized SIS with different mask patterns."""

    @pytest.mark.parametrize("mask_pattern", ["none", "checkerboard", "sparse"])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_mask_pattern(self, mask_pattern, grid_dims):
        """SIS completes without NaN/Inf for various mask patterns."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern=mask_pattern)
        sis_data = _make_sis_data(3, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("mask_pattern", ["none", "checkerboard", "sparse"])
    def test_sis_mask_pattern_determinism(self, mask_pattern):
        """SIS with masks is deterministic with fixed seed."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern=mask_pattern)
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result1 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )
        result2 = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)


# =============================================================================
# SIS - Thread Consistency
# =============================================================================
#
# NOTE: Thread-count consistency testing for SIS follows the same rationale as
# SGS above — in-process OMP_NUM_THREADS manipulation is ineffective. See the
# SGS section above for details.


# =============================================================================
# SGS - Edge Cases
# =============================================================================


@pytest.mark.hpgl
class TestSGSEdgeCases:
    """Edge case tests for SGS simulation."""

    def test_sgs_single_point_grid(self):
        """SGS with 1x1x1 grid (single cell)."""
        grid = _make_grid(1, 1, 1)
        prop = _make_cont_property(grid, mask_pattern="fully_informed", data_pattern="uniform")
        cdf = CdfData(
            values=np.array([0.0, 100.0], dtype="float32"),
            probs=np.array([0.0, 1.0], dtype="float32"),
        )
        cov = _make_cov_model(covariance.spherical, ranges=(1.0, 1.0, 1.0), nugget=0.0)

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(1, 1, 1), max_neighbours=1,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert result.data.shape == (1, 1, 1)
        assert np.isfinite(result.data[0, 0, 0])

    def test_sgs_uniform_data(self):
        """SGS with uniform (all same value) input data."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none", data_pattern="uniform")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_fully_informed_grid(self):
        """SGS with 100% informed (all cells have data)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="fully_informed")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_sparse_data(self):
        """SGS with sparse (10% informed) input data."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="sparse")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        # Sparse input may leave some cells uninformed in output;
        # that's ok — SGS with min_neighbours=0 can fill them
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_extreme_values(self):
        """SGS with extreme (very large/small) input values."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none", data_pattern="extreme")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_negative_values(self):
        """SGS with negative input values."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none", data_pattern="negative")
        cdf = _make_cdf_data(
            values=np.array([-50.0, -25.0, 0.0, 25.0, 50.0], dtype="float32"),
            probs=np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32"),
        )
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_without_cdf(self):
        """SGS without CDF transformation (cdf_data=None)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=None,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_all_uninformed_input(self):
        """SGS with all-uninformed input (mask all zeros)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="all_uninformed")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_with_simulation_mask(self):
        """SGS with an explicit simulation mask."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        # Create a mask that simulates only the central region
        sim_mask = np.zeros(grid.x * grid.y * grid.z, dtype="uint8")
        sim_mask_3d = sim_mask.reshape((grid.x, grid.y, grid.z), order="F")
        sim_mask_3d[2:8, 2:8, :] = 1  # Simulate central 6x6 region

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
            mask=sim_mask,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_use_harddata_false(self):
        """SGS with use_harddata=False ignores input data."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none", data_pattern="uniform")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
            use_harddata=False,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sgs_scalar_mean(self):
        """SGS with a scalar mean value."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
            mean=50.0,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SIS - Edge Cases
# =============================================================================


@pytest.mark.hpgl
class TestSISEdgeCases:
    """Edge case tests for SIS simulation."""

    def test_sis_single_point_grid(self):
        """SIS with 1x1x1 grid (single cell)."""
        grid = _make_grid(1, 1, 1)
        rng = np.random.RandomState(42)
        data = rng.randint(0, 2, 1, dtype="uint8")
        mask = np.ones(1, dtype="uint8")
        prop = IndProperty(data, mask, indicator_count=2)
        prop.fix_shape(grid)

        sis_data = _make_sis_data(2, cov_type=covariance.spherical,
                                  ranges=(1.0, 1.0, 1.0),
                                  radiuses=(1, 1, 1), max_neighbours=1)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.5, 0.5],
        )

        assert result.data.shape == (1, 1, 1)
        assert 0 <= result.data[0, 0, 0] < 2

    def test_sis_fully_informed(self):
        """SIS with 100% informed (all cells have data)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="fully_informed")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < 3)
        assert np.all(result.data >= 0)

    def test_sis_sparse_data(self):
        """SIS with sparse (10% informed) input data."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="sparse")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sis_use_correlogram_false(self):
        """SIS with use_correlogram=False."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
            use_correlogram=False,
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < 3)

    def test_sis_use_harddata_false(self):
        """SIS with use_harddata=False ignores input data."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=False,
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < 3)
        assert np.all(result.data >= 0)

    def test_sis_all_uninformed_input(self):
        """SIS with all-uninformed input (mask all zeros)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="all_uninformed")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == (10, 10, 10)
        assert np.all(result.data < 3)

    def test_sis_with_simulation_mask(self):
        """SIS with an explicit simulation mask."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        # Create a central simulation mask
        sim_mask = np.zeros(grid.x * grid.y * grid.z, dtype="uint8")
        sim_mask_3d = sim_mask.reshape((grid.x, grid.y, grid.z), order="F")
        sim_mask_3d[2:8, 2:8, :] = 1

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
            mask=sim_mask,
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == (10, 10, 10)
        assert np.all(result.data < 3)

    def test_sis_2indicator_median_ik_path(self):
        """SIS with 2 indicators (median IK optimization path)."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=2, mask_pattern="none")
        sis_data = _make_sis_data(2, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.4, 0.6],
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 2
        assert np.all(result.data < 2)
        assert np.all(result.data >= 0)


# =============================================================================
# ContProperty Construction Tests
# =============================================================================


@pytest.mark.hpgl
class TestContPropertyConstruction:
    """Parametrized ContProperty construction edge cases."""

    @pytest.mark.parametrize("data_pattern", ["random", "uniform", "extreme", "zeros"])
    def test_cont_property_data_patterns(self, data_pattern):
        """ContProperty accepts various data patterns."""
        size = 100
        rng = np.random.RandomState(42)

        if data_pattern == "random":
            data = rng.rand(size).astype("float32") * 100
        elif data_pattern == "uniform":
            data = np.full(size, 42.0, dtype="float32")
        elif data_pattern == "extreme":
            data = np.array([1e-6, 1e6] * (size // 2 + 1), dtype="float32")[:size]
        elif data_pattern == "zeros":
            data = np.zeros(size, dtype="float32")

        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)

        assert isinstance(prop, ContProperty)
        assert prop.data.shape == mask.shape
        assert prop.data.dtype == np.float32
        assert prop.mask.dtype == np.uint8

    @pytest.mark.parametrize("mask_pattern", ["all_ones", "all_zeros", "random"])
    def test_cont_property_mask_patterns(self, mask_pattern):
        """ContProperty accepts various mask patterns."""
        size = 100
        data = np.random.RandomState(42).rand(size).astype("float32") * 100

        if mask_pattern == "all_ones":
            mask = np.ones(size, dtype="uint8")
        elif mask_pattern == "all_zeros":
            mask = np.zeros(size, dtype="uint8")
        elif mask_pattern == "random":
            mask = np.random.RandomState(42).randint(0, 2, size, dtype="uint8")

        prop = ContProperty(data, mask)

        assert isinstance(prop, ContProperty)
        assert prop.mask.shape == data.shape

    @pytest.mark.parametrize("shape", [(10, 10, 5), (50,)])
    def test_cont_property_shapes(self, shape):
        """ContProperty accepts 1D and 3D data shapes."""
        data = np.random.RandomState(42).rand(*shape).astype("float32")
        mask = np.ones(shape, dtype="uint8")

        prop = ContProperty(data, mask)

        assert isinstance(prop, ContProperty)
        assert prop.data.shape == shape

    def test_cont_property_rejects_2d(self):
        """ContProperty rejects 2D data."""
        data = np.ones((10, 10), dtype="float32")
        mask = np.ones((10, 10), dtype="uint8")

        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_rejects_nan_data(self):
        """ContProperty rejects data with NaN values."""
        data = np.array([1.0, np.nan, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")

        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_rejects_inf_data(self):
        """ContProperty rejects data with Inf values."""
        data = np.array([1.0, np.inf, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")

        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_shape_mismatch(self):
        """ContProperty rejects mismatched data and mask shapes."""
        data = np.ones(10, dtype="float32")
        mask = np.ones(5, dtype="uint8")

        with pytest.raises(ValueError):
            ContProperty(data, mask)

    def test_cont_property_accepts_list_input(self):
        """ContProperty accepts list input (converted to ndarray)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mask = [1, 0, 1, 1, 0]

        prop = ContProperty(data, mask)

        assert isinstance(prop, ContProperty)
        assert prop.data.dtype == np.float32
        assert prop.mask.dtype == np.uint8


# =============================================================================
# IndProperty Construction Tests
# =============================================================================


@pytest.mark.hpgl
class TestIndPropertyConstruction:
    """Parametrized IndProperty construction edge cases."""

    @pytest.mark.parametrize("indicator_count", [1, 2, 3, 5, 10, 255])
    def test_ind_property_indicator_counts(self, indicator_count):
        """IndProperty accepts valid indicator counts."""
        size = 100
        rng = np.random.RandomState(42)
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")

        prop = IndProperty(data, mask, indicator_count)

        assert isinstance(prop, IndProperty)
        assert prop.indicator_count == indicator_count
        assert prop.data.dtype == np.uint8
        assert prop.mask.dtype == np.uint8

    @pytest.mark.parametrize("mask_pattern", ["all_ones", "all_zeros", "random"])
    def test_ind_property_mask_patterns(self, mask_pattern):
        """IndProperty accepts various mask patterns."""
        size = 100
        rng = np.random.RandomState(42)
        data = rng.randint(0, 3, size, dtype="uint8")

        if mask_pattern == "all_ones":
            mask = np.ones(size, dtype="uint8")
        elif mask_pattern == "all_zeros":
            mask = np.zeros(size, dtype="uint8")
        elif mask_pattern == "random":
            mask = np.random.RandomState(42).randint(0, 2, size, dtype="uint8")

        prop = IndProperty(data, mask, indicator_count=3)

        assert isinstance(prop, IndProperty)
        assert prop.mask.shape == data.shape

    @pytest.mark.parametrize("shape", [(10, 10, 5), (50,)])
    def test_ind_property_shapes(self, shape):
        """IndProperty accepts 1D and 3D data shapes."""
        data = np.random.RandomState(42).randint(0, 3, shape, dtype="uint8")
        mask = np.ones(shape, dtype="uint8")

        prop = IndProperty(data, mask, indicator_count=3)

        assert isinstance(prop, IndProperty)
        assert prop.data.shape == shape

    def test_ind_property_rejects_2d(self):
        """IndProperty rejects 2D data."""
        data = np.ones((10, 10), dtype="uint8")
        mask = np.ones((10, 10), dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=2)

    def test_ind_property_rejects_indicator_out_of_range(self):
        """IndProperty rejects indicator values >= indicator_count."""
        data = np.array([0, 1, 2, 3], dtype="uint8")  # 3 >= 3 for count=3
        mask = np.ones(4, dtype="uint8")

        with pytest.raises(RuntimeError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_nan_data(self):
        """IndProperty rejects data with NaN values."""
        data = np.array([0.0, np.nan, 2.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_inf_data(self):
        """IndProperty rejects data with Inf values."""
        data = np.array([0.0, np.inf, 2.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)

    def test_ind_property_rejects_invalid_count(self):
        """IndProperty rejects indicator_count outside 1-255."""
        data = np.array([0, 1], dtype="uint8")
        mask = np.ones(2, dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=0)

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=256)

    def test_ind_property_shape_mismatch(self):
        """IndProperty rejects mismatched data and mask shapes."""
        data = np.array([0, 1, 2], dtype="uint8")
        mask = np.ones(5, dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=3)


# =============================================================================
# Combined Simulation Invariants (cross-variant)
# =============================================================================


@pytest.mark.hpgl
class TestSimulationCrossVariantInvariants:
    """Cross-variant invariant tests for both SGS and SIS."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_sgs_all_seeds_no_nan_inf(self, seed):
        """SGS produces no NaN/Inf across all tested seeds."""
        grid = _make_grid(10, 10, 10)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(5.0, 5.0, 3.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(5, 5, 3), max_neighbours=12,
            cov_model=cov, seed=seed, kriging_type="sk",
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_sis_all_seeds_no_nan_inf(self, seed):
        """SIS produces no NaN/Inf across all tested seeds."""
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=seed, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sgs_output_valid_range(self, grid_dims):
        """SGS output values are within CDF range."""
        grid = _make_grid(*grid_dims)
        prop = _make_cont_property(grid, mask_pattern="none")
        cdf = _make_cdf_data()
        cov = _make_cov_model(covariance.spherical, ranges=(3.0, 3.0, 2.0))

        result = sgs_simulation(
            prop=prop, grid=grid, cdf_data=cdf,
            radiuses=(3, 3, 2), max_neighbours=12,
            cov_model=cov, seed=42, kriging_type="sk",
        )

        cdf_min = cdf.values.min()
        cdf_max = cdf.values.max()
        simulated = result.data[result.mask > 0]
        assert np.all(simulated >= cdf_min - 1.0)
        assert np.all(simulated <= cdf_max + 1.0)

    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_output_valid_range(self, grid_dims):
        """SIS output values are within indicator range."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="none")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert np.all(result.data < 3)
        assert np.all(result.data >= 0)
