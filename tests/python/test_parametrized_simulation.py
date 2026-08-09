"""
Parametrized tests for HPGL simulation algorithms:
- Sequential Gaussian Simulation (SGS)
- Sequential Indicator Simulation (SIS)

V-28 (65 -> 14): the vacuous invariant family (T-09) is deleted — the vast
majority of the original parametrized tests built fully-informed properties
(mask_pattern="none"), so the C++ kernels skipped every node and the
assertions held on the input clone. The same invariants are covered
non-vacuously by test_simulation_complete.py (10%-uninformed) and
test_edge_cases.py (50%-uninformed). The surviving tests cover genuinely
distinct simulation paths:
- mask-pattern rows (sparse/checkerboard actually simulate; SGS rows
  :463-481 assert on float32 output where no-NaN is meaningful — kept
  unchanged per N2-10; SIS rows :714-730 hardened to range asserts per
  N2-10, since uint8 no-NaN is trivially true)
- sparse-data (10%-informed real simulation) and all-uninformed (pure
  marginal-draw path) for both algorithms
- use_harddata_false (empty-clone path — all cells simulated from marginals)
- unique constructor-rejection tests not covered by test_classes.py
  (2D reject, NaN/Inf reject, invalid indicator count)
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
# SGS - Mask Patterns
# =============================================================================


@pytest.mark.hpgl
class TestSGSMaskPatterns:
    """Parametrized SGS with different mask patterns.

    N2-10: the SGS rows are kept unchanged — ContProperty data is float32,
    so the no-NaN/Inf assertions are meaningful (not trivially true like the
    uint8 SIS rows). The checkerboard/sparse rows genuinely simulate; the
    "none" row is fully-informed and retained as the baseline.
    """

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


# =============================================================================
# SIS - Mask Patterns
# =============================================================================


@pytest.mark.hpgl
class TestSISMaskPatterns:
    """Parametrized SIS with different mask patterns.

    N2-10: the SIS mask-pattern rows are HARDENED to range asserts only. The
    original no-NaN/Inf assertions were trivially true on uint8 indicator
    output (uint8 cannot hold NaN/Inf), so an out-of-range category (e.g.
    255) would pass. The checkerboard/sparse rows genuinely simulate and now
    assert the simulated categories stay within [0, indicator_count).
    """

    @pytest.mark.parametrize("mask_pattern", ["none", "checkerboard", "sparse"])
    @pytest.mark.parametrize("grid_dims", [(4, 4, 4), (10, 10, 10)])
    def test_sis_mask_pattern(self, mask_pattern, grid_dims):
        """SIS completes with valid indicator categories for various mask patterns."""
        grid = _make_grid(*grid_dims)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern=mask_pattern)
        sis_data = _make_sis_data(3, cov_type=covariance.spherical,
                                  ranges=(3.0, 3.0, 2.0), radiuses=(3, 3, 2))

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        # N2-10: range asserts — the meaningful discriminator on uint8 output.
        assert np.all(result.data < result.indicator_count)
        assert np.all(result.data >= 0)


# =============================================================================
# SGS - Edge Cases
# =============================================================================


@pytest.mark.hpgl
class TestSGSEdgeCases:
    """Edge case tests for SGS simulation."""

    def test_sgs_sparse_data(self):
        """SGS with sparse (10% informed) input data.

        R-14: kept as the only genuine sparse-regime SGS smoke after the
        vacuous invariant family is deleted (10%-informed -> real simulation).
        """
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

    def test_sgs_all_uninformed_input(self):
        """SGS with all-uninformed input (mask all zeros).

        R-17: every node takes the KI_NO_NEIGHBOURS marginal-draw path
        (sequential_simulation.h:393-395) — a genuinely distinct C++ path with
        no sibling coverage elsewhere.
        """
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


# =============================================================================
# SIS - Edge Cases
# =============================================================================


@pytest.mark.hpgl
class TestSISEdgeCases:
    """Edge case tests for SIS simulation."""

    def test_sis_sparse_data(self):
        """SIS with sparse (10% informed) input data.

        R-22/N2-09: kept as the genuine sparse-regime SIS path guard. The
        original no-NaN/Inf assertions are trivially true on uint8 indicator
        output; hardened with range asserts so an out-of-range category
        (e.g. 255 written by a broken marginal fallback) fails loudly.
        """
        grid = _make_grid(10, 10, 10)
        prop = _make_ind_property(grid, indicator_count=3, mask_pattern="sparse")
        sis_data = _make_sis_data(3, cov_type=covariance.spherical)

        result = sis_simulation(
            prop=prop, grid=grid, data=sis_data,
            seed=42, marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        # N2-09: range asserts — the meaningful discriminator on uint8 output.
        assert np.all(result.data < result.indicator_count)
        assert np.all(result.data >= 0)

    def test_sis_use_harddata_false(self):
        """SIS with use_harddata=False ignores input data.

        R-24: with an empty-clone path all cells are simulated from the
        marginals (sis.cpp:133-137) — a distinct path with no exact sibling.
        """
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
        """SIS with all-uninformed input (mask all zeros).

        R-25: pure marginal-draw path (KI_NO_NEIGHBOURS -> marginal
        fallback, sis.cpp:133-137) — distinct; asserts real category range.
        """
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


# =============================================================================
# ContProperty Construction Tests (unique rejects only)
# =============================================================================


@pytest.mark.hpgl
class TestContPropertyConstruction:
    """Unique ContProperty constructor-rejection pins.

    R-29/R-30: the happy-path constructor matrix duplicated test_classes.py
    and is deleted; only the rejects NOT covered by test_classes.py survive
    (2D shape, NaN data, Inf data).
    """

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


# =============================================================================
# IndProperty Construction Tests (unique rejects only)
# =============================================================================


@pytest.mark.hpgl
class TestIndPropertyConstruction:
    """Unique IndProperty constructor-rejection pins.

    R-29/R-30/R-34: the happy-path constructor matrix duplicated
    test_classes.py and is deleted; only the rejects NOT covered by
    test_classes.py survive (2D shape, NaN/Inf data, invalid count bounds).
    """

    def test_ind_property_rejects_2d(self):
        """IndProperty rejects 2D data."""
        data = np.ones((10, 10), dtype="uint8")
        mask = np.ones((10, 10), dtype="uint8")

        with pytest.raises(ValueError):
            IndProperty(data, mask, indicator_count=2)

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
