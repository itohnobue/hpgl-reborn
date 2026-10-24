"""
Comprehensive tests for ALL kriging algorithms in HPGL.

Note: test_kriging.py (deprecated) was removed in v1.5.0 cleanup (D96).
Its 4 tests were fully covered here with better fixtures and parameterized variants.

Tests cover:
1. ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model)
2. simple_kriging(prop, grid, radiuses, max_neighbours, cov_model, mean=None)
3. lvm_kriging(prop, grid, mean_data, radiuses, max_neighbours, cov_model)
4. indicator_kriging(prop, grid, data, marginal_probs)
5. median_ik(prop, grid, marginal_probs, radiuses, max_neighbours, cov_model)
6. simple_cokriging_markI(prop, grid, radiuses, max_neighbours, cov_model, secondary_data, primary_mean, secondary_mean, secondary_variance, correlation_coef)
7. simple_cokriging_markII(grid, primary_data, secondary_data, correlation_coef, radiuses, max_neighbours)
8. simple_kriging_weights(center_point, n_x, n_y, n_z, ranges, sill, cov_type, nugget, angles)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        indicator_kriging,
        lvm_kriging,
        median_ik,
        ordinary_kriging,
        simple_cokriging_markI,
        simple_cokriging_markII,
        simple_kriging,
        simple_kriging_weights,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def krig_small_grid():
    """Create a small 3D grid for kriging tests"""
    return SugarboxGrid(x=5, y=5, z=3)


@pytest.fixture
def krig_medium_grid():
    """Create a medium 3D grid for kriging tests"""
    return SugarboxGrid(x=10, y=10, z=5)


@pytest.fixture
def continuous_property_small(krig_small_grid):
    """Create continuous property for small grid"""
    np.random.seed(42)
    size = krig_small_grid.x * krig_small_grid.y * krig_small_grid.z
    data = np.random.rand(size).astype("float32") * 100
    mask = np.ones(size, dtype="uint8")
    mask[::5] = 0  # 20% uninformed
    return ContProperty(data, mask)


@pytest.fixture
def continuous_property_medium(krig_medium_grid):
    """Create continuous property for medium grid"""
    np.random.seed(42)
    size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
    data = np.random.rand(size).astype("float32") * 100
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0  # 10% uninformed
    return ContProperty(data, mask)


@pytest.fixture
def indicator_property_small(krig_small_grid):
    """Create indicator property for small grid"""
    np.random.seed(42)
    size = krig_small_grid.x * krig_small_grid.z * krig_small_grid.y
    data = np.random.randint(0, 3, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::5] = 0
    return IndProperty(data, mask, 3)


@pytest.fixture
def indicator_property_2cat_small(krig_small_grid):
    """Create 2-category indicator property for small grid (median_ik requires exactly 2)."""
    np.random.seed(42)
    size = krig_small_grid.x * krig_small_grid.z * krig_small_grid.y
    data = np.random.randint(0, 2, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::5] = 0
    return IndProperty(data, mask, 2)


@pytest.fixture
def indicator_property_2cat_medium(krig_medium_grid):
    """Create 2-category indicator property for medium grid (median_ik requires exactly 2)."""
    np.random.seed(42)
    size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
    data = np.random.randint(0, 2, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0
    return IndProperty(data, mask, 2)


@pytest.fixture
def indicator_property_medium(krig_medium_grid):
    """Create indicator property for medium grid"""
    np.random.seed(42)
    size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
    data = np.random.randint(0, 3, size, dtype="uint8")
    mask = np.ones(size, dtype="uint8")
    mask[::10] = 0
    return IndProperty(data, mask, 3)


@pytest.fixture
def covariance_spherical():
    """Spherical covariance model"""
    return CovarianceModel(
        type=covariance.spherical,
        ranges=(5.0, 5.0, 3.0),
        angles=(0.0, 0.0, 0.0),
        sill=1.0,
        nugget=0.1,
    )


@pytest.fixture
def mean_data_medium(krig_medium_grid):
    """Create mean data array for LVM kriging"""
    np.random.seed(42)
    size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
    return np.random.rand(size).astype("float32") * 50


@pytest.fixture
def secondary_property_medium(krig_medium_grid):
    """Create secondary property for cokriging.

    G-50: the secondary is read ONLY at the nodes the kernel interpolates
    (primary-uninformed cells) — ``simple_cokriging_markI.cpp:433-440`` sets
    ``secondary_present = secondary_data.is_informed(i)`` there, and the
    shared process_node_loop serves Mark II too. This fixture used to reuse
    the primary's ``mask[::10] = 0``, so the secondary was masked at exactly
    the 50 kriged nodes and every cokriging test ran primary-only (measured:
    output equal to ``simple_kriging(mean=50.0)`` to 1.5e-05). It is now
    fully informed, so a secondary datum exists wherever the kernel kriges.
    The absent-secondary branch stays covered by
    ``test_ck_markI_secondary_datum_changes_kriged_value`` (an all-masked
    copy of this data) and by the empty/mismatched-secondary negative tests.
    """
    np.random.seed(43)
    size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
    data = np.random.rand(size).astype("float32") * 80
    mask = np.ones(size, dtype="uint8")
    return ContProperty(data, mask)


@pytest.fixture
def neighbor_points():
    """Create neighbor points for weight calculation"""
    np.random.seed(42)
    n = 12
    n_x = np.random.rand(n).astype("float32") * 10
    n_y = np.random.rand(n).astype("float32") * 10
    n_z = np.random.rand(n).astype("float32") * 5
    return n_x, n_y, n_z


# =============================================================================
# Shared assertion helpers
# =============================================================================


def _kriged_cells(result, input_mask):
    """Result values at the cells that were UNINFORMED in the input mask.

    The continuous kernel copies informed cells through verbatim
    (cont_kriging.h:241-244) and only interpolates the uninformed ones, so a
    value assertion over the whole array is satisfied by the copy alone.

    The selection MUST come from the ORIGINAL input mask: the C++ fill writes
    ``mask = 1`` for every written node (property_array.h:57-70), so
    ``result.mask`` is all-ones after any kriging call and selects nothing.
    """
    uninformed = np.asarray(input_mask) == 0
    assert np.any(uninformed), (
        "fixture has no uninformed cells — the kriged-cell assertion would be vacuous"
    )
    return result.data.astype("float64")[uninformed]


def _assert_kriged_cells_not_all_zero(result, input_mask, context):
    """F-582: the kriged (originally-uninformed) cells must not all be zero."""
    kriged = _kriged_cells(result, input_mask)
    assert np.all(np.isfinite(kriged)), f"{context}: non-finite value in a kriged cell"
    assert np.any(kriged != 0.0), (
        f"{context}: every kriged (originally-uninformed) cell came back 0.0 — "
        "the kriging estimate was not written"
    )


# =============================================================================
# Ordinary Kriging Tests
# =============================================================================


@pytest.mark.hpgl
class TestOrdinaryKriging:
    """Comprehensive tests for Ordinary Kriging"""

    def test_ok_basic_execution(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test basic OK execution completes without errors"""
        result = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == continuous_property_medium.data.shape
        assert result.mask.shape == continuous_property_medium.mask.shape

    def test_ok_all_covariance_types(self, continuous_property_medium, krig_medium_grid):
        """Test OK with all covariance types (spherical, exponential, gaussian)"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            cov_model = CovarianceModel(
                type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
            )
            result = ordinary_kriging(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, ContProperty), f"Failed for {name}"
            assert not np.any(np.isnan(result.data))
            assert not np.any(np.isinf(result.data))
            # G-54: `result.mask != 0` is EMPTY after a no-op kernel (the
            # output mask starts zero-filled), and isfinite() over an empty
            # selection is vacuously True — assert the write happened first.
            assert np.count_nonzero(result.mask) > 0, f"no output written for {name}"
            assert np.all(np.isfinite(result.data[result.mask != 0]))

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_ok_various_neighbor_counts(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical, max_neighbours
    ):
        """Test OK with various neighbor counts"""
        result = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        # F-122 / F-582: guard against the all-zero regression on the cells the
        # kernel actually kriges (informed cells are copied through verbatim).
        _assert_kriged_cells_not_all_zero(
            result,
            continuous_property_medium.mask,
            f"OK with max_neighbours={max_neighbours}",
        )

    @pytest.mark.parametrize("radiuses", [(3, 3, 2), (5, 5, 3), (10, 10, 5), (15, 15, 8)])
    def test_ok_various_radiuses(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical, radiuses
    ):
        """Test OK with various search radius sizes"""
        result = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=radiuses,
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        # F-122 / F-582: guard against the all-zero regression on the kriged cells.
        _assert_kriged_cells_not_all_zero(
            result,
            continuous_property_medium.mask,
            f"OK with radiuses={radiuses}",
        )

    def test_ok_reproducibility(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test OK produces reproducible results"""
        np.random.seed(42)

        result1 = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.random.seed(42)
        result2 = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_ok_result_validation(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test OK produces valid results (no NaN, Inf, reasonable bounds)"""
        result = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        # Check for NaN and Inf
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

        # G-54: every other assertion here is passed by an all-zero output (0
        # is inside [min-1, max+1] for this fixture), so assert the kernel
        # actually wrote the kriged cells.
        _assert_kriged_cells_not_all_zero(
            result, continuous_property_medium.mask, "OK result validation"
        )

        # Check results are within reasonable bounds.
        # OK is a convex combination (weights sum to 1), so estimates MUST
        # stay within [min, max] of input data. ±1.0 tolerance accounts for
        # float32 precision in the full C++ pipeline.
        informed_mask = continuous_property_medium.mask == 1
        if np.any(informed_mask):
            original_min = np.min(continuous_property_medium.data[informed_mask])
            original_max = np.max(continuous_property_medium.data[informed_mask])
            assert np.all(result.data >= original_min - 1.0)
            assert np.all(result.data <= original_max + 1.0)

    def test_ok_small_grid(self, continuous_property_small, krig_small_grid, covariance_spherical):
        """Test OK with small grid"""
        result = ordinary_kriging(
            prop=continuous_property_small,
            grid=krig_small_grid,
            radiuses=(2, 2, 2),
            max_neighbours=4,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))
        # F-122 / F-582: guard the kriged cells, not the copy-through cells.
        _assert_kriged_cells_not_all_zero(result, continuous_property_small.mask, "OK small grid")

    def test_ok_with_nugget(self, continuous_property_medium, krig_medium_grid):
        """Test OK with various nugget values"""
        for nugget in [0.0, 0.1, 0.5, 1.0]:
            cov_model = CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=nugget,
            )
            result = ordinary_kriging(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, ContProperty)
            assert np.isfinite(result.data.astype("float64")).all()

    def test_ok_zero_nugget_exact_interpolation(self, krig_small_grid):
        """C23: OK with zero nugget reproduces the data exactly at informed cells.

        Ordinary Kriging with zero nugget is an exact interpolator:
        at an informed cell with a neighbor at the same location and weight=1,
        the estimate must equal the data value.

        F-572: the fixture is PARTIALLY informed and the uninformed cells hold
        the sentinel 0.0 in the input array. The kernel copies informed cells
        through (cont_kriging.h:241-244) and interpolates only the uninformed
        ones, so the kriged cells are distinguishable from the copy: every
        interpolation of the positive datum field must land far above the 0.0
        sentinel. An all-ones mask (the pre-fix fixture) kriged zero cells and
        proved nothing about the interpolator.
        """
        np.random.seed(42)
        size = krig_small_grid.x * krig_small_grid.y * krig_small_grid.z
        data = np.zeros(size, dtype="float32")
        mask = np.ones(size, dtype="uint8")
        mask[::10] = 0  # 20% uninformed -> the kernel must interpolate them
        data[mask == 1] = 50.0
        data[1] = 73.5  # Second known value, at an informed cell
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=prop,
            grid=krig_small_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
        )

        # At informed cells with zero nugget, OK must reproduce data values
        informed_mask = mask == 1
        informed_result = result.data[informed_mask].astype("float64")
        informed_data = data[informed_mask].astype("float64")

        # Tolerance: float32 precision for full pipeline
        np.testing.assert_allclose(
            informed_result,
            informed_data,
            rtol=1e-4,
            atol=1e-4,
            err_msg="OK with zero nugget must exactly reproduce input data at informed cells",
        )

        # The interpolated cells must actually be interpolated (never left at
        # the 0.0 sentinel the input array carries there).
        kriged = _kriged_cells(result, mask)
        assert np.all(np.isfinite(kriged)), "non-finite value at a kriged cell"
        assert np.all(kriged > 1.0), (
            f"OK left a kriged cell at {kriged.min():.4f} — the cell was copied "
            "through instead of interpolated (F-572)"
        )

    def test_ok_colocated_estimate_equals_data(self, krig_small_grid):
        """C23: OK at colocated point with zero nugget spherical model: estimate = data.

        With a single informed point on a 2x2x1 grid and the kriging target at
        the same cell, the OK estimate must equal the data value since the
        weight on the only neighbor is 1.0.

        F-572: index 0 is COPIED through (cont_kriging.h:241-244), so only the
        uninformed cells exercise the interpolator. With a single datum and
        zero nugget the OK estimate there is that datum (one neighbour, weight
        1), which is asserted below.
        """
        data = np.array([42.0, 0.0, 0.0, 0.0], dtype="float32")
        mask = np.array([1, 0, 0, 0], dtype="uint8")
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )

        # The colocated cell (index 0) must have estimate == data value
        assert abs(float(result.data[0]) - 42.0) < 1e-4, (
            f"Colocated OK estimate {result.data[0]} should equal data value 42.0"
        )

        # F-572: the KRIGED cells must equal the single datum too. A
        # copy-through regression leaves them at the 0.0 sentinel and fails.
        np.testing.assert_allclose(
            _kriged_cells(result, mask),
            42.0,
            rtol=0,
            atol=1e-4,
            err_msg="OK with one datum and zero nugget must krige every cell to that datum",
        )

    def test_ok_symmetric_two_point_estimate_is_exact(self):
        """F-572 / F-583: OK estimate is analytically exact on a symmetric layout.

        Grid x=3 holds 10.0 at cell 0 and 30.0 at cell 2; cell 1 is uninformed
        and equidistant from both data, so the OK system is symmetric: the two
        weights are equal, they sum to 1, and the estimate is exactly
        (10.0 + 30.0) / 2 = 20.0 — independent of the covariance type, its range
        and the nugget.

        This is the value-level OK oracle the file lacked: it reads a KRIGED
        cell (cell 1 is interpolated, never copied through) and a regression in
        the weight solve, in the sum-to-one constraint or in the Lagrange
        correction moves the value.
        """
        grid = SugarboxGrid(x=3, y=1, z=1)
        data = np.array([10.0, 0.0, 30.0], dtype="float32")
        mask = np.array([1, 0, 1], dtype="uint8")
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=ContProperty(data, mask),
            grid=grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
        )

        np.testing.assert_allclose(
            _kriged_cells(result, mask),
            20.0,
            rtol=0,
            atol=1e-3,
            err_msg=(
                "symmetric two-point OK estimate must be the midpoint 20.0 — "
                "weight solve or Lagrange correction regressed"
            ),
        )

    def test_ok_kriged_cells_use_covariance(self, continuous_property_medium, krig_medium_grid):
        """F-572: the KRIGED cells respond to the covariance range.

        Every other OK value pin in this file reads informed cells, which
        cont_kriging.h:241-244 copies through, so a covariance regression is
        invisible to them. This test restricts every assertion to the
        originally-uninformed cells and requires two different ranges to move
        them.
        """
        np.random.seed(42)
        cov_short = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )
        cov_long = CovarianceModel(
            type=covariance.spherical,
            ranges=(30.0, 30.0, 18.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result_short = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_short,
        )
        result_long = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_long,
        )

        kriged_short = _kriged_cells(result_short, continuous_property_medium.mask)
        kriged_long = _kriged_cells(result_long, continuous_property_medium.mask)
        assert np.all(np.isfinite(kriged_short))
        assert np.all(np.isfinite(kriged_long))

        # Anti-copy-through: the kriged cells must not reproduce the input
        # array that sits in those slots.
        input_at_kriged = continuous_property_medium.data.astype("float64")[
            continuous_property_medium.mask == 0
        ]
        assert not np.allclose(kriged_short, input_at_kriged, rtol=1e-4, atol=1e-4), (
            "kriged (originally-uninformed) cells repeat the input array — "
            "the kernel copied instead of interpolating"
        )

        assert not np.allclose(kriged_short, kriged_long, rtol=1e-6, atol=1e-6), (
            "covariance range 5 vs 30 left every kriged cell unchanged — the "
            "covariance model does not reach the kriging solve"
        )

    # F-05: Kriging variance regression protection — indirect tests since
    # HPGL API does not expose kriging variance (documented in hpgl_wrap.py:11-22).
    # These tests verify variance-related behavior: increasing nugget changes
    # results, and different covariance parameters produce measurably different estimates.

    def test_ok_nugget_increases_smoothing(self, continuous_property_medium, krig_medium_grid):
        """F-05: Increasing nugget increases kriging smoothness (reduces variance).

        The kriging variance σ² = C(0) − λᵀk decreases with larger nugget
        (weights become more uniform). This produces kriging estimates that
        are pulled toward the local mean — more "smooth".

        Verifies that changing nugget from 0.0 to 0.5 produces measurably
        different kriging results (variance is reduced by >0.5 → RMS change).
        """
        cov_low_nug = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )
        cov_high_nug = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.5,
        )

        result_low = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_low_nug,
        )

        result_high = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_high_nug,
        )

        # Both must be valid
        assert not np.any(np.isnan(result_low.data.astype("float64")))
        assert not np.any(np.isnan(result_high.data.astype("float64")))
        assert not np.any(np.isinf(result_low.data.astype("float64")))
        assert not np.any(np.isinf(result_high.data.astype("float64")))

        # Different nugget → measurably different results (smoothing effect)
        rms_diff = np.sqrt(
            np.mean((result_low.data.astype("float64") - result_high.data.astype("float64")) ** 2)
        )
        assert rms_diff > 1.0, (
            f"nugget change (0.0→0.5) should produce RMS > 1.0 change, got {rms_diff:.3f}"
        )

        # Higher nugget → result has lower variance (more smoothing toward mean)
        low_var = np.var(result_low.data.astype("float64"))
        high_var = np.var(result_high.data.astype("float64"))
        # Variance should decrease with higher nugget (smoothing pulls toward mean)
        assert high_var < low_var, (
            f"Higher nugget result variance ({high_var:.3f}) should be < "
            f"lower nugget variance ({low_var:.3f}) — nugget effect not visible"
        )

    # F-124: Cross-parameter effect verification
    def test_ok_neighbor_count_affects_result(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-124: max_neighbours=4 vs 16 produces measurably different kriging result.

        Different neighbor counts change which data points influence the estimate,
        so the results MUST differ — otherwise max_neighbours is ignored.
        """
        result4 = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=4,
            cov_model=covariance_spherical,
        )
        result16 = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=16,
            cov_model=covariance_spherical,
        )

        assert not np.allclose(
            result4.data.astype("float64"), result16.data.astype("float64"), rtol=1e-4, atol=1e-4
        ), "max_neighbours=4 vs 16: results should differ, parameter effect unverified"

    def test_ok_radius_affects_result(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-124: radius=(1,1,1) vs (15,15,8) produces measurably different result.

        Different search radii change the neighborhood of informed data, so the
        estimates MUST differ — otherwise the radius parameter is ignored.

        NOTE: the pair must be discriminating on the dense 10x10x5 fixture —
        every target cell has >=36 informed candidates within radius (3,3,2),
        so that pair selects the same 12 nearest neighbors under
        max_neighbours=12 (bit-identical output). Radius (1,1,1) is small
        enough that every target cell is neighborhood-constrained (<12
        candidates), forcing a genuinely different neighborhood than (15,15,8).
        """
        result_small = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(1, 1, 1),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        result_large = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(15, 15, 8),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert not np.allclose(
            result_small.data.astype("float64"),
            result_large.data.astype("float64"),
            rtol=1e-4,
            atol=1e-4,
        ), "radius (1,1,1) vs (15,15,8): results should differ, parameter effect unverified"

    def test_ok_covariance_type_affects_result(self, continuous_property_medium, krig_medium_grid):
        """F-124: spherical vs exponential produces measurably different kriging result."""
        cov_sph = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )
        cov_exp = CovarianceModel(
            type=covariance.exponential,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result_sph = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_sph,
        )
        result_exp = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_exp,
        )

        assert not np.allclose(
            result_sph.data.astype("float64"),
            result_exp.data.astype("float64"),
            rtol=1e-4,
            atol=1e-4,
        ), "spherical vs exponential: results should differ, covariance type ignored?"

    def test_ok_various_nugget_changes_estimate(self, continuous_property_medium, krig_medium_grid):
        """F-05: Each nugget value changes the kriging estimate measurably.

        Verifies that four different nugget values (0.0, 0.1, 0.5, 1.0) all
        produce results that differ from each other by at least some tolerance.
        """
        results = {}
        for nugget in [0.0, 0.1, 0.5, 1.0]:
            cov_model = CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=nugget,
            )
            result = ordinary_kriging(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, ContProperty)
            assert np.all(np.isfinite(result.data.astype("float64")))
            # Not all-zero (variance regression protection) — F-582: read the
            # kriged cells, not the copied-through informed cells.
            _assert_kriged_cells_not_all_zero(
                result, continuous_property_medium.mask, f"nugget={nugget}"
            )
            results[nugget] = result.data.astype("float64")

        # Each nugget pair should produce different results
        nugget_values = [0.0, 0.1, 0.5, 1.0]
        for i in range(len(nugget_values)):
            for j in range(i + 1, len(nugget_values)):
                ng_i = nugget_values[i]
                ng_j = nugget_values[j]
                rms = np.sqrt(np.mean((results[ng_i] - results[ng_j]) ** 2))
                assert rms > 0.05, (
                    f"nugget {ng_i} vs {ng_j}: RMS diff {rms:.6f} too small, "
                    f"nugget change should affect result"
                )

    # F-36: Golden-file comparison test for kriging regression
    def test_ok_golden_file_reproducible(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-36: OK result is reproducible and matches known reference values.

        Performs OK kriging with fixed seed/parameters and compares the first
        few INFORMED-cell values against hard-coded reference values.

        Scope (F-581): the kernel copies informed cells through verbatim
        (cont_kriging.h:241-244), so these reference values equal the fixture's
        own input data — this test pins the copy-through branch and the fixture,
        NOT the kriging mathematics. A change to the weights, the Lagrange
        correction or the covariance model does not move them; that oracle is
        test_ok_kriged_cells_use_covariance / test_ok_symmetric_two_point_estimate_is_exact.

        The reference values are the first 5 values at cells informed in
        the ORIGINAL input mask (mask==1), captured from a verified v2.0.5
        run (seed=42, spherical covariance, medium grid). The test uses
        float32-compatible tolerance to avoid false positives from
        platform-specific LAPACK differences.
        """
        np.random.seed(42)
        result = ordinary_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert isinstance(result, ContProperty)

        # Golden reference: first 5 informed-cell values from the
        # known-good run. The result mask is mutated to all-ones by the C++
        # fill, so the ORIGINAL input mask selects the truly informed cells.
        input_informed = continuous_property_medium.mask == 1
        assert np.any(input_informed), "Fixture should have informed cells"
        first_values = result.data.astype("float64")[input_informed][:5]
        assert len(first_values) == 5, "Expected at least 5 informed cells"

        golden = np.array([95.071426, 73.199394, 59.865849, 15.601865, 15.599452])
        np.testing.assert_allclose(
            first_values,
            golden,
            rtol=1e-3,
            atol=0.02,
            err_msg="OK golden reference mismatch — copy-through branch or fixture changed",
        )

        # F-572: the assertion above reads COPY-THROUGH cells. The cells the
        # kernel actually interpolates must not simply repeat the input array.
        uninformed = continuous_property_medium.mask == 0
        kriged = result.data.astype("float64")[uninformed]
        input_at_kriged = continuous_property_medium.data.astype("float64")[uninformed]
        assert not np.allclose(kriged, input_at_kriged, rtol=1e-4, atol=1e-4), (
            "kriged (originally-uninformed) cells repeat the input array — "
            "the kernel copied instead of interpolating"
        )


# =============================================================================
# Simple Kriging Tests
# =============================================================================


@pytest.mark.hpgl
class TestSimpleKriging:
    """Comprehensive tests for Simple Kriging"""

    def test_sk_basic_execution(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test basic SK execution completes without errors"""
        result = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=None,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == continuous_property_medium.data.shape
        # F-122 / F-582: guard the kriged cells, not the copy-through cells.
        _assert_kriged_cells_not_all_zero(
            result, continuous_property_medium.mask, "SK basic execution"
        )

    def test_sk_all_covariance_types(self, continuous_property_medium, krig_medium_grid):
        """Test SK with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            cov_model = CovarianceModel(
                type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
            )
            result = simple_kriging(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
                mean=None,
            )
            assert isinstance(result, ContProperty), f"Failed for {name}"

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_sk_various_neighbor_counts(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical, max_neighbours
    ):
        """Test SK with various neighbor counts"""
        result = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
            cov_model=covariance_spherical,
            mean=None,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
        assert np.all(np.isfinite(result.data[result.mask != 0]))
        # F-122 / F-582: guard the kriged cells, not the copy-through cells.
        _assert_kriged_cells_not_all_zero(
            result,
            continuous_property_medium.mask,
            f"SK with max_neighbours={max_neighbours}",
        )

    # F-575 / F-585: SK parameter-effect pin. The F-124 effect family covered
    # OK only; SK's max_neighbours was swept but never asserted to have an
    # effect on the result.
    def test_sk_neighbor_count_affects_result(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-575: SK max_neighbours=4 vs 16 changes the KRIGED cells.

        Different neighbour counts change which data points influence the
        estimate, so the kriged cells MUST differ — otherwise SK ignores the
        parameter. Asserted on the originally-uninformed cells only: informed
        cells are copied through and cannot move.
        """
        result4 = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=4,
            cov_model=covariance_spherical,
            mean=None,
        )
        result16 = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=16,
            cov_model=covariance_spherical,
            mean=None,
        )

        kriged4 = _kriged_cells(result4, continuous_property_medium.mask)
        kriged16 = _kriged_cells(result16, continuous_property_medium.mask)
        assert not np.allclose(kriged4, kriged16, rtol=1e-4, atol=1e-4), (
            "SK max_neighbours=4 vs 16 left every kriged cell unchanged — "
            "parameter effect unverified"
        )

    def test_sk_explicit_mean(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test SK with explicit mean value"""
        mean = 50.0
        result = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=mean,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sk_automatic_mean(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test SK with automatic mean calculation"""
        result = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=None,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sk_reproducibility(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test SK produces reproducible results"""
        np.random.seed(42)

        result1 = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=None,
        )

        np.random.seed(42)
        result2 = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=None,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_sk_result_validation(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """Test SK produces valid results"""
        result = simple_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
            mean=None,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_sk_small_grid(self, continuous_property_small, krig_small_grid, covariance_spherical):
        """Test SK with small grid"""
        result = simple_kriging(
            prop=continuous_property_small,
            grid=krig_small_grid,
            radiuses=(2, 2, 2),
            max_neighbours=4,
            cov_model=covariance_spherical,
            mean=None,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# LVM Kriging Tests
# =============================================================================


@pytest.mark.hpgl
class TestLVMKriging:
    """Comprehensive tests for Locally Varying Mean (LVM) Kriging"""

    def test_lvm_basic_execution(
        self, continuous_property_medium, krig_medium_grid, mean_data_medium, covariance_spherical
    ):
        """Test basic LVM kriging execution"""
        result = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == continuous_property_medium.data.shape

    def test_lvm_all_covariance_types(
        self, continuous_property_medium, krig_medium_grid, mean_data_medium
    ):
        """Test LVM kriging with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            cov_model = CovarianceModel(
                type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
            )
            result = lvm_kriging(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                mean_data=mean_data_medium,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, ContProperty), f"Failed for {name}"
            assert not np.any(np.isnan(result.data.astype("float64"))), f"NaN in {name}"

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_lvm_various_neighbor_counts(
        self,
        continuous_property_medium,
        krig_medium_grid,
        mean_data_medium,
        covariance_spherical,
        max_neighbours,
    ):
        """Test LVM kriging with various neighbor counts"""
        result = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    # F-575: LVM's parameter plumbing was swept but never asserted to matter.
    def test_lvm_neighbor_count_affects_result(
        self, continuous_property_medium, krig_medium_grid, mean_data_medium, covariance_spherical
    ):
        """F-575: LVM max_neighbours=4 vs 16 must move the KRIGED cells."""
        result4 = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=4,
            cov_model=covariance_spherical,
        )
        result16 = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=16,
            cov_model=covariance_spherical,
        )
        assert not np.allclose(
            _kriged_cells(result4, continuous_property_medium.mask),
            _kriged_cells(result16, continuous_property_medium.mask),
            rtol=1e-4,
            atol=1e-4,
        ), (
            "LVM max_neighbours=4 vs 16 left every kriged cell unchanged — "
            "parameter effect unverified"
        )

    def test_lvm_reproducibility(
        self, continuous_property_medium, krig_medium_grid, mean_data_medium, covariance_spherical
    ):
        """Test LVM kriging produces reproducible results"""
        np.random.seed(42)

        result1 = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.random.seed(42)
        result2 = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_lvm_result_validation(
        self, continuous_property_medium, krig_medium_grid, mean_data_medium, covariance_spherical
    ):
        """Test LVM kriging produces valid results"""
        result = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=mean_data_medium,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    # F-584: the local mean field is LVM's defining input and no test made it
    # influence the output.
    def test_lvm_mean_field_drives_estimate(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-584: the LVM mean field must reach the estimate.

        Two halves, both read assertion-relevant cells:

        (a) an all-uninformed property takes the documented mean-fill on every
            cell (cont_kriging no-neighbour branch), so the output must equal
            the mean field exactly — here 10.0 then 60.0;
        (b) on a sparse property, raising the mean field by +10 must move the
            KRIGED cells.

        A regression that drops the mean field from the C++ solve (LVM
        degenerating to SK-with-mean-zero) passes the copy-through, uniform-data
        and isinstance tests this file had, and fails both halves here.
        """
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z

        # (a) Empty property: the fill IS the mean field.
        empty_prop = ContProperty(np.zeros(size, dtype="float32"), np.zeros(size, dtype="uint8"))
        for level in (10.0, 60.0):
            mean_data = np.full(size, level, dtype="float32")
            result = lvm_kriging(
                prop=empty_prop,
                grid=krig_medium_grid,
                mean_data=mean_data,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=covariance_spherical,
            )
            np.testing.assert_allclose(
                result.data.astype("float64"),
                mean_data.astype("float64"),
                rtol=0,
                atol=1e-5,
                err_msg=(
                    f"LVM with no informed cell must fill every cell with the mean "
                    f"field ({level}), got {np.unique(result.data)[:5]}"
                ),
            )

        # (b) Sparse property: the mean gradient must move the kriged cells.
        result_low = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=np.full(size, 10.0, dtype="float32"),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        result_high = lvm_kriging(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            mean_data=np.full(size, 20.0, dtype="float32"),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        delta = _kriged_cells(result_high, continuous_property_medium.mask) - _kriged_cells(
            result_low, continuous_property_medium.mask
        )
        assert np.any(np.abs(delta) > 1e-3), (
            "LVM mean +10 left every kriged cell unchanged — the mean field does "
            "not reach the estimate"
        )

    def test_lvm_small_grid(self, continuous_property_small, krig_small_grid, covariance_spherical):
        """Test LVM kriging with small grid"""
        size = krig_small_grid.x * krig_small_grid.y * krig_small_grid.z
        mean_data = np.random.rand(size).astype("float32") * 50

        result = lvm_kriging(
            prop=continuous_property_small,
            grid=krig_small_grid,
            mean_data=mean_data,
            radiuses=(2, 2, 2),
            max_neighbours=4,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# Indicator Kriging Tests
# =============================================================================


@pytest.mark.hpgl
class TestIndicatorKriging:
    """Comprehensive tests for Indicator Kriging"""

    def test_ik_basic_execution(self, indicator_property_medium, krig_medium_grid):
        """Test basic IK execution"""
        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]

        for _i in range(3):
            ik_data.append(
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

        result = indicator_kriging(
            prop=indicator_property_medium,
            grid=krig_medium_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

    def test_ik_all_covariance_types(self, indicator_property_medium, krig_medium_grid):
        """Test IK with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            ik_data = []
            marginal_probs = [0.3, 0.4, 0.3]

            for _i in range(3):
                ik_data.append(
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

            result = indicator_kriging(
                prop=indicator_property_medium,
                grid=krig_medium_grid,
                data=ik_data,
                marginal_probs=marginal_probs,
            )
            assert isinstance(result, IndProperty), f"Failed for {name}"
            assert np.all(result.data < result.indicator_count), f"invalid values for {name}"

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12])
    def test_ik_various_neighbor_counts(
        self, indicator_property_medium, krig_medium_grid, max_neighbours
    ):
        """Test IK with various neighbor counts"""
        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]

        for _i in range(3):
            ik_data.append(
                {
                    "cov_model": CovarianceModel(
                        type=covariance.spherical,
                        ranges=(5.0, 5.0, 3.0),
                        angles=(0.0, 0.0, 0.0),
                        sill=1.0,
                        nugget=0.1,
                    ),
                    "radiuses": (5, 5, 3),
                    "max_neighbours": max_neighbours,
                }
            )

        result = indicator_kriging(
            prop=indicator_property_medium,
            grid=krig_medium_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )
        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)

    def test_ik_reproducibility(self, indicator_property_medium, krig_medium_grid):
        """Test IK produces reproducible results"""
        np.random.seed(42)

        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]

        for _i in range(3):
            ik_data.append(
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

        result1 = indicator_kriging(
            prop=indicator_property_medium,
            grid=krig_medium_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )

        np.random.seed(42)
        result2 = indicator_kriging(
            prop=indicator_property_medium,
            grid=krig_medium_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_ik_result_validation(self, indicator_property_medium, krig_medium_grid):
        """Test IK produces valid results (indicators in valid range)"""
        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]

        for _i in range(3):
            ik_data.append(
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

        result = indicator_kriging(
            prop=indicator_property_medium,
            grid=krig_medium_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )

        # Check indicators are within valid range
        assert np.all(result.data < result.indicator_count)

    # F-574 / F-583: the value-level IK oracle. The range check above is
    # satisfiable by an all-zero (or any in-range) output, so a constant-category
    # regression passed every IK test in this file.
    def test_ik_uniform_category_is_reproduced(self, krig_medium_grid):
        """F-574: a uniform indicator field must come back as that category.

        Every datum is category k (k = 0, 1, 2), so every kriged cell must be
        category k. A regression that returns a constant category — the exact
        failure the `np.isnan(uint8)` checks could never catch — fails for two
        of the three fixtures.
        """
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        mask = np.ones(size, dtype="uint8")
        ik_data = [
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
            for _ in range(3)
        ]

        for category in (0, 1, 2):
            result = indicator_kriging(
                prop=IndProperty(np.full(size, category, dtype="uint8"), mask.copy(), 3),
                grid=krig_medium_grid,
                data=ik_data,
                marginal_probs=[0.3, 0.4, 0.3],
            )
            categories = np.unique(result.data)
            assert np.all(result.data == category), (
                f"uniform indicator field of category {category}: expected every cell "
                f"in [{category}], got {categories.tolist()}"
            )

    # G-53: every IK test above builds N identical per-category dicts, so no
    # test could observe the per-category dispatch — the only oracle was
    # `data < indicator_count`.
    def test_ik_per_category_parameters_reach_the_solve(
        self, indicator_property_medium, krig_medium_grid
    ):
        """G-53: changing ONE category's parameters must move the output.

        The per-category dicts are identical except for the knob under test, so
        an implementation that shares one parameter set across categories — or
        ignores marginal_probs — returns the base array unchanged. Measured on
        this 500-cell fixture: category-2 ranges (5,5,3) -> (1,1,1) changes 3
        cells (this is what distinguishes a per-category dispatch from a
        category-0 profile broadcast to all), category-0 ranges changes 13
        cells, and marginal_probs (0.3,0.4,0.3) -> (0.8,0.1,0.1) changes 4.

        Catches: per-category covariance/range dispatch collapsed to one shared
        set or to category 0's profile (0 differing cells for the category-2
        assertion), and marginal_probs not reaching the per-category solve
        (0 differing cells for the marginal assertion).
        """

        def entry():
            return {
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

        def run(ik_data, marginal_probs):
            return indicator_kriging(
                prop=indicator_property_medium,
                grid=krig_medium_grid,
                data=ik_data,
                marginal_probs=marginal_probs,
            )

        base = run([entry() for _ in range(3)], [0.3, 0.4, 0.3])

        # Category 2, not category 0: a dispatch that broadcasts category 0's
        # profile to every category leaves this variant identical to the base.
        cat2_short_range = [entry() for _ in range(3)]
        cat2_short_range[2]["cov_model"] = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 1.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )
        cat2_changed = int(
            np.count_nonzero(run(cat2_short_range, [0.3, 0.4, 0.3]).data != base.data)
        )
        assert cat2_changed > 0, (
            "IK category-2 ranges (5,5,3) -> (1,1,1) left every cell on the same "
            "category — category 2's covariance does not reach its own solve "
            "(measured effect on this fixture: 3 of 500 cells). A dispatch that "
            "applies category 0's parameter set to every category produces this."
        )

        cat0_short_range = [entry() for _ in range(3)]
        cat0_short_range[0]["cov_model"] = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 1.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )
        ranges_changed = int(
            np.count_nonzero(run(cat0_short_range, [0.3, 0.4, 0.3]).data != base.data)
        )
        assert ranges_changed > 0, (
            "IK category-0 ranges (5,5,3) -> (1,1,1) left every cell on the same "
            "category — the per-category covariance does not reach the solve "
            "(measured effect on this fixture: 13 of 500 cells)"
        )

        marginal_changed = int(
            np.count_nonzero(run([entry() for _ in range(3)], [0.8, 0.1, 0.1]).data != base.data)
        )
        assert marginal_changed > 0, (
            "IK marginal_probs (0.3,0.4,0.3) -> (0.8,0.1,0.1) left every cell on the "
            "same category — the marginals do not reach the per-category solve "
            "(measured effect on this fixture: 4 of 500 cells)"
        )

    # F-575: the neighbour-count sweep above asserts only isinstance, so an IK
    # kernel that ignored max_neighbours was green.
    def test_ik_neighbor_count_affects_result(self, indicator_property_medium, krig_medium_grid):
        """F-575: max_neighbours 2 vs 12 must move the IK categories.

        Measured on this fixture: 12 of the 500 cells change. Catches a kernel
        that drops the neighbourhood cap from the per-category solve (0
        differing cells). The 0-difference result reported for the earlier
        residual was a property of that report's own fixture shapes, not of
        this fixture — see the report's Investigated-and-Rejected section.
        """

        def entry(max_neighbours):
            return {
                "cov_model": CovarianceModel(
                    type=covariance.spherical,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
                "radiuses": (5, 5, 3),
                "max_neighbours": max_neighbours,
            }

        def run(max_neighbours):
            return indicator_kriging(
                prop=indicator_property_medium,
                grid=krig_medium_grid,
                data=[entry(max_neighbours) for _ in range(3)],
                marginal_probs=[0.3, 0.4, 0.3],
            )

        changed = int(np.count_nonzero(run(2).data != run(12).data))
        assert changed > 0, (
            "IK max_neighbours=2 vs 12 left every cell on the same category — the "
            "neighbour cap does not reach the per-category solve "
            "(measured effect on this fixture: 12 of 500 cells)"
        )

    def test_ik_small_grid(self, indicator_property_small, krig_small_grid):
        """Test IK with small grid"""
        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]

        for _i in range(3):
            ik_data.append(
                {
                    "cov_model": CovarianceModel(
                        type=covariance.spherical,
                        ranges=(2.0, 2.0, 2.0),
                        angles=(0.0, 0.0, 0.0),
                        sill=1.0,
                        nugget=0.1,
                    ),
                    "radiuses": (2, 2, 2),
                    "max_neighbours": 4,
                }
            )

        result = indicator_kriging(
            prop=indicator_property_small,
            grid=krig_small_grid,
            data=ik_data,
            marginal_probs=marginal_probs,
        )
        assert isinstance(result, IndProperty)


# =============================================================================
# Median Indicator Kriging Tests
# =============================================================================


@pytest.mark.hpgl
class TestMedianIK:
    """Comprehensive tests for Median Indicator Kriging.

    median_ik requires exactly 2 indicator categories per its C API
    design (interleaved data layout). All tests use 2-category fixtures.
    """

    def test_median_ik_basic_execution(self, indicator_property_2cat_medium, krig_medium_grid):
        """Test basic median IK execution"""
        marginal_probs = (0.5, 0.5)

        result = median_ik(
            prop=indicator_property_2cat_medium,
            grid=krig_medium_grid,
            marginal_probs=marginal_probs,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        )

        assert isinstance(result, IndProperty)
        # F-574: IndProperty.data is uint8 (geo.py:504), so a NaN/Inf check on
        # it can never fail. Assert the satisfiable output contract instead.
        assert result.data.dtype == np.uint8
        assert np.all(result.data < result.indicator_count)

    def test_median_ik_all_covariance_types(self, indicator_property_2cat_medium, krig_medium_grid):
        """Test median IK with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        marginal_probs = (0.5, 0.5)

        for cov_type, name in cov_types:
            cov_model = CovarianceModel(
                type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
            )

            result = median_ik(
                prop=indicator_property_2cat_medium,
                grid=krig_medium_grid,
                marginal_probs=marginal_probs,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, IndProperty), f"Failed for {name}"
            # F-574: uint8 data can never be NaN — assert the output contract.
            assert result.data.dtype == np.uint8
            assert np.all(result.data < result.indicator_count), f"invalid values for {name}"

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_median_ik_various_neighbor_counts(
        self, indicator_property_2cat_medium, krig_medium_grid, max_neighbours
    ):
        """Test median IK with various neighbor counts"""
        result = median_ik(
            prop=indicator_property_2cat_medium,
            grid=krig_medium_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
            cov_model=CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        )
        assert isinstance(result, IndProperty)
        # F-574: uint8 data can never be NaN/Inf — assert the output contract.
        assert result.data.dtype == np.uint8
        assert np.all(result.data < result.indicator_count)
        assert result.data.shape == indicator_property_2cat_medium.data.shape

    def test_median_ik_reproducibility(self, indicator_property_2cat_medium, krig_medium_grid):
        """Test median IK produces reproducible results"""
        np.random.seed(42)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result1 = median_ik(
            prop=indicator_property_2cat_medium,
            grid=krig_medium_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
        )

        np.random.seed(42)
        result2 = median_ik(
            prop=indicator_property_2cat_medium,
            grid=krig_medium_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_median_ik_result_validation(self, indicator_property_2cat_medium, krig_medium_grid):
        """Test median IK produces valid results"""
        result = median_ik(
            prop=indicator_property_2cat_medium,
            grid=krig_medium_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        )

        # Check result shape
        assert result.data.shape == indicator_property_2cat_medium.data.shape
        # F-574: uint8 data can never be NaN/Inf — assert the output contract.
        assert result.data.dtype == np.uint8
        assert np.all(result.data < result.indicator_count)

    def test_median_ik_small_grid(self, indicator_property_2cat_small, krig_small_grid):
        """Test median IK with small grid"""
        result = median_ik(
            prop=indicator_property_2cat_small,
            grid=krig_small_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(2, 2, 2),
            max_neighbours=4,
            cov_model=CovarianceModel(
                type=covariance.spherical,
                ranges=(2.0, 2.0, 2.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        )
        assert isinstance(result, IndProperty)
        # F-574: uint8 data can never be NaN/Inf — assert the output contract.
        assert result.data.dtype == np.uint8
        assert np.all(result.data < result.indicator_count)

    # F-574 / F-583: value-level MIK oracle. `data < indicator_count` is
    # satisfied by an all-zero array, so a constant-category regression was
    # green across every MIK test above.
    def test_median_ik_uniform_category_is_reproduced(self, krig_medium_grid):
        """F-574: a uniform 2-category indicator field must come back as that category.

        Every datum is category 0 (then 1), so every cell must be categorised
        0 (then 1). A constant-category regression fails one of the two.
        """
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        mask = np.ones(size, dtype="uint8")

        for category in (0, 1):
            result = median_ik(
                prop=IndProperty(np.full(size, category, dtype="uint8"), mask.copy(), 2),
                grid=krig_medium_grid,
                marginal_probs=(0.5, 0.5),
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=CovarianceModel(
                    type=covariance.spherical,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            )
            categories = np.unique(result.data)
            assert np.all(result.data == category), (
                f"uniform indicator field of category {category}: expected every cell "
                f"in [{category}], got {categories.tolist()}"
            )

    # F-575 / F-585 residual: the MIK sweeps above assert isinstance and the
    # value range only, so a kernel that ignored the neighbourhood or the
    # covariance model was green. Measured on THIS fixture both levers move the
    # categories (9 and 5 cells of 500), so the residual is pinnable here.
    def test_median_ik_parameters_affect_result(
        self, indicator_property_2cat_medium, krig_medium_grid
    ):
        """F-575/F-585: max_neighbours and covariance type must move MIK categories.

        Measured on this fixture over 500 cells: max_neighbours 2 vs 12 changes
        9 cells; spherical vs exponential at 12 neighbours changes 5 cells.
        Catches: the neighbourhood cap or the covariance model dropped from the
        median-IK solve (0 differing cells).
        """

        def run(max_neighbours=12, cov_type=covariance.spherical):
            return median_ik(
                prop=indicator_property_2cat_medium,
                grid=krig_medium_grid,
                marginal_probs=(0.5, 0.5),
                radiuses=(5, 5, 3),
                max_neighbours=max_neighbours,
                cov_model=CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            )

        base = run()
        neighbors_changed = int(np.count_nonzero(run(max_neighbours=2).data != base.data))
        assert neighbors_changed > 0, (
            "median_ik max_neighbours=2 vs 12 left every cell on the same category — "
            "the neighbour cap does not reach the solve (measured: 9 of 500 cells)"
        )

        covariance_changed = int(
            np.count_nonzero(run(cov_type=covariance.exponential).data != base.data)
        )
        assert covariance_changed > 0, (
            "median_ik spherical vs exponential left every cell on the same category — "
            "the covariance model does not reach the solve (measured: 5 of 500 cells)"
        )


# =============================================================================
# Simple Cokriging Mark I Tests
# =============================================================================


@pytest.mark.hpgl
class TestSimpleCokrigingMarkI:
    """Comprehensive tests for Simple Cokriging Mark I"""

    def test_ck_markI_basic_execution(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
    ):
        """Test basic cokriging Mark I execution"""
        result = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == continuous_property_medium.data.shape

    def test_ck_markI_all_covariance_types(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """Test cokriging Mark I with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            cov_model = CovarianceModel(
                type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
            )

            result = simple_cokriging_markI(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                secondary_data=secondary_property_medium,
                primary_mean=50.0,
                secondary_mean=40.0,
                secondary_variance=100.0,
                correlation_coef=0.8,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
            )
            assert isinstance(result, ContProperty), f"Failed for {name}"

    # F-575 / F-585 (aggravated by G-50): the covariance type was swept here but
    # never asserted to have an effect, and the shared secondary fixture is
    # masked exactly where the primary fixture is, so every kriged node ran
    # without a secondary datum. This test builds a FULLY informed secondary.
    def test_ck_markI_covariance_type_affects_result(
        self, continuous_property_medium, krig_medium_grid
    ):
        """F-575: spherical vs exponential must change the KRIGED cells (Mark I)."""
        np.random.seed(43)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        secondary = ContProperty(
            np.random.rand(size).astype("float32") * 80, np.ones(size, dtype="uint8")
        )

        def run(cov_type):
            return simple_cokriging_markI(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
                secondary_data=secondary,
                primary_mean=50.0,
                secondary_mean=40.0,
                secondary_variance=100.0,
                correlation_coef=0.8,
            )

        kriged_sph = _kriged_cells(run(covariance.spherical), continuous_property_medium.mask)
        kriged_exp = _kriged_cells(run(covariance.exponential), continuous_property_medium.mask)
        assert not np.allclose(kriged_sph, kriged_exp, rtol=1e-4, atol=1e-4), (
            "Mark I covariance type spherical vs exponential left every kriged cell "
            "unchanged — the covariance model does not reach the cokriging solve"
        )

    @pytest.mark.parametrize("correlation_coef", [0.2, 0.5, 0.8, 0.95])
    def test_ck_markI_various_correlations(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
        correlation_coef,
    ):
        """Test cokriging Mark I with various correlation coefficients"""
        result = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=correlation_coef,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_ck_markI_various_neighbor_counts(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
        max_neighbours,
    ):
        """Test cokriging Mark I with various neighbor counts"""
        result = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)

    # F-575 / F-585: the Mark I neighbour-count sweep asserted only isinstance.
    def test_ck_markI_neighbor_count_affects_result(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """F-575: Mark I max_neighbours=4 vs 16 must move the KRIGED cells.

        The secondary fixture is fully informed (see
        test_ck_markI_covariance_type_affects_result), so the secondary equation
        is populated at every kriged node.
        """
        np.random.seed(43)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        secondary = ContProperty(
            np.random.rand(size).astype("float32") * 80, np.ones(size, dtype="uint8")
        )

        def run(max_neighbours):
            return simple_cokriging_markI(
                prop=continuous_property_medium,
                grid=krig_medium_grid,
                radiuses=(5, 5, 3),
                max_neighbours=max_neighbours,
                cov_model=covariance_spherical,
                secondary_data=secondary,
                primary_mean=50.0,
                secondary_mean=40.0,
                secondary_variance=100.0,
                correlation_coef=0.8,
            )

        assert not np.allclose(
            _kriged_cells(run(4), continuous_property_medium.mask),
            _kriged_cells(run(16), continuous_property_medium.mask),
            rtol=1e-4,
            atol=1e-4,
        ), (
            "Mark I max_neighbours=4 vs 16 left every kriged cell unchanged — "
            "parameter effect unverified"
        )

    # G-50: the collocated secondary datum must change the estimate. With the
    # shared fixture masked at the kriged nodes the secondary equation was never
    # entered, so every Mark I test above reduced to plain SK.
    def test_ck_markI_secondary_datum_changes_kriged_value(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
    ):
        """G-50: masking the secondary must move the kriged cells by ~19 units.

        The same secondary DATA is run twice and only its mask differs, so the
        single variable is whether a secondary datum exists at the node the
        kernel interpolates. Measured on this fixture at rho=0.8:
        max|Delta| = 19.198 and mean|Delta| = 7.225 over the 50 kriged cells
        (the s5 grid's 17.7-19.2 band). The 15-25 window pins that magnitude.

        Catches: the secondary dropped from the kriging system (Delta == 0 —
        the pre-fix fixture state) and an over-weighted secondary
        (Delta >> 25).
        """
        masked_secondary = ContProperty(
            secondary_property_medium.data.copy(),
            np.zeros_like(secondary_property_medium.mask),
        )

        def kriged(secondary):
            return _kriged_cells(
                simple_cokriging_markI(
                    prop=continuous_property_medium,
                    grid=krig_medium_grid,
                    secondary_data=secondary,
                    primary_mean=50.0,
                    secondary_mean=40.0,
                    secondary_variance=100.0,
                    correlation_coef=0.8,
                    radiuses=(5, 5, 3),
                    max_neighbours=12,
                    cov_model=covariance_spherical,
                ),
                continuous_property_medium.mask,
            )

        with_secondary = kriged(secondary_property_medium)
        without_secondary = kriged(masked_secondary)
        max_delta = float(np.max(np.abs(with_secondary - without_secondary)))
        assert 15.0 <= max_delta <= 25.0, (
            f"Mark I: removing the secondary datum changed the kriged cells by "
            f"{max_delta:.4f}; expected the measured ~19.2. A value near 0 means the "
            f"secondary never entered the kriging system (the G-50 fixture blindness)"
        )

    def test_ck_markI_reproducibility(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
    ):
        """Test cokriging Mark I produces reproducible results"""
        np.random.seed(42)

        result1 = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.random.seed(42)
        result2 = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_ck_markI_result_validation(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
    ):
        """Test cokriging Mark I produces valid results"""
        result = simple_cokriging_markI(
            prop=continuous_property_medium,
            grid=krig_medium_grid,
            secondary_data=secondary_property_medium,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

    def test_ck_markI_small_grid(
        self, continuous_property_small, krig_small_grid, covariance_spherical
    ):
        """Test cokriging Mark I with small grid"""
        size = krig_small_grid.x * krig_small_grid.y * krig_small_grid.z
        np.random.seed(43)
        sec_data = np.random.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        sec_mask[::5] = 0
        secondary_property = ContProperty(sec_data, sec_mask)

        result = simple_cokriging_markI(
            prop=continuous_property_small,
            grid=krig_small_grid,
            secondary_data=secondary_property,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(2, 2, 2),
            max_neighbours=4,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)


# =============================================================================
# Simple Cokriging Mark II Tests
# =============================================================================


@pytest.mark.hpgl
class TestSimpleCokrigingMarkII:
    """Comprehensive tests for Simple Cokriging Mark II"""

    def test_ck_markII_basic_execution(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """Test basic cokriging Mark II execution"""
        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        secondary_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        result = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == continuous_property_medium.data.shape

    def test_ck_markII_all_covariance_types(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """Test cokriging Mark II with all covariance types"""
        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            primary_data = {
                "data": continuous_property_medium,
                "mean": 50.0,
                "cov_model": CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            }

            secondary_data = {
                "data": secondary_property_medium,
                "mean": 40.0,
                "cov_model": CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            }

            result = simple_cokriging_markII(
                grid=krig_medium_grid,
                primary_data=primary_data,
                secondary_data=secondary_data,
                correlation_coef=0.8,
                radiuses=(5, 5, 3),
                max_neighbours=12,
            )
            assert isinstance(result, ContProperty), f"Failed for {name}"

    @staticmethod
    def _markII_result(primary, grid, secondary, cov_type, correlation_coef):
        """Run Mark II with one covariance type for both variables (F-585 helper)."""
        return simple_cokriging_markII(
            grid=grid,
            primary_data={
                "data": primary,
                "mean": 50.0,
                "cov_model": CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            },
            secondary_data={
                "data": secondary,
                "mean": 40.0,
                "cov_model": CovarianceModel(
                    type=cov_type,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
            },
            correlation_coef=correlation_coef,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )

    # F-585 (aggravated by G-50): the Mark II correlation sweep below asserted
    # only isinstance/isfinite, and test_regression_v150.py:288-295 states the
    # Mark II cross-covariance ratio path is NOT exercised there either. The
    # secondary fixture here is fully informed, so a secondary datum exists at
    # every primary-uninformed (kriged) node.
    def test_ck_markII_correlation_affects_result(
        self, continuous_property_medium, krig_medium_grid
    ):
        """F-585: Mark II correlation_coef 0.8 vs 0.2 must move the KRIGED cells."""
        np.random.seed(43)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        secondary = ContProperty(
            np.random.rand(size).astype("float32") * 80, np.ones(size, dtype="uint8")
        )

        kriged_high = _kriged_cells(
            self._markII_result(
                continuous_property_medium, krig_medium_grid, secondary, covariance.spherical, 0.8
            ),
            continuous_property_medium.mask,
        )
        kriged_low = _kriged_cells(
            self._markII_result(
                continuous_property_medium, krig_medium_grid, secondary, covariance.spherical, 0.2
            ),
            continuous_property_medium.mask,
        )
        assert not np.allclose(kriged_high, kriged_low, rtol=1e-4, atol=1e-4), (
            "Mark II correlation_coef 0.8 vs 0.2 left every kriged cell unchanged — "
            "the correlation coefficient does not reach the cokriging solve"
        )

    def test_ck_markII_covariance_type_affects_result(
        self, continuous_property_medium, krig_medium_grid
    ):
        """F-585: Mark II spherical vs exponential must move the KRIGED cells."""
        np.random.seed(43)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        secondary = ContProperty(
            np.random.rand(size).astype("float32") * 80, np.ones(size, dtype="uint8")
        )

        kriged_sph = _kriged_cells(
            self._markII_result(
                continuous_property_medium, krig_medium_grid, secondary, covariance.spherical, 0.8
            ),
            continuous_property_medium.mask,
        )
        kriged_exp = _kriged_cells(
            self._markII_result(
                continuous_property_medium,
                krig_medium_grid,
                secondary,
                covariance.exponential,
                0.8,
            ),
            continuous_property_medium.mask,
        )
        assert not np.allclose(kriged_sph, kriged_exp, rtol=1e-4, atol=1e-4), (
            "Mark II covariance type spherical vs exponential left every kriged cell "
            "unchanged — the covariance model does not reach the cokriging solve"
        )

    # G-50: Mark II had no value coverage at all, and the secondary is only read
    # at the kriged nodes (same process_node_loop as Mark I).
    def test_ck_markII_secondary_datum_changes_kriged_value(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """G-50: masking the secondary must move the Mark II kriged cells by ~33 units.

        Same secondary DATA, only the mask differs — so the single variable is
        whether a secondary datum exists at the interpolated node. Measured on
        this fixture at rho=0.8: max|Delta| = 32.910, mean|Delta| = 14.576 over
        the 50 kriged cells; the 25-45 window pins that magnitude.

        Catches: the secondary dropped from the Markov II system (Delta == 0)
        and an over-weighted cross-covariance (Delta >> 45).
        """
        masked_secondary = ContProperty(
            secondary_property_medium.data.copy(),
            np.zeros_like(secondary_property_medium.mask),
        )

        with_secondary = _kriged_cells(
            self._markII_result(
                continuous_property_medium,
                krig_medium_grid,
                secondary_property_medium,
                covariance.spherical,
                0.8,
            ),
            continuous_property_medium.mask,
        )
        without_secondary = _kriged_cells(
            self._markII_result(
                continuous_property_medium,
                krig_medium_grid,
                masked_secondary,
                covariance.spherical,
                0.8,
            ),
            continuous_property_medium.mask,
        )
        max_delta = float(np.max(np.abs(with_secondary - without_secondary)))
        assert 25.0 <= max_delta <= 45.0, (
            f"Mark II: removing the secondary datum changed the kriged cells by "
            f"{max_delta:.4f}; expected the measured ~32.9. A value near 0 means the "
            f"secondary never entered the Markov II system (the G-50 fixture blindness)"
        )

    @pytest.mark.parametrize("correlation_coef", [0.2, 0.5, 0.8, 0.95])
    def test_ck_markII_various_correlations(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        correlation_coef,
    ):
        """Test cokriging Mark II with various correlation coefficients"""
        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        secondary_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        result = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=correlation_coef,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
        assert np.count_nonzero(result.mask) > 0, "Mark II wrote no output cells"
        assert np.all(np.isfinite(result.data[result.mask != 0]))

    # G-54: this Mark II sweep selected by `result.mask != 0`, which is EMPTY
    # after a no-op kernel — isfinite() over an empty selection is vacuously
    # True — so the finiteness check alone could not see a write failure.
    @pytest.mark.parametrize("max_neighbours", [4, 8, 12, 16])
    def test_ck_markII_various_neighbor_counts(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        max_neighbours,
    ):
        """Test cokriging Mark II with various neighbor counts"""
        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        secondary_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        result = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=max_neighbours,
        )
        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
        assert np.count_nonzero(result.mask) > 0, "Mark II wrote no output cells"
        assert np.all(np.isfinite(result.data[result.mask != 0]))

    def test_ck_markII_reproducibility(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """Test cokriging Mark II produces reproducible results"""
        np.random.seed(42)

        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        secondary_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        result1 = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )

        np.random.seed(42)
        result2 = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )

        np.testing.assert_array_almost_equal(result1.data, result2.data, decimal=5)

    def test_ck_markII_result_validation(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """Test cokriging Mark II produces valid results"""
        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        secondary_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            ),
        }

        result = simple_cokriging_markII(
            grid=krig_medium_grid,
            primary_data=primary_data,
            secondary_data=secondary_data,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
        )

        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# Simple Kriging Weights Tests
# =============================================================================


@pytest.mark.hpgl
class TestSimpleKrigingWeights:
    """Comprehensive tests for Simple Kriging Weights calculation"""

    def test_weights_basic_execution(self, neighbor_points):
        """Test basic weights calculation"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        weights = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )

        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(n_x)
        assert weights.dtype == np.float32

    def test_weights_all_covariance_types(self, neighbor_points):
        """Test weights with all covariance types"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        cov_types = [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]

        for cov_type, name in cov_types:
            weights = simple_kriging_weights(
                center_point=center_point,
                n_x=n_x,
                n_y=n_y,
                n_z=n_z,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                cov_type=cov_type,
                nugget=0.1,
            )
            assert isinstance(weights, np.ndarray), f"Failed for {name}"

    def test_weights_various_nugget(self, neighbor_points):
        """Test weights with various nugget values"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        # Test with nugget values that are less than sill
        # Note: nugget=1.0 with sill=1.0 causes C++ exception in HPGL
        # This is a known limitation - skip that case
        for nugget in [0.0, 0.1, 0.5]:
            weights = simple_kriging_weights(
                center_point=center_point,
                n_x=n_x,
                n_y=n_y,
                n_z=n_z,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                cov_type=covariance.exponential,
                nugget=nugget,
            )
            assert isinstance(weights, np.ndarray)

    def test_weights_various_ranges(self, neighbor_points):
        """Test weights with various range values"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        ranges = [(3.0, 3.0, 2.0), (5.0, 5.0, 3.0), (10.0, 10.0, 5.0)]

        for ranges_val in ranges:
            weights = simple_kriging_weights(
                center_point=center_point,
                n_x=n_x,
                n_y=n_y,
                n_z=n_z,
                ranges=ranges_val,
                sill=1.0,
                cov_type=covariance.exponential,
                nugget=0.1,
            )
            assert isinstance(weights, np.ndarray)

    def test_weights_default_parameters(self, neighbor_points):
        """Test weights with default parameters"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        # Use defaults for angles and nugget
        weights = simple_kriging_weights(center_point=center_point, n_x=n_x, n_y=n_y, n_z=n_z)

        assert isinstance(weights, np.ndarray)

    def test_weights_custom_angles(self, neighbor_points):
        """Test weights with custom angles (rotation path coverage).

        This is the ONLY angle-path test for simple_kriging_weights in the
        suite — a rotation-matrix regression must be detected, not just
        "doesn't crash". Asserts finite weights and that a non-trivial
        rotation changes the weights vs the zero-rotation baseline.
        """
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        angles = [(0.0, 0.0, 0.0), (30.0, 45.0, 60.0), (90.0, 0.0, 0.0)]

        all_weights = {}
        for angles_val in angles:
            weights = simple_kriging_weights(
                center_point=center_point,
                n_x=n_x,
                n_y=n_y,
                n_z=n_z,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                cov_type=covariance.exponential,
                nugget=0.1,
                angles=angles_val,
            )
            assert isinstance(weights, np.ndarray)
            assert len(weights) == len(n_x)
            assert np.all(np.isfinite(weights)), (
                f"Non-finite weights at angles={angles_val}"
            )
            all_weights[angles_val] = weights

        # A non-trivial rotation must actually change the weights (the
        # rotation convention is a documented hazard — L-34). If the rotation
        # path is ignored, (30,45,60) equals (0,0,0) and the test fails.
        assert not np.allclose(
            all_weights[(0.0, 0.0, 0.0)],
            all_weights[(30.0, 45.0, 60.0)],
            rtol=1e-4,
            atol=1e-5,
        ), "Rotation angles (30,45,60) should produce weights different from (0,0,0)"

    def test_weights_result_validation(self, neighbor_points):
        """Test weights are valid (no NaN, Inf, correct count, finite)"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        weights = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )

        # Check for NaN and Inf
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        # SK weights must be finite and match neighbor count
        assert len(weights) == len(n_x)
        assert np.all(np.isfinite(weights))

    def test_weights_reproducibility(self, neighbor_points):
        """Test weights calculation is reproducible"""
        n_x, n_y, n_z = neighbor_points
        center_point = (5.0, 5.0, 2.5)

        np.random.seed(42)
        weights1 = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )

        np.random.seed(42)
        weights2 = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )

        np.testing.assert_array_almost_equal(weights1, weights2, decimal=5)

    def test_weights_various_neighbor_counts(self):
        """Test weights with various numbers of neighbors"""
        center_point = (5.0, 5.0, 2.5)

        for n in [4, 8, 12, 16]:
            np.random.seed(42)
            n_x = np.random.rand(n).astype("float32") * 10
            n_y = np.random.rand(n).astype("float32") * 10
            n_z = np.random.rand(n).astype("float32") * 5

            weights = simple_kriging_weights(
                center_point=center_point,
                n_x=n_x,
                n_y=n_y,
                n_z=n_z,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                cov_type=covariance.exponential,
                nugget=0.1,
            )
            assert len(weights) == n

    def test_weights_single_neighbor(self):
        """Test weights with single neighbor (edge case)"""
        center_point = (5.0, 5.0, 2.5)
        n_x = np.array([5.5], dtype="float32")
        n_y = np.array([5.5], dtype="float32")
        n_z = np.array([2.5], dtype="float32")

        weights = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.exponential,
            nugget=0.1,
        )
        assert len(weights) == 1

    def test_weights_single_neighbor_zero_nugget(self):
        """With a single neighbor at the same location and nugget=0, SK weight is 1.0."""
        center_point = (5.0, 5.0, 2.5)
        # Neighbor at exactly the same 3D location as center
        n_x = np.array([5.0], dtype="float32")
        n_y = np.array([5.0], dtype="float32")
        n_z = np.array([2.5], dtype="float32")

        weights = simple_kriging_weights(
            center_point=center_point,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        assert len(weights) == 1
        # With zero distance and zero nugget, the covariance is sill,
        # so the SK weight C(0)/C(0) = 1.0
        assert abs(weights[0] - 1.0) < 1e-5, (
            f"SK weight for co-located neighbor with nugget=0 should be 1.0, got {weights[0]}"
        )

    # Error-path tests: exercise all validation branches in simple_kriging_weights

    def test_weights_mismatched_array_sizes(self):
        """Test weights raises RuntimeError for mismatched n_x, n_y, n_z lengths"""
        center_point = (5.0, 5.0, 2.5)
        n_x = np.array([1.0, 2.0, 3.0], dtype="float32")
        n_y = np.array([1.0, 2.0], dtype="float32")  # shorter
        n_z = np.array([1.0, 2.0, 3.0], dtype="float32")

        with pytest.raises(RuntimeError, match="Invalid pointset"):
            simple_kriging_weights(center_point=center_point, n_x=n_x, n_y=n_y, n_z=n_z)

    def test_weights_zero_data_points(self):
        """Test weights raises RuntimeError for empty neighbor arrays"""
        center_point = (5.0, 5.0, 2.5)
        empty = np.array([], dtype="float32")

        with pytest.raises(RuntimeError, match="at least one data point is required"):
            simple_kriging_weights(center_point=center_point, n_x=empty, n_y=empty, n_z=empty)

    def test_weights_nan_inf_in_neighbors(self):
        """Test weights raises ValueError when neighbor arrays contain NaN or Inf"""
        center_point = (5.0, 5.0, 2.5)
        n_x = np.array([1.0, 2.0, 3.0], dtype="float32")
        n_y = np.array([1.0, np.nan, 3.0], dtype="float32")  # contains NaN
        n_z = np.array([1.0, 2.0, 3.0], dtype="float32")

        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            simple_kriging_weights(center_point=center_point, n_x=n_x, n_y=n_y, n_z=n_z)

    def test_weights_nan_inf_in_center_point(self):
        """Test weights raises ValueError when center_point contains NaN or Inf"""
        center_point = (5.0, np.nan, 2.5)  # contains NaN
        n_x = np.array([1.0, 2.0, 3.0], dtype="float32")
        n_y = np.array([1.0, 2.0, 3.0], dtype="float32")
        n_z = np.array([1.0, 2.0, 3.0], dtype="float32")

        with pytest.raises(ValueError, match="center_point contains NaN or infinite values"):
            simple_kriging_weights(center_point=center_point, n_x=n_x, n_y=n_y, n_z=n_z)


# =============================================================================
# Tuplet Input Tests for Kriging Functions (M22)
# =============================================================================


@pytest.mark.hpgl
class TestKrigingTupleInputs:
    """Test that kriging functions decorated with @accepts_tuple accept tuple input.

    Each of the 6 kriging functions with @accepts_tuple("prop", 0) should
    accept prop as a tuple (data, mask) or (data, mask, n_indicators),
    mirroring the pattern in test_simulation_complete.py.
    """

    def test_ok_accepts_tuple_prop(self, krig_medium_grid, covariance_spherical):
        """ordinary_kriging accepts prop as (data, mask) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")

        result = ordinary_kriging(
            prop=(data, mask),
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert result.data.shape == data.shape

    def test_sk_accepts_tuple_prop(self, krig_medium_grid, covariance_spherical):
        """simple_kriging accepts prop as (data, mask) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")

        result = simple_kriging(
            prop=(data, mask),
            grid=krig_medium_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert result.data.shape == data.shape

    def test_lvm_accepts_tuple_prop(self, krig_medium_grid, covariance_spherical):
        """lvm_kriging accepts prop as (data, mask) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        mean_data = np.random.rand(size).astype("float32") * 50

        result = lvm_kriging(
            prop=(data, mask),
            grid=krig_medium_grid,
            mean_data=mean_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert result.data.shape == data.shape

    def test_mik_accepts_tuple_prop(self, krig_medium_grid):
        """median_ik accepts prop as (data, mask, indicator_count) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = median_ik(
            prop=(data, mask, 2),
            grid=krig_medium_grid,
            marginal_probs=(0.5, 0.5),
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
        )
        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)

    def test_ik_accepts_tuple_prop(self, krig_medium_grid):
        """indicator_kriging accepts prop as (data, mask, indicator_count) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.randint(0, 3, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")

        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]
        for _ in range(3):
            ik_data.append(
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

        result = indicator_kriging(
            prop=(data, mask, 3), grid=krig_medium_grid, data=ik_data, marginal_probs=marginal_probs
        )
        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)

    def test_ck_markI_accepts_tuple_prop(self, krig_medium_grid, covariance_spherical):
        """simple_cokriging_markI accepts prop as (data, mask) tuple."""
        np.random.seed(42)
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")

        np.random.seed(43)
        sec_data = np.random.rand(size).astype("float32") * 80
        sec_mask = np.ones(size, dtype="uint8")
        secondary = ContProperty(sec_data, sec_mask)

        result = simple_cokriging_markI(
            prop=(data, mask),
            grid=krig_medium_grid,
            secondary_data=secondary,
            primary_mean=50.0,
            secondary_mean=40.0,
            secondary_variance=100.0,
            correlation_coef=0.8,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=covariance_spherical,
        )
        assert isinstance(result, ContProperty)
        assert result.data.shape == data.shape


# =============================================================================
# Negative Tests for Kriging Functions (H2)
# =============================================================================


@pytest.mark.hpgl
class TestKrigingNegativeCases:
    """Negative/error-path tests for all 6 kriging functions.

    Tests exercise validation branches for invalid input: empty properties,
    mismatched dimensions, invalid parameters, and wrong input types.
    """

    def test_sk_max_neighbours_zero(self, krig_medium_grid, covariance_spherical):
        """simple_kriging raises CriticalValidationError when max_neighbours=0"""
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError):
            simple_kriging(prop, krig_medium_grid, (5, 5, 3), 0, covariance_spherical)

    def test_sk_none_prop_raises(self):
        """simple_kriging raises RuntimeError when prop is None."""
        with pytest.raises(RuntimeError, match="expected ContProperty"):
            simple_kriging(
                None,
                SugarboxGrid(10, 10, 5),
                (5, 5, 3),
                12,
                CovarianceModel(covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1),
            )

    def test_lvm_mismatched_mean_data(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """lvm_kriging raises ValueError when mean_data size doesn't match grid"""
        # Create mean_data with wrong size
        bad_mean = np.random.rand(100).astype("float32") * 50  # grid is 500 cells
        with pytest.raises(ValueError, match="mean_data size"):
            lvm_kriging(
                continuous_property_medium,
                krig_medium_grid,
                bad_mean,
                (5, 5, 3),
                12,
                covariance_spherical,
            )

    def test_lvm_non_array_mean(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """lvm_kriging raises ValueError when mean_data is not a numpy array"""
        with pytest.raises(ValueError, match="mean_data must be a numpy array"):
            lvm_kriging(
                continuous_property_medium,
                krig_medium_grid,
                "not_an_array",
                (5, 5, 3),
                12,
                covariance_spherical,
            )

    def test_lvm_max_neighbours_zero(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """lvm_kriging raises CriticalValidationError when max_neighbours=0"""
        size = krig_medium_grid.x * krig_medium_grid.y * krig_medium_grid.z
        mean_data = np.random.rand(size).astype("float32") * 50
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError):
            lvm_kriging(
                continuous_property_medium,
                krig_medium_grid,
                mean_data,
                (5, 5, 3),
                0,
                covariance_spherical,
            )

    def test_ik_mismatched_marginal_probs(self, indicator_property_medium, krig_medium_grid):
        """indicator_kriging raises ValueError when marginal_probs doesn't match data"""
        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (5, 5, 3),
                "max_neighbours": 12,
            }
            for _ in range(3)
        ]
        # Wrong length: 2 probs for 3 indicators
        with pytest.raises(ValueError, match="marginal_probs length"):
            indicator_kriging(indicator_property_medium, krig_medium_grid, ik_data, [0.3, 0.7])

    def test_ik_empty_data_list(self, indicator_property_medium, krig_medium_grid):
        """indicator_kriging raises CriticalValidationError when data list is empty."""
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError, match="Indicator count must be positive"):
            indicator_kriging(indicator_property_medium, krig_medium_grid, [], [0.5])

    def test_mik_wrong_marginal_probs_count(self, indicator_property_medium, krig_medium_grid):
        """median_ik raises ValueError when marginal_probs doesn't have 2 elements"""
        cov_model = CovarianceModel(covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1)
        with pytest.raises(ValueError, match="2 elements"):
            median_ik(
                indicator_property_medium,
                krig_medium_grid,
                (0.3, 0.4, 0.3),
                (5, 5, 3),
                12,
                cov_model,
            )

    def test_mik_max_neighbours_zero(self, indicator_property_medium, krig_medium_grid):
        """median_ik raises CriticalValidationError when max_neighbours=0"""
        cov_model = CovarianceModel(covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1)
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError):
            median_ik(
                indicator_property_medium, krig_medium_grid, (0.5, 0.5), (5, 5, 3), 0, cov_model
            )

    def test_ck_markI_mismatched_secondary(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical
    ):
        """simple_cokriging_markI raises ValueError when secondary_data size mismatches grid"""
        # Create secondary_data with wrong size
        bad_sec_data = np.random.rand(100).astype("float32") * 80
        bad_sec_mask = np.ones(100, dtype="uint8")
        bad_secondary = ContProperty(bad_sec_data, bad_sec_mask)
        with pytest.raises(ValueError, match="secondary_data size"):
            simple_cokriging_markI(
                continuous_property_medium,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
                bad_secondary,
                50.0,
                40.0,
                100.0,
                0.8,
            )

    def test_ck_markI_invalid_correlation(
        self,
        continuous_property_medium,
        krig_medium_grid,
        secondary_property_medium,
        covariance_spherical,
    ):
        """simple_cokriging_markI raises CriticalValidationError for invalid correlation"""
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError):
            simple_cokriging_markI(
                continuous_property_medium,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
                secondary_property_medium,
                50.0,
                40.0,
                100.0,
                1.5,
            )

    def test_ck_markII_non_dict_primary(self, krig_medium_grid, secondary_property_medium):
        """simple_cokriging_markII raises CriticalValidationError for non-dict primary_data"""
        from geo_bsd.validation import CriticalValidationError

        sec_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1
            ),
        }
        with pytest.raises(CriticalValidationError, match="primary_data must be a dict"):
            simple_cokriging_markII(krig_medium_grid, "not_a_dict", sec_data, 0.8, (5, 5, 3), 12)

    def test_ck_markII_missing_key(
        self, continuous_property_medium, krig_medium_grid, secondary_property_medium
    ):
        """simple_cokriging_markII raises CriticalValidationError when primary_data missing key"""
        from geo_bsd.validation import CriticalValidationError

        primary_data = {
            "data": continuous_property_medium,
            "mean": 50.0,
            # 'cov_model' missing
        }
        sec_data = {
            "data": secondary_property_medium,
            "mean": 40.0,
            "cov_model": CovarianceModel(
                covariance.spherical, (5.0, 5.0, 3.0), (0, 0, 0), 1.0, 0.1
            ),
        }
        with pytest.raises(CriticalValidationError, match="missing required key"):
            simple_cokriging_markII(krig_medium_grid, primary_data, sec_data, 0.8, (5, 5, 3), 12)

    # ---- F-204: simple_cokriging_markI uncovered validation guards ----

    def test_ck_markI_wrong_prop_type(
        self, indicator_property_medium, krig_medium_grid, secondary_property_medium,
        covariance_spherical,
    ):
        """F-204: simple_cokriging_markI raises TypeError when prop is IndProperty."""
        with pytest.raises(TypeError, match="prop must be ContProperty"):
            simple_cokriging_markI(
                indicator_property_medium,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
                secondary_property_medium,
                50.0,
                40.0,
                100.0,
                0.8,
            )

    def test_ck_markI_empty_secondary(
        self, continuous_property_medium, krig_medium_grid, covariance_spherical,
    ):
        """F-204: simple_cokriging_markI raises ValueError when secondary_data is empty."""
        empty_data = np.array([], dtype="float32")
        empty_mask = np.array([], dtype="uint8")
        empty_secondary = ContProperty(empty_data, empty_mask)
        with pytest.raises(ValueError, match="secondary_data.data is empty"):
            simple_cokriging_markI(
                continuous_property_medium,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
                empty_secondary,
                50.0,
                40.0,
                100.0,
                0.8,
            )

    # ---- F-205: simple_kriging uncovered validation guards ----

    def test_sk_wrong_prop_type(self, indicator_property_medium, krig_medium_grid,
                                  covariance_spherical):
        """F-205: simple_kriging raises TypeError when prop is IndProperty."""
        with pytest.raises(TypeError, match="prop must be ContProperty"):
            simple_kriging(
                indicator_property_medium,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
            )

    def test_sk_empty_data(self, krig_medium_grid, covariance_spherical):
        """F-205: simple_kriging raises ValueError when prop.data is empty."""
        empty_data = np.array([], dtype="float32")
        empty_mask = np.array([], dtype="uint8")
        empty_prop = ContProperty(empty_data, empty_mask)
        with pytest.raises(ValueError, match="prop.data is empty"):
            simple_kriging(
                empty_prop,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
            )

    def test_sk_mismatched_data_size(self, krig_medium_grid, covariance_spherical):
        """F-205: simple_kriging raises ValueError when prop.data size mismatches grid."""
        wrong_data = np.random.rand(100).astype("float32") * 100  # grid is 500 cells
        wrong_mask = np.ones(100, dtype="uint8")
        wrong_prop = ContProperty(wrong_data, wrong_mask)
        with pytest.raises(ValueError, match="does not match grid size"):
            simple_kriging(
                wrong_prop,
                krig_medium_grid,
                (5, 5, 3),
                12,
                covariance_spherical,
            )

    # ---- F-206: simple_kriging_weights error return path ----

    def test_skw_degenerate_points_all_identical(self):
        """F-206: simple_kriging_weights raises RuntimeError with degenerate (all identical) points.

        All-identical neighbour points produce a singular covariance matrix,
        causing the C++ function to return rc != 0 which triggers RuntimeError.
        """
        center = (0.0, 0.0, 0.0)
        n_x = [0.0, 0.0, 0.0, 0.0, 0.0]
        n_y = [0.0, 0.0, 0.0, 0.0, 0.0]
        n_z = [0.0, 0.0, 0.0, 0.0, 0.0]
        with pytest.raises(RuntimeError, match="simple_kriging_weights failed"):
            simple_kriging_weights(
                center,
                n_x, n_y, n_z,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                cov_type=covariance.spherical,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
