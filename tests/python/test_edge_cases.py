"""
Edge cases and error handling tests for HPGL algorithms.

This module tests boundary conditions, extreme inputs, and error scenarios
across all HPGL functions including:
- Grid edge cases (empty, single cell, large, non-cubic)
- Data edge cases (sparse, dense, uniform, extreme values, NaN)
- Parameter validation
- Property edge cases
- Simulation edge cases (determinism, masks)
- CDF edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import CdfData, calc_cdf
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        calc_mean,
        covariance,
        ordinary_kriging,
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.validation import CriticalValidationError
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


# =============================================================================
# 1. GRID EDGE CASES
# =============================================================================


@pytest.mark.hpgl
class TestGridEdgeCases:
    """Test edge cases related to grid configurations"""

    def test_non_cubic_grid_flat_x(self):
        """Test with flat grid along X axis (1 x 10 x 10)

        Uses radius=1 for the single-cell X dimension since HPGL
        does not support radius=0 (causes access violation in C++).
        """
        grid = SugarboxGrid(x=1, y=10, z=10)
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 5.0, 5.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(1, 5, 5), max_neighbours=12, cov_model=cov_model
        )

        assert result.data.shape == (1, 10, 10)
        # H-10: original informed cells must stay informed on non-cubic grids
        # (a storage-order/index-mapping regression would produce wrong masks).
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    def test_non_cubic_grid_flat_z(self):
        """Test with flat grid along Z axis (10 x 10 x 1)

        Uses radius=1 and range=1.0 for the single-cell Z dimension since
        HPGL does not support zero radius/range values (causes C++ exception).
        """
        grid = SugarboxGrid(x=10, y=10, z=1)
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        mask[::5] = 0
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 1), max_neighbours=12, cov_model=cov_model
        )

        assert result.data.shape == (10, 10, 1)
        # H-10: original informed cells must stay informed on non-cubic grids.
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    def test_non_cubic_grid_different_dimensions(self):
        """Test with grid having all different dimensions (5 x 10 x 20)"""
        grid = SugarboxGrid(x=5, y=10, z=20)
        data = np.random.rand(1000).astype("float32") * 100
        mask = np.ones(1000, dtype="uint8")
        # Make some uninformed
        mask[::5] = 0
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # IMPORTANT: Reshape 1D data to match grid
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(3.0, 5.0, 10.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(2, 5, 10), max_neighbours=12, cov_model=cov_model
        )

        assert result.data.shape == (5, 10, 20)
        # H-10: original informed cells must stay informed on non-cubic grids.
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    @pytest.mark.slow  # T-30: 125,000 cells (50³) — >100K-cell Phase-4 threshold, machine-freeze risk
    def test_large_grid_stress(self):
        """Test with large grid (50 x 50 x 50 = 125,000 cells)"""
        grid = SugarboxGrid(x=50, y=50, z=50)
        data = np.random.rand(125000).astype("float32") * 100
        mask = np.ones(125000, dtype="uint8")
        mask[::10] = 0  # 10% uninformed
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 10.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 5), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)

        assert result.data.shape == (50, 50, 50)
        assert not np.all(result.data == 0)
        # H-06: informed cells must stay informed on the large grid (guards a
        # silent early-return/partial-execution regression on big workloads).
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    @pytest.mark.parametrize(
        "x,y,z",
        [
            (2, 3, 5),  # Small but not trivial
            (3, 2, 2),  # X dominant
            (2, 5, 3),  # Y dominant
        ],
    )
    def test_small_non_cubic_grids(self, x, y, z):
        """Test small non-cubic grids with various dimensions"""
        grid = SugarboxGrid(x=x, y=y, z=z)
        size = x * y * z
        data = np.arange(size, dtype="float32")
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # IMPORTANT: Reshape 1D data to match grid
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 1.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(1, 1, 1),
            max_neighbours=min(8, size),
            cov_model=cov_model,
        )

        assert result.data.shape == (x, y, z)
        # H-12: original informed cells must stay informed (data is arange,
        # so a wrong-value storage-order regression would also be detectable
        # here — mask preservation is the discriminator).
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)


# =============================================================================
# 2. DATA EDGE CASES
# =============================================================================


@pytest.mark.hpgl
class TestDataEdgeCases:
    """Test edge cases related to data values and sparsity"""

    def test_sparse_data_90_percent_uninformed(self):
        """Test with 90% of data uninformed (sparse data scenario)

        Note: HPGL uses undefined_on_failure, meaning cells without neighbors
        will remain uninformed (mask=0) and their data values may be unchanged.
        With 90% sparsity, many cells may not find neighbors within the search radius.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.zeros(500, dtype="uint8")
        # Only 10% informed
        mask[::10] = 1
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 5.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(10, 10, 5), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)
        # Original informed cells should remain informed
        # Note: result.mask is now 3D, so we need to flatten it for comparison
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)
        # Some uninformed cells may become informed if they found neighbors
        # (but with high sparsity, many may remain uninformed)

    def test_sparse_data_99_percent_uninformed(self):
        """Test with 99% of data uninformed (nearly empty)

        Note: With only 1% informed data, most cells will not find neighbors.
        This is an extreme sparse case that tests HPGL's graceful degradation.
        """
        grid = SugarboxGrid(x=10, y=10, z=10)
        data = np.random.rand(1000).astype("float32") * 100
        mask = np.zeros(1000, dtype="uint8")
        # Only 1% informed (10 values out of 1000)
        mask[::100] = 1
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 10.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 10),
            max_neighbours=8,  # Use fewer neighbors since data is sparse
            cov_model=cov_model,
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Should still complete
        assert result.data.shape == (10, 10, 10)
        # Original informed cells should remain informed
        # Note: result.mask is now 3D, so we need to flatten it for comparison
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    def test_dense_data_100_percent_informed(self):
        """Test with 100% of data informed (dense data scenario)

        Note: When all cells are informed, kriging still performs estimation
        using neighbors. Values will be smoothed but similar to input range.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")  # All informed
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)
        # All values should remain informed
        assert np.all(result.mask == 1)
        # Kriging performs estimation - values will be smoothed
        # Just check that the result is in a reasonable range (similar to input)
        assert np.all(result.data >= 0)  # Non-negative values
        assert np.all(result.data < 150)  # Reasonable upper bound (slightly above input max)

    def test_uniform_data_all_same_value(self):
        """Test with all data having the same value (uniform distribution)"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.ones(500, dtype="float32") * 42.0  # All same value
        mask = np.ones(500, dtype="uint8")
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
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Results should be close to the uniform value
        assert np.allclose(result.data, 42.0, atol=0.1)

    def test_extreme_values_very_large(self):
        """Test with very large positive values"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.full(500, 1e10, dtype="float32")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
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
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Should handle large values without overflow
        assert not np.any(np.isinf(result.data))
        assert not np.any(np.isnan(result.data))

    def test_extreme_values_very_small(self):
        """Test with very small positive values"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.full(500, 1e-10, dtype="float32")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
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
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Should handle small values without underflow
        assert not np.any(np.isinf(result.data))
        assert not np.any(np.isnan(result.data))

    def test_negative_values(self):
        """Test with negative data values

        Note: HPGL should handle negative values without issues.
        The kriging algorithm works with any floating point values.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100 - 50  # Range: -50 to 50
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Should handle negative values
        assert result.data.shape == (10, 10, 5)
        # Result should contain negative values (not NaN or Inf)
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
        # Some values should be negative
        assert np.any(result.data < 0)

    def test_all_zeros(self):
        """Test with all zero values"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.zeros(500, dtype="float32")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
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

        # Results should be close to zero
        assert np.allclose(result.data, 0.0, atol=0.01)

    def test_nan_values_in_data(self):
        """Test handling of masked NaN-like positions in data

        Note: This test verifies that positions resembling NaN (placeholders + mask=0)
        are properly handled by HPGL. HPGL expects valid float32 arrays, so NaN values
        must be masked (set to uninformed).
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100

        # Create mask - mark all as informed initially
        mask = np.ones(500, dtype="uint8")

        # Identify positions that would have NaN (for simulation)
        # In practice, users should mask these positions
        nan_positions = slice(None, None, 50)  # Every 50th element

        # Replace NaN-like positions with placeholder and mask them
        data_with_placeholder = data.copy()
        data_with_placeholder[nan_positions] = 0.0  # Placeholder value
        mask[nan_positions] = 0  # Mark as uninformed

        prop = ContProperty(data_with_placeholder, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        # Should handle masked NaN-like positions
        assert result.data.shape == (10, 10, 5)
        # Original informed cells should remain informed
        # Note: result.mask is now 3D, so we need to flatten it for comparison
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)
        # No NaN in result
        assert not np.any(np.isnan(result.data))

    def test_actual_nan_values_are_rejected(self):
        """Test that actual NaN values in ContProperty data are rejected loudly.

        The constructor validates data finiteness (geo.py:303), so a NaN in
        an informed cell raises instead of silently propagating NaN means
        through the C++ layer.
        """
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10, dtype="float32")
        mask = np.ones(50, dtype="uint8")

        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(data, mask)


# =============================================================================
# 3. PARAMETER VALIDATION
# =============================================================================


@pytest.mark.hpgl
class TestParameterValidation:
    """Test parameter validation and edge cases"""

    def test_minimal_radius(self):
        """Test with minimal search radius (1, 1, 1)

        Uses radius=1, the smallest valid value. With such a small radius,
        only the nearest cells can be used as neighbors, limiting the
        kriging estimation to very local information.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0  # 10% uninformed
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 1.0, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(1, 1, 1), max_neighbours=1, cov_model=cov_model
        )

        assert result.data.size == 500
        # Informed cells should remain informed
        assert np.all(result.mask.flat[mask == 1] == 1)

    def test_single_neighbor_max_neighbours_1(self):
        """Test with max_neighbours=1 (minimal neighborhood)

        Note: max_neighbours=1 means using only the nearest neighbor for estimation.
        This should work correctly but results may be less smooth.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=1, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)
        # Should have some informed cells in result
        assert np.any(result.mask == 1)

    def test_zero_neighbors(self):
        """Test with max_neighbours=0 - should raise validation error"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # max_neighbours=0 should raise validation error (min is 1)
        with pytest.raises(CriticalValidationError):
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=0, cov_model=cov_model
            )

    def test_negative_range(self):
        """Test with negative range value - should raise validation error"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        # Negative range should raise validation error.
        # H-08: pin the exact exception type — the CovarianceModel ctor rejects
        # r <= MIN_RANGE via validation.py:846-851 (CriticalValidationError).
        with pytest.raises(CriticalValidationError):
            cov_model = CovarianceModel(
                type=covariance.spherical,
                ranges=(-5.0, 5.0, 3.0),  # Negative X range
                angles=(0.0, 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            )
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
            )

    def test_negative_angle(self):
        """Test with negative angle values

        Note: Negative angles are valid (just rotation in opposite direction).
        HPGL should handle them correctly.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)

        # Negative angles are valid (just rotation direction)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(-45.0, -30.0, -15.0),
            sill=1.0,
            nugget=0.1,
        )

        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)
        # H-11: negative angles must produce valid finite output and preserve
        # the informed-mask (a garbage-of-the-right-shape result fails here).
        assert not np.any(np.isnan(result.data))
        assert not np.any(np.isinf(result.data))
        assert np.all(result.mask.flatten(order="F")[mask == 1] == 1)

    def test_mismatched_grid_size_vs_data(self):
        """Test with grid size that doesn't match data size

        Note: HPGL requires data size to match grid size. This test verifies
        that size mismatches are properly detected.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)  # 500 cells
        data = np.random.rand(400).astype("float32")  # Only 400 values - MISMATCH
        mask = np.ones(400, dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # H-09: fix_shape does NOT raise on mismatch (it no-ops when sizes
        # differ) — the raise comes from ordinary_kriging's size-vs-grid
        # ValueError check (geo.py:1761-1765).
        with pytest.raises(ValueError):
            prop.fix_shape(grid)
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
            )

    def test_wrong_data_type_int_instead_of_float(self):
        """Integer data is converted to float32 on construction (numpy.require)."""
        data = np.array([1, 2, 3, 4, 5] * 100, dtype="int32")  # 500 values
        mask = np.ones(500, dtype="uint8")

        prop = ContProperty(data, mask)
        assert prop.data.dtype == np.float32, (
            "int32 data must be converted to float32 (numpy.require)"
        )
        assert prop.data.flags["F_CONTIGUOUS"]

    def test_wrong_mask_type(self):
        """Integer mask is converted to uint8 on construction."""
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="int32")  # Wrong type

        prop = ContProperty(data, mask)
        assert prop.mask.dtype == np.uint8, (
            "int32 mask must be converted to uint8 (numpy.require)"
        )
        assert prop.mask.flags["F_CONTIGUOUS"]

    def test_very_large_max_neighbours(self):
        """Test with max_neighbours larger than available data"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0  # Only 450 informed
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(20.0, 20.0, 10.0),  # Large radius
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # max_neighbours=1000 but only 450 informed cells
        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(20, 20, 10), max_neighbours=1000, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)


# =============================================================================
# 4. PROPERTY EDGE CASES
# =============================================================================


@pytest.mark.hpgl
class TestPropertyEdgeCases:
    """Test edge cases related to property configurations"""

    def test_empty_property_no_informed_data(self):
        """Test property with all uninformed (no data)

        Note: When all data is uninformed, HPGL has no source data for kriging.
        All cells will remain uninformed (KI_NO_NEIGHBOURS for all cells).
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100
        mask = np.zeros(500, dtype="uint8")  # All uninformed
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # With no informed data, kriging will complete but all cells remain uninformed
        result = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )
        result.fix_shape(grid)  # HPGL returns 1D, reshape to grid dimensions

        assert result.data.shape == (10, 10, 5)
        # All cells should remain uninformed since there's no source data
        assert np.all(result.mask == 0)

    def test_indicator_at_boundary(self):
        """Test indicator with value at indicator_count - 1 boundary"""
        data = np.array([0, 1, 2], dtype="uint8")  # 2 is valid for indicator_count=3
        mask = np.ones(3, dtype="uint8")
        prop = IndProperty(data, mask, indicator_count=3)

        assert prop.indicator_count == 3
        # Value 2 is valid (0, 1, 2 for count=3)


# =============================================================================
# 5. SIMULATION EDGE CASES
# =============================================================================


@pytest.mark.hpgl
class TestSimulationEdgeCases:
    """Test edge cases for SGS and SIS simulations"""

    def test_same_seed_produces_same_result_sgs(self):
        """Test SGS determinism: same seed should produce same results"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::2] = 0  # 50% uninformed
        prop = ContProperty(data, mask)

        # Create CDF from data
        informed_data = data[mask == 1]
        cdf_data = CdfData(
            values=np.sort(informed_data),
            probs=np.linspace(0, 1, len(informed_data)).astype("float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        seed = 42

        result1 = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=seed,
        )

        result2 = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=seed,
        )

        # Same seed should produce identical results
        np.testing.assert_array_equal(result1.data, result2.data)

    def test_different_seeds_produce_different_results_sgs(self):
        """Test SGS: different seeds should produce different results"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        mask[::2] = 0
        prop = ContProperty(data, mask)

        informed_data = data[mask == 1]
        cdf_data = CdfData(
            values=np.sort(informed_data),
            probs=np.linspace(0, 1, len(informed_data)).astype("float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result1 = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
        )

        result2 = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=123,
        )

        # Different seeds should produce different results
        assert not np.array_equal(result1.data, result2.data)

    def test_use_harddata_false_starts_from_scratch_sgs(self):
        """Test SGS with use_harddata=False should ignore initial data"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.ones(500, dtype="float32") * 100.0  # All high values
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        # Use CDF with different range
        cdf_data = CdfData(
            values=np.array([0.0, 10.0, 20.0], dtype="float32"),
            probs=np.array([0.33, 0.66, 1.0], dtype="float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
            use_harddata=False,
        )

        # Results should be based on CDF, not initial 100.0 values
        # Most values should be in CDF range, not near 100
        assert np.mean(result.data) < 50.0

    def test_mask_covering_all_cells_simulate_nothing(self):
        """Test simulation with mask covering all cells (simulate nothing)

        Note: When simulate_mask has all zeros, no cells are selected for simulation.
        The original hard data should be preserved in the output.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        informed_data = data[mask == 1]
        cdf_data = CdfData(
            values=np.sort(informed_data),
            probs=np.linspace(0, 1, len(informed_data)).astype("float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # Mask with all zeros - simulate nothing (all zeros means don't simulate these cells)
        simulate_mask = np.zeros(500, dtype="uint8")

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
            mask=simulate_mask,
        )

        # Verify result shape and no NaN
        assert result.data.shape == (10, 10, 5)
        assert not np.any(np.isnan(result.data))
        # A-06/H-01: with an all-zero simulate_mask and use_harddata=True
        # (the default), the C++ gate (sequential_simulation.h: mask[node]==1)
        # skips EVERY node — the output must be the ORIGINAL hard data
        # preserved (within float32 normal-score round-trip error, ~1e-5).
        # If the mask were dropped from the FFI call (or ignored by the
        # kernel), every cell would be re-drawn from the CDF and diverge
        # from the input data by orders of magnitude.
        np.testing.assert_allclose(
            result.data, data.reshape((10, 10, 5), order="F"), rtol=1e-4, atol=1e-4
        )

    def test_mask_covering_no_cells_simulate_all(self):
        """Test simulation with mask covering no cells (simulate all)

        Note: With simulate_mask all ones and use_harddata=False, all cells are simulated
        from the CDF, ignoring the original property data.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(500).astype("float32") * 100
        mask = np.zeros(500, dtype="uint8")  # No hard data
        prop = ContProperty(data, mask)

        # Use synthetic CDF
        cdf_data = CdfData(
            values=np.array([0.0, 50.0, 100.0], dtype="float32"),
            probs=np.array([0.33, 0.66, 1.0], dtype="float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        # Mask with all ones - simulate all cells
        simulate_mask = np.ones(500, dtype="uint8")

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
            mask=simulate_mask,
            use_harddata=False,
        )

        # Should simulate all cells
        assert result.data.shape == (10, 10, 5)
        # Values should be from CDF range (approximately - may vary slightly)
        assert np.all(result.data >= 0) and np.all(np.isfinite(result.data))
        # A-06/H-02: use_harddata=False + all-ones simulate_mask must actually
        # SIMULATE all cells from the CDF — an all-zeros output (no simulation
        # at all) would pass the >= 0 check but must fail here.
        assert np.all(result.data <= 100), (
            "Simulated values must be within the CDF range [0, 100]"
        )
        assert np.std(result.data.astype("float64")) > 0.0, (
            "use_harddata=False + all-ones mask must produce varied CDF draws, "
            "not a constant (empty-clone) output"
        )

    def test_sis_same_seed_determinism(self):
        """Test SIS determinism: same seed produces same results"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.randint(0, 3, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")
        mask[::2] = 0
        prop = IndProperty(data, mask, indicator_count=3)

        # Setup IK data
        ik_data = []
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

        marginal_probs = [0.33, 0.34, 0.33]
        seed = 42

        result1 = sis_simulation(
            prop=prop, grid=grid, data=ik_data, seed=seed, marginal_probs=marginal_probs
        )

        result2 = sis_simulation(
            prop=prop, grid=grid, data=ik_data, seed=seed, marginal_probs=marginal_probs
        )

        # Same seed should produce identical results
        np.testing.assert_array_equal(result1.data, result2.data)

    def test_simulation_with_zero_radius(self):
        """Test simulation with zero search radius

        Note: Zero radius means no neighbors can be found during simulation.
        HPGL will use random values from CDF for cells that can't find neighbors.
        With use_harddata=True, original informed cells should be preserved.
        """
        grid = SugarboxGrid(x=5, y=5, z=5)
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(125).astype("float32") * 100
        mask = np.ones(125, dtype="uint8")
        mask[::2] = 0  # Half uninformed
        prop = ContProperty(data, mask)

        cdf_data = CdfData(
            values=np.linspace(0, 100, 50).astype("float32"),
            probs=np.linspace(0, 1, 50).astype("float32"),
        )

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(1.0, 1.0, 1.0),  # Use non-zero for covariance model
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(0, 0, 0),  # But zero search radius
            max_neighbours=1,
            cov_model=cov_model,
            seed=42,
        )

        # Should complete even with zero radius
        assert result.data.shape == (5, 5, 5)
        # All cells should have some result (simulated or original)
        assert np.all(np.isfinite(result.data))
        # I2-F24: with zero search radius, SGS must draw from CDF for uninformed cells.
        # Verify output values fall within the CDF range and reflect CDF-random behavior.
        cdf_min = float(cdf_data.values[0])
        cdf_max = float(cdf_data.values[-1])
        uninformed = result.data[result.mask == 0] if np.any(result.mask == 0) else result.data
        assert np.all(uninformed >= cdf_min), (
            f"Simulation output below CDF min {cdf_min}"
        )
        assert np.all(uninformed <= cdf_max), (
            f"Simulation output above CDF max {cdf_max}"
        )
        # With zero radius, uninformed cells get random CDF draws — verify
        # they are not all identical (which would indicate degenerate behavior).
        if len(uninformed) > 1:
            assert np.std(uninformed.astype("float64")) > 0.0, (
                "Zero-radius simulation should produce varied CDF-random output"
            )

    def test_ik_with_indicator_count_1(self):
        """C009 — IK with indicator_count=1 (degenerate single-category case).

        Single-category IK is a degenerate case that must not crash and must
        produce valid output of the correct shape. All data values are 0
        (the only valid indicator for count=1).
        """
        from geo_bsd.geo import indicator_kriging

        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.zeros(50, dtype="uint8")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = IndProperty(data, mask, indicator_count=1)

        cov = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.0
        )

        ik_data = [{"cov_model": cov, "radiuses": (2, 2, 1), "max_neighbours": 6}]

        result = indicator_kriging(prop=prop, grid=grid, data=ik_data, marginal_probs=(1.0,))

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 1
        # HPGL returns flat 1D output; verify cell count matches grid
        assert result.data.size == 50  # 5*5*2
        # With count=1, all indicator values must be 0
        assert np.all(result.data == 0)

    def test_sis_with_indicator_count_1(self):
        """C009 — SIS with indicator_count=1 (degenerate single-category case).

        Single-category SIS is a degenerate case that must not crash. All
        simulated cells should receive category 0 (the only possible value).
        """
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=5, y=5, z=2)
        data = np.zeros(50, dtype="uint8")
        mask = np.ones(50, dtype="uint8")
        mask[::5] = 0
        prop = IndProperty(data, mask, indicator_count=1)

        cov = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.0
        )

        sis_data = [{"cov_model": cov, "radiuses": (2, 2, 1), "max_neighbours": 6}]

        result = sis_simulation(prop=prop, grid=grid, data=sis_data, seed=42, marginal_probs=(1.0,))

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 1
        # HPGL returns flat 1D output; verify cell count matches grid
        assert result.data.size == 50  # 5*5*2
        # With count=1, all values must be 0
        assert np.all(result.data == 0)


# =============================================================================
# 6. CDF EDGE CASES
# =============================================================================


@pytest.mark.hpgl
class TestCDFEdgeCases:
    """Test edge cases for cumulative distribution functions

    Note: calc_cdf expects properties with 3D shaped data.
    These tests create properties with proper 3D shape using fix_shape.
    """

    def test_empty_cdf_no_values(self):
        """Test CDF calculation with property with no informed values raises ValueError"""
        grid = SugarboxGrid(x=2, y=2, z=2)  # 8 cells
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float32")
        mask = np.zeros(8, dtype="uint8")  # All uninformed
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        with pytest.raises(ValueError, match="no informed values"):
            calc_cdf(prop)

    def test_single_value_cdf(self):
        """Test CDF with only one unique value"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        data = np.array([42.0] * 100, dtype="float32")
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # Single value case: HPGL returns size 1 (the single value itself)
        assert cdf.values.size == 1
        assert cdf.probs.size == 1
        assert cdf.values[0] == 42.0
        # F-04: last CDF probability is clamped strictly below 1.0 so the
        # max datum does not map to p=1.0 in the SGS back-transform.
        assert cdf.probs[0] < 1.0

    def test_uniform_distribution_cdf(self):
        """Test CDF with uniformly distributed values"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        # Create uniform distribution
        data = np.array([1, 2, 3, 4, 5] * 20, dtype="float32")  # Each value 20 times
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # Returns all N unique values
        # 5 unique values -> 5 values
        assert cdf.values.size == 5
        # F-04/T-16: the final cumulative probability is clamped strictly
        # below 1.0 (nextafter(1.0f, 0.0f)) so the max datum maps to a large
        # but finite normal score in the SGS back-transform, not the median.
        # With the clamp removed the tail is exactly 1.0 and the equality fails.
        assert cdf.probs[-1] == np.nextafter(np.float32(1.0), np.float32(0.0))
        expected_probs = np.array([0.2, 0.4, 0.6, 0.8], dtype="float32")
        np.testing.assert_array_almost_equal(cdf.probs[:-1], expected_probs, decimal=5)

    def test_cdf_with_two_unique_values(self):
        """Test CDF with exactly two unique values"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        data = np.array([1.0] * 50 + [2.0] * 50, dtype="float32")
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # Two unique values -> returns both values
        assert cdf.values.size == 2
        assert cdf.values[0] == 1.0
        assert cdf.values[1] == 2.0
        assert cdf.probs[0] == 0.5  # 50% of values at/below first value
        # F-04: final cumulative probability is clamped strictly below 1.0.
        assert cdf.probs[-1] < 1.0

    def test_cdf_with_many_unique_values(self):
        """Test CDF with many unique values"""
        grid = SugarboxGrid(x=10, y=10, z=10)  # 1000 cells
        np.random.seed(42)
        data = np.random.rand(1000).astype("float32") * 100
        mask = np.ones(1000, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # Returns all N unique values
        # With 1000 random values, most will be unique, so expect ~1000 values
        assert cdf.values.size > 900  # Allow for some duplicates
        # Probabilities should be monotonically increasing (not strictly - can have ties)
        assert np.all(np.diff(cdf.probs) >= 0)

    def test_cdf_values_are_sorted(self):
        """Test that CDF values are always sorted"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # HPGL calc_cdf sorts values, so they should be in ascending order
        if len(cdf.values) > 1:
            assert np.all(cdf.values[:-1] <= cdf.values[1:])

    def test_cdf_probs_are_monotonic(self):
        """Test that CDF probabilities are monotonically increasing"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # HPGL calc_cdf probabilities are cumulative, so should be monotonically increasing
        if len(cdf.probs) > 1:
            assert np.all(cdf.probs[:-1] <= cdf.probs[1:])

    def test_cdf_final_probability_is_one(self):
        """Test that final CDF probability is 1.0"""
        grid = SugarboxGrid(x=5, y=5, z=4)  # 100 cells
        np.random.seed(42)
        data = np.random.rand(100).astype("float32") * 100
        mask = np.ones(100, dtype="uint8")
        prop = ContProperty(data, mask)
        prop.fix_shape(grid)  # Make 3D for calc_cdf

        cdf = calc_cdf(prop)

        # F-04/T-16: the many-value CDF tail is clamped strictly below 1.0
        # (nextafter(1.0f, 0.0f)). This is the many-value clamp pin: with the
        # clamp removed the tail is exactly 1.0 and the first assert fails.
        assert cdf.probs[-1] < 1.0
        assert cdf.probs[-1] > 1 - 1e-6


# =============================================================================
# UTILITY TESTS
# =============================================================================


@pytest.mark.hpgl
class TestUtilityEdgeCases:
    """Test edge cases for utility functions"""

    def test_calc_mean_with_negative_values(self):
        """Test calc_mean with negative values"""
        data = np.array([-50.0, 0.0, 50.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)

        mean = calc_mean(prop)
        assert mean == 0.0

    def test_empty_clone_preserves_indicator_count(self):
        """Test that _empty_clone preserves indicator count"""
        from geo_bsd.geo import _empty_clone

        data = np.random.randint(0, 3, 100, dtype="uint8")
        mask = np.ones(100, dtype="uint8")
        prop = IndProperty(data, mask, 5)

        cloned = _empty_clone(prop)

        assert isinstance(cloned, IndProperty)
        assert cloned.indicator_count == 5
        assert np.all(cloned.data == 0)
        assert np.all(cloned.mask == 0)

    def test_clone_property_preserves_data(self):
        """Test that _clone_prop creates a proper copy"""
        from geo_bsd.geo import _clone_prop

        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.array([1, 1, 0], dtype="uint8")
        prop = ContProperty(data, mask)

        cloned = _clone_prop(prop)

        assert isinstance(cloned, ContProperty)
        np.testing.assert_array_equal(cloned.data, data)
        np.testing.assert_array_equal(cloned.mask, mask)
        # Modifying clone should not affect original
        cloned.data[0] = 999.0
        assert prop.data[0] == 1.0


@pytest.mark.hpgl
class TestProductionFixes:
    """Tests for production readiness fixes applied to the codebase."""

    def test_load_cont_slow_skips_non_numeric_tokens(self):
        """_load_prop_cont_slow should skip non-numeric tokens without crashing."""
        import os
        import tempfile

        from geo_bsd.geo import _load_prop_cont_slow

        content = "-- comment line\n1.0 2.0 BADTOKEN 3.0\n-- another comment\n4.0\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inc", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmpfile = f.name
        try:
            # F-28: pass an explicit trusted base (tempfile lives outside cwd).
            prop = _load_prop_cont_slow(tmpfile, -99.0, basedir=str(Path(tmpfile).parent))
            assert len(prop.data) == 4  # 1.0, 2.0, 3.0, 4.0 (BADTOKEN skipped)
            assert np.all(prop.mask == 1)
        finally:
            os.remove(tmpfile)

    def test_load_ind_slow_skips_non_numeric_tokens(self):
        """_load_prop_ind_slow should skip non-numeric tokens without crashing."""
        import os
        import tempfile

        from geo_bsd.geo import _load_prop_ind_slow

        content = "0 1 BADTOKEN 0 1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inc", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmpfile = f.name
        try:
            # F-28: pass an explicit trusted base (tempfile lives outside cwd).
            prop = _load_prop_ind_slow(tmpfile, -99, [0, 1], basedir=str(Path(tmpfile).parent))
            assert len(prop.data) == 4  # 0, 1, 0, 1 (BADTOKEN skipped)
        finally:
            os.remove(tmpfile)


# =============================================================================
# Production Fix Tests: Error Handling & Indicator Kriging Correctness
# =============================================================================


@pytest.mark.hpgl
class TestErrorHandling:
    """Tests for the _check_hpgl_error error handling (CRITICAL fix)."""

    def test_hpgl_get_last_exception_message_restype_set(self):
        """CRITICAL: Verify restype is correctly set to c_char_p, not default c_int."""
        from geo_bsd.hpgl_wrap import _hpgl_so

        assert hasattr(_hpgl_so, "hpgl_get_last_exception_message"), (
            "hpgl_get_last_exception_message not loaded"
        )
        restype = _hpgl_so.hpgl_get_last_exception_message.restype
        import ctypes as C

        assert restype == C.c_char_p, (
            f"restype must be c_char_p to avoid pointer truncation, got {restype}"
        )


@pytest.mark.hpgl
class TestIndicatorKrigingFix:
    """Tests for the most_probable_category PMF fix (HIGH priority)."""

    def test_most_probable_category_via_median_ik(self):
        """Verify that the most_probable_category fix produces correct results
        through median_ik with balanced probabilities (should not always pick
        highest index)."""
        import numpy as np

        from geo_bsd.geo import CovarianceModel, IndProperty, SugarboxGrid, covariance, median_ik

        grid = SugarboxGrid(x=5, y=5, z=2)
        # Create indicator data: all category 0 (informed)
        data = np.zeros(50, dtype="uint8")
        mask = np.ones(50, dtype="uint8")
        prop = IndProperty(data, mask, 2)

        cov = CovarianceModel(
            type=covariance.spherical, ranges=(10.0, 10.0, 5.0), sill=1.0, nugget=0.1
        )

        # With marginal probs [0.9, 0.1], kriging should heavily favor category 0.
        # H-03: the pre-fix always-highest-index bug (cdf_utils.cpp
        # most_probable_category) produces all-1s output — `data <= 1` alone
        # cannot discriminate it; majority-0 dominance is the discriminator.
        result = median_ik(prop, grid, (0.9, 0.1), (3, 3, 1), 12, cov)
        assert result.indicator_count == 2
        assert np.all(result.data <= 1), "Indicator values must be 0 or 1"
        assert np.mean(result.data == 0) > 0.9, (
            "With p=[0.9, 0.1] and all-0 hard data, most cells must be category 0"
        )

    def test_indicator_kriging_with_3_categories(self):
        """Verify indicator_kriging with K=3 categories doesn't crash and produces
        valid results after the most_probable_category fix."""
        import numpy as np

        from geo_bsd.geo import (
            CovarianceModel,
            IndProperty,
            SugarboxGrid,
            covariance,
            indicator_kriging,
        )

        grid = SugarboxGrid(x=6, y=6, z=3)
        data = np.random.randint(0, 3, 108, dtype="uint8")
        mask = np.ones(108, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        cov0 = CovarianceModel(
            type=covariance.spherical, ranges=(8.0, 8.0, 4.0), sill=1.0, nugget=0.1
        )
        cov1 = CovarianceModel(
            type=covariance.spherical, ranges=(8.0, 8.0, 4.0), sill=1.0, nugget=0.1
        )
        cov2 = CovarianceModel(
            type=covariance.spherical, ranges=(8.0, 8.0, 4.0), sill=1.0, nugget=0.1
        )

        ik_data = [
            {"cov_model": cov0, "radiuses": (2, 2, 1), "max_neighbours": 12},
            {"cov_model": cov1, "radiuses": (2, 2, 1), "max_neighbours": 12},
            {"cov_model": cov2, "radiuses": (2, 2, 1), "max_neighbours": 12},
        ]

        result = indicator_kriging(prop, grid, ik_data, (1.0 / 3, 1.0 / 3, 1.0 / 3))
        assert result.indicator_count == 3
        # All output values should be valid indicator values (0, 1, or 2)
        assert np.all(result.data >= 0) and np.all(result.data < 3), (
            f"Invalid indicator values: min={result.data.min()}, max={result.data.max()}"
        )


# =============================================================================
# NaN/Inf Input Handling Tests
# =============================================================================


@pytest.mark.hpgl
class TestNaNInfInputHandling:
    """Test that HPGL properly handles NaN and Inf inputs."""

    def test_covariance_model_nan_sill_raises(self):
        """CovarianceModel with NaN sill should raise CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=float("nan"), nugget=0.1
            )

    def test_covariance_model_inf_sill_raises(self):
        """CovarianceModel with Inf sill should raise CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=float("inf"), nugget=0.1
            )

    def test_covariance_model_nan_nugget_raises(self):
        """CovarianceModel with NaN nugget should raise."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=float("nan")
            )

    def test_covariance_model_inf_nugget_raises(self):
        """CovarianceModel with Inf nugget should raise."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=float("inf")
            )

    def test_covariance_model_nan_range_raises(self):
        """CovarianceModel with NaN range should raise."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(float("nan"), 5.0, 3.0), sill=1.0, nugget=0.1
            )

    def test_covariance_model_inf_range_raises(self):
        """CovarianceModel with Inf range should raise."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical, ranges=(float("inf"), 5.0, 3.0), sill=1.0, nugget=0.1
            )

    def test_covariance_model_nan_angle_raises(self):
        """CovarianceModel with NaN angle should raise."""
        with pytest.raises(CriticalValidationError):
            CovarianceModel(
                type=covariance.spherical,
                ranges=(5.0, 5.0, 3.0),
                angles=(float("nan"), 0.0, 0.0),
                sill=1.0,
                nugget=0.1,
            )

    def test_cont_property_nan_data_construction(self):
        """ContProperty rejects NaN data at construction."""
        data = np.array([1.0, np.nan, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(data, mask)

    def test_cont_property_inf_data_construction(self):
        """ContProperty rejects Inf data at construction."""
        data = np.array([1.0, np.inf, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(data, mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
