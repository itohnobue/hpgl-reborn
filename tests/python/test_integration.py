"""
Integration tests for HPGL workflows
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import CdfData
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        calc_mean,
        covariance,
        indicator_kriging,
        load_cont_property,
        ordinary_kriging,
        write_property,
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


@pytest.mark.hpgl
@pytest.mark.integration
class TestWorkflowIntegration:
    """Test complete geostatistical workflows"""

    def test_kriging_then_simulation(self):
        """Test using kriging results for simulation"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1
        )

        # First run kriging
        kriged = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )

        # Data integrity: kriged output has no NaN/Inf
        assert isinstance(kriged, ContProperty)
        assert not np.any(np.isnan(kriged.data.astype("float64")))
        assert not np.any(np.isinf(kriged.data.astype("float64")))
        assert kriged.data.shape == (500,)

        # Use kriged result for simulation
        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype="float32"),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32"),
        )

        sim_result = sgs_simulation(
            prop=kriged,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
        )

        assert isinstance(sim_result, ContProperty)
        # Data integrity: simulation output is finite
        sim_f64 = sim_result.data.astype("float64")
        assert np.all(np.isfinite(sim_f64)), "SGS output must have all finite values"
        assert np.any(sim_f64 != 0), "SGS output must not be all zeros"

    def test_multiple_realizations_workflow(self):
        """Test creating multiple realizations"""
        grid = SugarboxGrid(x=10, y=10, z=5)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1
        )

        cdf_data = CdfData(
            np.array([0.0, 50.0, 100.0], dtype="float32"),
            np.array([0.0, 0.5, 1.0], dtype="float32"),
        )

        realizations = []
        for i in range(3):
            # Create fresh input property for each realization
            # because SGS modifies the input property in-place
            data = np.random.rand(500).astype("float32") * 100
            mask = np.ones(500, dtype="uint8")
            prop = ContProperty(data, mask)

            result = sgs_simulation(
                prop=prop,
                grid=grid,
                cdf_data=cdf_data,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
                seed=1000 + i,
            )
            realizations.append(result)

        assert len(realizations) == 3
        # Each realization should be different
        for i in range(1, 3):
            assert not np.array_equal(realizations[0].data, realizations[i].data)
        # Data integrity: all values finite, reasonable range
        for i, r in enumerate(realizations):
            r_f64 = r.data.astype("float64")
            assert np.all(np.isfinite(r_f64)), f"Realization {i} has non-finite values"
            assert np.std(r_f64) > 0.0, f"Realization {i} has zero variance"
            # Values should be within the CDF range [0, 100] with tolerance
            assert np.min(r_f64) >= -1.0, f"Realization {i} below CDF minimum"
            assert np.max(r_f64) <= 101.0, f"Realization {i} above CDF maximum"

    def test_sgs_returns_new_property_preserves_input(self):
        """F-215: sgs_simulation returns a new ContProperty and does NOT mutate input data.

        sgs_simulation clones the input property internally (_clone_prop at sgs.py:168),
        mutates the clone in C++, and returns the clone. The original prop.data values
        are preserved (only shape may change via fix_shape).
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1
        )

        cdf_data = CdfData(
            np.array([0.0, 50.0, 100.0], dtype="float32"),
            np.array([0.0, 0.5, 1.0], dtype="float32"),
        )

        # Save original data values (flattened to handle potential reshaping)
        original_values = prop.data.ravel().copy()

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
        )

        # Verify result is a NEW ContProperty (not the same object as input)
        assert result is not prop, (
            "sgs_simulation should return a new ContProperty, not the input"
        )

        # Verify original data values are preserved (use Fortran-order ravel
        # since fix_shape reshapes to Fortran-order 3D)
        current_values = prop.data.ravel(order="F")
        assert np.array_equal(original_values, current_values), (
            "sgs_simulation should NOT mutate input property data values"
        )

        # Verify result is a valid ContProperty with correct shape
        assert isinstance(result, ContProperty)
        expected_size = grid.x * grid.y * grid.z
        assert result.data.size == expected_size, (
            f"Result data size {result.data.size} != grid size {expected_size}"
        )
        # Result should have 3D shape after internal fix_shape
        assert result.data.ndim == 3, (
            f"Result data should be 3D after fix_shape, got ndim={result.data.ndim}"
        )

        # Verify result has finite values (valid simulation output)
        assert np.all(np.isfinite(result.data)), "Result contains NaN or Inf"


@pytest.mark.hpgl
@pytest.mark.integration
class TestIOIntegration:
    """Test data I/O workflows"""

    def test_property_roundtrip(self, tmp_path):
        """Test writing and reading properties"""
        data = np.arange(500, dtype="float32") % 100
        mask = np.ones(500, dtype="uint8")
        prop = ContProperty(data, mask)

        # Write property
        output_file = tmp_path / "test_output.inc"
        write_property(prop, str(output_file), "TestProperty", -999.0)

        # Verify file was created
        assert output_file.exists()

        # Read back and verify data integrity
        read_prop = load_cont_property(str(output_file), -999.0, (10, 10, 5))
        assert isinstance(read_prop, ContProperty)
        # Data is returned as 1D Fortran-order array of total grid size
        assert read_prop.data.shape == (500,)
        # Informed cells should match original data
        informed = mask.astype(bool)
        np.testing.assert_array_equal(read_prop.data[informed], data[informed])


# =============================================================================
# Multi-Stage Workflow Tests (M21)
# =============================================================================


@pytest.mark.hpgl
@pytest.mark.integration
class TestMultiStageWorkflows:
    """Test multi-stage geostatistical workflows: variogram→kriging, IK→SIS."""

    def test_variogram_to_kriging_chain(self):
        """Compute variogram model from data, then use it for ordinary_kriging.

        The CovarianceModel serves as the variogram model (they are equivalent
        in the HPGL framework). This test validates that a covariance model
        constructed from data characteristics can drive a complete kriging
        workflow.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")

        # Step 1: Compute mean and variance from data (simulates variogram analysis)
        prop = ContProperty(data, mask)
        data_mean = calc_mean(prop)
        data_std = np.sqrt(np.mean((data - data_mean) ** 2))

        # Step 2: Build covariance/variogram model from data statistics
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=float(data_std**2),
            nugget=float(data_std**2 * 0.1),
        )

        # Step 3: Run ordinary kriging with the derived model
        kriged = ordinary_kriging(
            prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
        )

        assert isinstance(kriged, ContProperty)
        assert kriged.data.shape == (500,)
        assert not np.any(np.isnan(kriged.data.astype("float64")))
        assert not np.any(np.isinf(kriged.data.astype("float64")))
        # Data integrity: kriged values should be within reasonable range
        kriged_f64 = kriged.data.astype("float64")
        assert np.std(kriged_f64) > 0.0, "Kriged output must have non-zero variance"
        assert np.min(kriged_f64) >= -50.0, "Kriged values must not be far below zero"

    def test_indicator_kriging_to_sis_chain(self):
        """Run indicator_kriging, then use results for sis_simulation.

        This validates the complete IK→SIS multi-stage workflow:
        indicator kriging produces probability maps, which feed into
        sequential indicator simulation for realization generation.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)
        size = grid.x * grid.y * grid.z

        # Create indicator property (3 categories)
        data = np.random.randint(0, 3, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        ind_prop = IndProperty(data, mask, 3)

        # Step 1: Setup indicator kriging data
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

        # Step 2: Run indicator kriging
        ik_result = indicator_kriging(
            prop=ind_prop, grid=grid, data=ik_data, marginal_probs=marginal_probs
        )
        assert isinstance(ik_result, IndProperty)
        assert ik_result.indicator_count == 3

        # Step 3: Run SIS using IK results
        sis_result = sis_simulation(
            prop=ik_result, grid=grid, data=ik_data, seed=42, marginal_probs=marginal_probs
        )

        assert isinstance(sis_result, IndProperty)
        assert sis_result.indicator_count == 3
        assert sis_result.data.shape == ik_result.data.shape
        # Data integrity: SIS output has valid indicator categories
        sis_data = sis_result.data.astype("uint8")
        assert np.all(sis_data < 3), "SIS indicator values must be within category count"
        # Verify category distribution is reasonable (roughly matches marginal probs)
        category_counts = [int(np.sum(sis_data == c)) for c in range(3)]
        total = sum(category_counts)
        for c in range(3):
            observed = category_counts[c] / total
            expected = marginal_probs[c]
            # Allow generous 0.15 tolerance for small sample randomness
            assert abs(observed - expected) < 0.20, (
                f"Category {c}: observed proportion {observed:.2f} too far from expected {expected:.2f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
