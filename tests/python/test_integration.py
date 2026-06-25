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
        simple_kriging,
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
        data = np.random.rand(500).astype('float32') * 100
        mask = np.ones(500, dtype='uint8')
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )

        # First run kriging
        kriged = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model
        )

        # Use kriged result for simulation
        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )

        sim_result = sgs_simulation(
            prop=kriged,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42
        )

        assert isinstance(sim_result, ContProperty)

    def test_multiple_realizations_workflow(self):
        """Test creating multiple realizations"""
        grid = SugarboxGrid(x=10, y=10, z=5)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )

        cdf_data = CdfData(
            np.array([0.0, 50.0, 100.0], dtype='float32'),
            np.array([0.0, 0.5, 1.0], dtype='float32')
        )

        realizations = []
        for i in range(3):
            # Create fresh input property for each realization
            # because SGS modifies the input property in-place
            data = np.random.rand(500).astype('float32') * 100
            mask = np.ones(500, dtype='uint8')
            prop = ContProperty(data, mask)

            result = sgs_simulation(
                prop=prop,
                grid=grid,
                cdf_data=cdf_data,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model,
                seed=1000 + i
            )
            realizations.append(result)

        assert len(realizations) == 3
        # Each realization should be different
        for i in range(1, 3):
            assert not np.array_equal(realizations[0].data, realizations[i].data)

    def test_sgs_modifies_input_property_in_place(self):
        """Test SGS modifies the input property array in-place (side-effect).

        Verifies the identity (same object) and content change (data modified)
        of the input property after SGS simulation.
        """
        grid = SugarboxGrid(x=10, y=10, z=5)
        np.random.seed(42)
        data = np.random.rand(500).astype('float32') * 100
        mask = np.ones(500, dtype='uint8')
        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )

        cdf_data = CdfData(
            np.array([0.0, 50.0, 100.0], dtype='float32'),
            np.array([0.0, 0.5, 1.0], dtype='float32')
        )

        # Save original data copy and object identity
        original_data = prop.data.copy()
        prop_id_before = id(prop)

        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42
        )

        # Verify identity: same object reference (in-place modification)
        prop_id_after = id(prop)
        assert prop_id_before == prop_id_after, (
            "SGS should modify the input property in-place (same object identity)"
        )

        # Verify content: data has been modified
        assert not np.array_equal(original_data, prop.data), (
            "SGS should modify the input property data in-place"
        )

        # Verify the result is a valid ContProperty
        assert isinstance(result, ContProperty)
        assert result.data.shape == prop.data.shape


@pytest.mark.hpgl
@pytest.mark.integration
class TestIOIntegration:
    """Test data I/O workflows"""

    def test_property_roundtrip(self, tmp_path):
        """Test writing and reading properties"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.arange(500, dtype='float32') % 100
        mask = np.ones(500, dtype='uint8')
        prop = ContProperty(data, mask)

        # Write property
        output_file = tmp_path / "test_output.inc"
        write_property(
            prop,
            str(output_file),
            "TestProperty",
            -999.0
        )

        # Verify file was created
        assert output_file.exists()

        # Read back and verify data integrity
        read_prop = load_cont_property(str(output_file), -999.0, (10, 10, 5))
        assert isinstance(read_prop, ContProperty)
        # Data is returned as 1D Fortran-order array of total grid size
        assert read_prop.data.shape == (500,)
        # Informed cells should match original data
        informed = mask.astype(bool)
        np.testing.assert_array_equal(
            read_prop.data[informed],
            data[informed]
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
