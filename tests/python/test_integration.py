"""
Integration tests for HPGL workflows
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
        SugarboxGrid, calc_mean, write_property
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
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


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
