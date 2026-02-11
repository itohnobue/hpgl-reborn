"""Tests for simulation algorithms"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "geo_bsd"))

try:
    import geo_bsd
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
    from geo_bsd.geo import ContProperty, IndProperty, CovarianceModel, covariance, SugarboxGrid
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")







@pytest.fixture
def sample_cdf_data():
    """Create sample CDF data for SGS"""
    values = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype="float32")
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32")
    return CdfData(values, probs)

@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialGaussianSimulation:
    """Test Sequential Gaussian Simulation"""
    
    def test_sgs_basic_execution_sk(self, sample_property, sample_grid, sample_covariance_model, sample_cdf_data):
        """Test SGS with Simple Kriging"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sample_cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk"
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape




    def test_sgs_reproducibility(self, sample_property, sample_grid, sample_covariance_model, sample_cdf_data):
        """Test SGS produces same results with same seed"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sample_cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345
        )
        
        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sample_cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345
        )
        
        np.testing.assert_array_equal(result1.data, result2.data)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSequentialIndicatorSimulation:
    """Test Sequential Indicator Simulation"""
    
    def test_sis_basic_execution(self, sample_indicator_property, sample_grid):
        """Test basic SIS execution"""
        sis_data = []
        marginal_probs = [0.3, 0.4, 0.3]
        
        for i in range(3):
            sis_data.append({
                'cov_model': CovarianceModel(
                    type=covariance.spherical,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1
                ),
                'radiuses': (5, 5, 3),
                'max_neighbours': 12
            })
        
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data,
            seed=42,
            marginal_probs=marginal_probs
        )
        
        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
