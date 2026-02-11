"""
Tests for all kriging algorithms:
- Ordinary Kriging (OK)
- Simple Kriging (SK)
- Indicator Kriging (IK)
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
        SugarboxGrid, calc_mean
    )
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestOrdinaryKriging:
    """Test Ordinary Kriging algorithm"""
    
    def test_ok_basic_execution(self, sample_property, sample_grid, sample_covariance_model):
        """Test basic OK execution completes without errors"""
        result = ordinary_kriging(
            prop=sample_property,
            grid=sample_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model
        )
        
        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape
        assert result.mask.shape == sample_property.mask.shape
    
    def test_ok_result_bounds(self, sample_property, sample_grid, sample_covariance_model):
        """Test OK produces results within reasonable bounds"""
        result = ordinary_kriging(
            prop=sample_property,
            grid=sample_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model
        )
        
        # Results should not be all zeros or NaN
        assert not np.all(result.data == 0)
        assert not np.any(np.isnan(result.data.astype('float64')))
        assert not np.any(np.isinf(result.data.astype('float64')))


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSimpleKriging:
    """Test Simple Kriging algorithm"""
    
    def test_sk_basic_execution(self, sample_property, sample_grid, sample_covariance_model):
        """Test basic SK execution completes without errors"""
        result = simple_kriging(
            prop=sample_property,
            grid=sample_grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            mean=None
        )
        
        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestIndicatorKriging:
    """Test Indicator Kriging algorithm"""
    
    def test_ik_basic_execution(self, sample_indicator_property, sample_grid):
        """Test basic IK execution"""
        ik_data = []
        marginal_probs = [0.3, 0.4, 0.3]
        
        for i in range(3):
            ik_data.append({
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
        
        result = indicator_kriging(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=ik_data,
            marginal_probs=marginal_probs
        )
        
        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
