"""
Performance benchmarking tests for HPGL
"""
import numpy as np
import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ordinary_kriging, simple_kriging,
        ContProperty, SugarboxGrid, CovarianceModel, covariance,
        calc_mean
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.slow
class TestPerformance:
    """Performance benchmarking tests"""
    
    def test_ok_small_grid_performance(self):
        """Benchmark ordinary kriging on small grid (10x10x5)"""
        grid = SugarboxGrid(x=10, y=10, z=5)
        data = np.random.rand(500).astype('float32') * 100
        mask = np.ones(500, dtype='uint8')
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )
        
        start = time.time()
        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model
        )
        elapsed = time.time() - start
        
        print(f"OK on 10x10x5 grid: {elapsed:.3f}s")
        assert elapsed < 10.0, "OK took too long"
    
    def test_ok_medium_grid_performance(self):
        """Benchmark ordinary kriging on medium grid (50x50x20)"""
        grid = SugarboxGrid(x=50, y=50, z=20)
        data = np.random.rand(50000).astype('float32') * 100
        mask = np.ones(50000, dtype='uint8')
        mask[::10] = 0
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 5.0),
            sill=1.0,
            nugget=0.1
        )
        
        start = time.time()
        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 5),
            max_neighbours=12,
            cov_model=cov_model
        )
        elapsed = time.time() - start
        
        print(f"OK on 50x50x20 grid: {elapsed:.3f}s")
        assert elapsed < 120.0, "OK took too long"
    
    def test_sgs_small_grid_performance(self):
        """Benchmark SGS on small grid"""
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
        cdf_data = CdfData(
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype='float32'),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
        )
        
        start = time.time()
        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf_data,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42
        )
        elapsed = time.time() - start
        
        print(f"SGS on 10x10x5 grid: {elapsed:.3f}s")
        assert elapsed < 30.0, "SGS took too long"
    
    def test_neighbour_count_performance_impact(self):
        """Test performance impact of different neighbour counts"""
        grid = SugarboxGrid(x=20, y=20, z=10)
        data = np.random.rand(4000).astype('float32') * 100
        mask = np.ones(4000, dtype='uint8')
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )
        
        timings = {}
        for max_neighbours in [4, 8, 12, 16]:
            start = time.time()
            result = ordinary_kriging(
                prop=prop,
                grid=grid,
                radiuses=(5, 5, 3),
                max_neighbours=max_neighbours,
                cov_model=cov_model
            )
            elapsed = time.time() - start
            timings[max_neighbours] = elapsed
            print(f"OK with {max_neighbours} neighbours: {elapsed:.3f}s")
            # Each run should complete in reasonable time
            assert elapsed < 5.0, f"Kriging with {max_neighbours} neighbours took too long"
    
    def test_covariance_type_performance(self):
        """Test performance of different covariance types"""
        grid = SugarboxGrid(x=15, y=15, z=8)
        data = np.random.rand(1800).astype('float32') * 100
        mask = np.ones(1800, dtype='uint8')
        prop = ContProperty(data, mask)
        
        timings = {}
        for cov_type, cov_name in [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian")
        ]:
            cov_model = CovarianceModel(
                type=cov_type,
                ranges=(5.0, 5.0, 3.0),
                sill=1.0,
                nugget=0.1
            )
            
            start = time.time()
            result = ordinary_kriging(
                prop=prop,
                grid=grid,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=cov_model
            )
            elapsed = time.time() - start
            timings[cov_name] = elapsed
            print(f"OK with {cov_name}: {elapsed:.3f}s")
        
        # All should complete within reasonable time
        for cov_name, elapsed in timings.items():
            assert elapsed < 30.0, f"{cov_name} took too long"


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
def test_mean_calculation_performance():
    """Test mean calculation performance"""
    from geo_bsd.geo import ContProperty
    large_data = np.random.rand(100000).astype('float32') * 100
    large_mask = np.ones(100000, dtype='uint8')
    prop = ContProperty(large_data, large_mask)
    
    start = time.time()
    mean_val = calc_mean(prop)
    elapsed = time.time() - start
    
    print(f"Mean calculation on 100k elements: {elapsed:.6f}s")
    assert elapsed < 1.0, "Mean calculation too slow"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
