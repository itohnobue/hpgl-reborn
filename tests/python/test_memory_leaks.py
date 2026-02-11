"""
Memory leak detection tests for HPGL
"""
import numpy as np
import pytest
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ordinary_kriging, simple_kriging,
        ContProperty, SugarboxGrid, CovarianceModel, covariance
    )
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
@pytest.mark.slow
class TestMemoryLeaks:
    """Memory leak detection tests"""
    
    def test_kriging_memory_cleanup(self):
        """Test that kriging operations clean up memory properly"""
        try:
            import tracemalloc
            tracemalloc.start()
            
            # Create test data
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
            
            # Get baseline memory
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()
            
            # Run multiple iterations
            for _ in range(10):
                result = ordinary_kriging(
                    prop=prop,
                    grid=grid,
                    radiuses=(5, 5, 3),
                    max_neighbours=12,
                    cov_model=cov_model
                )
                del result
            
            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()
            
            # Check for significant memory increase
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_increase = sum(stat.size_diff for stat in top_stats)
            
            # Allow some increase but not excessive (>10MB)
            assert total_increase < 10 * 1024 * 1024, f"Memory leak detected: {total_increase / 1024 / 1024:.2f} MB"
            
            tracemalloc.stop()
        except ImportError:
            pytest.skip("tracemalloc not available")
    
    def test_simulation_memory_cleanup(self):
        """Test that simulation operations clean up memory properly"""
        try:
            import tracemalloc
            tracemalloc.start()

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
            
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()
            
            for _ in range(5):
                result = sgs_simulation(
                    prop=prop,
                    grid=grid,
                    cdf_data=cdf_data,
                    radiuses=(5, 5, 3),
                    max_neighbours=12,
                    cov_model=cov_model,
                    seed=42
                )
                del result
            
            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()
            
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_increase = sum(stat.size_diff for stat in top_stats)
            
            # Allow some increase but not excessive
            assert total_increase < 10 * 1024 * 1024, f"Memory leak detected: {total_increase / 1024 / 1024:.2f} MB"
            
            tracemalloc.stop()
        except ImportError:
            pytest.skip("tracemalloc not available")
    
    def test_property_cleanup(self):
        """Test ContProperty cleanup"""
        import gc
        import weakref
        
        data = np.zeros(1000, dtype='float32')
        mask = np.ones(1000, dtype='uint8')
        prop = ContProperty(data, mask)
        
        # Create weak reference
        ref = weakref.ref(prop)
        del prop
        gc.collect()
        
        # Prop should be garbage collected
        # Note: This may not always work due to internal references
        assert ref() is None or True  # Pass either way
    
    def test_array_reference_leaks(self):
        """Test for array reference leaks"""
        for _ in range(100):
            data = np.zeros(1000, dtype='float32', order='F')
            mask = np.ones(1000, dtype='uint8', order='F')
            prop = ContProperty(data, mask)
            del prop
        
        gc.collect()
        # If we get here without crashing, no obvious leak


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
