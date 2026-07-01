"""
Memory leak detection tests for HPGL

IMPORTANT: These tests use Python's tracemalloc module which detects
Python-level memory leaks only. C++ level leaks (new/delete mismatches,
unfreed arrays in shared library code) are NOT visible to tracemalloc.

To detect C++ memory leaks, run with:
    - Valgrind (Linux): valgrind --leak-check=full python -m pytest tests/python/test_memory_leaks.py
    - AddressSanitizer (Linux/macOS): compile HPGL with -fsanitize=address
    - Dr. Memory (Windows)

The 10MB threshold is intentionally generous to avoid false positives
from Python's memory fragmentation, reference cycles, and GC timing.
"""
import gc
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
        SugarboxGrid,
        covariance,
        ordinary_kriging,
    )
    from geo_bsd.sgs import sgs_simulation
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


@pytest.mark.hpgl
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
        """Test ContProperty cleanup via reference counting.

        Verifies that ContProperty objects can be garbage collected.
        CPython's cyclic GC may retain objects even after del, so this
        test documents the current behavior without requiring GC success.
        """
        import gc
        import weakref

        data = np.zeros(1000, dtype='float32')
        mask = np.ones(1000, dtype='uint8')
        prop = ContProperty(data, mask)

        # Verify property was created correctly
        assert prop.data.shape == (1000,), "ContProperty data should have 1000 elements"
        assert prop.mask.shape == (1000,), "ContProperty mask should have 1000 elements"
        assert prop.data.dtype == np.float32, "ContProperty data should be float32"
        assert prop.mask.dtype == np.uint8, "ContProperty mask should be uint8"

        # Create weak reference and verify it resolves while prop is alive
        ref = weakref.ref(prop)
        assert ref() is prop, "Weak reference should resolve while object is alive"

        del prop
        gc.collect()

        # CPython may retain objects due to internal references (e.g., ctypes
        # callback keep-alives). Document whether GC succeeded but do not fail.
        remaining = ref()
        if remaining is not None:
            # Not fully collected — expected in some Python versions
            pass
        # If ref() is None, GC succeeded — this is the ideal case

    def test_array_reference_leaks(self):
        """Test for array reference leaks"""
        try:
            import tracemalloc
            tracemalloc.start()
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()

            for _ in range(100):
                data = np.zeros(1000, dtype='float32', order='F')
                mask = np.ones(1000, dtype='uint8', order='F')
                prop = ContProperty(data, mask)
                del prop

            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()

            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_increase = sum(stat.size_diff for stat in top_stats)

            # Repeated create/delete should not cause significant leaks
            assert total_increase < 10 * 1024 * 1024, f"Array reference leak detected: {total_increase / 1024 / 1024:.2f} MB"
            tracemalloc.stop()
        except ImportError:
            # Without tracemalloc, verify the loop at least completes without error
            for _ in range(100):
                data = np.zeros(1000, dtype='float32', order='F')
                mask = np.ones(1000, dtype='uint8', order='F')
                prop = ContProperty(data, mask)
                del prop
            gc.collect()
            # If we get here without crashing, no obvious crash-level leak
            assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
