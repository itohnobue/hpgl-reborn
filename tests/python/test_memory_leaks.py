"""
Memory leak detection tests for HPGL

IMPORTANT: These tests use Python's tracemalloc module which detects
Python-level memory leaks only. C++ level leaks (new/delete mismatches,
unfreed arrays in shared library code) are NOT visible to tracemalloc.

To detect C++ memory leaks, run with:
    - Valgrind (Linux): valgrind --leak-check=full python -m pytest tests/python/test_memory_leaks.py
    - AddressSanitizer (Linux/macOS): compile HPGL with -fsanitize=address
    - Dr. Memory (Windows)

Each gate below derives its budget from the fixture it measures instead of
using a fixed 10MB: the budget is three results' worth of the loop's own
payload, so a loop that retained EVERY result fails the gate while allocator
residue still passes (measured on an untouched run: 2-4 KB of residue against
a 200 KB / 18 KB / 536 KB total-leak volume for the kriging, SGS and array
fixtures respectively). A fixed 10MB threshold was 50-1000x above those
volumes, i.e. a 100% leak of every iteration could not fail it.
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
            data = np.random.rand(4000).astype("float32") * 100
            mask = np.ones(4000, dtype="uint8")
            prop = ContProperty(data, mask)
            cov_model = CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1
            )

            # Get baseline memory
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()

            # Run multiple iterations
            for _ in range(10):
                result = ordinary_kriging(
                    prop=prop, grid=grid, radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov_model
                )
                del result

            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()

            # Check for significant memory increase
            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_increase = sum(stat.size_diff for stat in top_stats)

            # F-553: budget derived from the fixture, not a magic constant.
            # One result is 4,000 float32 cells (16,000 B) + a 4,000-cell
            # uint8 mask (4,000 B) = 20,000 B, so retaining all 10 results
            # leaks ~200 KB. Allow 3 results' worth (60,000 B): a full leak
            # fails, while the measured untouched-run residue is ~3.5 KB.
            leak_budget = 3 * 4000 * (4 + 1)
            assert total_increase < leak_budget, (
                f"Memory leak detected: {total_increase} B retained; budget is "
                f"{leak_budget} B (3 kriging results' worth)"
            )

            tracemalloc.stop()
        except ImportError:
            pytest.skip("tracemalloc not available")

    def test_simulation_memory_cleanup(self):
        """Test that simulation operations clean up memory properly"""
        try:
            import tracemalloc

            tracemalloc.start()

            grid = SugarboxGrid(x=10, y=10, z=5)
            data = np.random.rand(500).astype("float32") * 100
            mask = np.ones(500, dtype="uint8")
            prop = ContProperty(data, mask)
            cov_model = CovarianceModel(
                type=covariance.spherical, ranges=(5.0, 5.0, 3.0), sill=1.0, nugget=0.1
            )
            cdf_data = CdfData(
                np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype="float32"),
                np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32"),
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
                    seed=42,
                )
                del result

            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()

            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_increase = sum(stat.size_diff for stat in top_stats)

            # F-553: 500-cell result = 2,000 B data + 500 B mask = 2,500 B;
            # all 5 results would leak ~18 KB. Budget 3 results' worth
            # (7,500 B) vs ~2.4 KB measured residue on an untouched run.
            leak_budget = 3 * 500 * (4 + 1)
            assert total_increase < leak_budget, (
                f"Memory leak detected: {total_increase} B retained; budget is "
                f"{leak_budget} B (3 simulation results' worth)"
            )

            tracemalloc.stop()
        except ImportError:
            pytest.skip("tracemalloc not available")

    def test_array_reference_leaks(self):
        """Test for array reference leaks"""
        try:
            import tracemalloc

            tracemalloc.start()
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()

            for _ in range(100):
                data = np.zeros(1000, dtype="float32", order="F")
                mask = np.ones(1000, dtype="uint8", order="F")
                prop = ContProperty(data, mask)
                del prop

            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()

            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_increase = sum(stat.size_diff for stat in top_stats)

            # F-553: 100 iterations x (1,000 float32 + 1,000-cell uint8) =
            # 100 x 5,000 B = ~536 KB if every property were retained. Budget
            # 3 results' worth (15,000 B) vs ~2.3 KB measured residue.
            leak_budget = 3 * 1000 * (4 + 1)
            assert total_increase < leak_budget, (
                f"Array reference leak detected: {total_increase} B retained; "
                f"budget is {leak_budget} B (3 properties' worth)"
            )
            tracemalloc.stop()
        except ImportError:
            pytest.skip("tracemalloc not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
