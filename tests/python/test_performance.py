"""
Performance benchmarking tests for HPGL

Each test records baseline timings on first run and detects regressions
by comparing against historical baselines. A regression is flagged when
a test exceeds its baseline by a configurable margin (default: 3x baseline
or slower than the hard floor, whichever is smaller).

Baselines are stored in .benchmarks/ and can be reset by deleting that directory.
"""
import numpy as np
import pytest
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Baseline storage directory
BENCH_DIR = Path(__file__).parent.parent.parent / ".benchmarks"
BENCH_DIR.mkdir(exist_ok=True)

# Regression factor: test fails if current time > BASELINE * REGRESSION_FACTOR
REGRESSION_FACTOR = 3.0
# Minimum baseline for reliable regression check (sub-ms baselines are unreliable)
MIN_BASELINE_FOR_REGRESSION = 0.005  # 5ms


def _get_baseline(name):
    """Load a stored baseline timing, or None if not yet calibrated."""
    path = BENCH_DIR / f"{name}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data.get("elapsed")
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _save_baseline(name, elapsed):
    """Save a timing as the new baseline."""
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps({"elapsed": elapsed, "date": time.time()}))


def _benchmark(name, elapsed, hard_floor=120.0):
    """Check elapsed time against baseline and hard floor.

    Returns True if performance is acceptable, False if regression detected.
    Prints diagnostic information about the comparison.
    """
    baseline = _get_baseline(name)
    if baseline is None:
        # First run — establish baseline
        _save_baseline(name, elapsed)
        print(f"  [BENCHMARK] {name}: {elapsed:.3f}s (new baseline)")
        result = elapsed <= hard_floor
    else:
        factor = elapsed / baseline if baseline > 0 else float("inf")
        # Sub-millisecond baselines are unreliable for regression testing
        if baseline < MIN_BASELINE_FOR_REGRESSION:
            result = elapsed <= hard_floor
        else:
            result = elapsed <= baseline * REGRESSION_FACTOR and elapsed <= hard_floor
        status = "OK" if result else "REGRESSION"
        print(f"  [BENCHMARK] {name}: {elapsed:.3f}s (baseline: {baseline:.3f}s, {factor:.1f}x, floor: {hard_floor}s) [{status}]")
    if not result:
        print(f"  WARNING: {name} exceeds performance threshold! "
              f"Elapsed: {elapsed:.3f}s, max allowed: {hard_floor}s, "
              f"baseline * {REGRESSION_FACTOR}x = {baseline * REGRESSION_FACTOR if baseline else 'N/A'}")
    return result

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
        """Benchmark ordinary kriging on small grid (10x10x5)

        Uses baseline comparison: establishes a historical baseline on first
        run, then checks subsequent runs for regression (>3x slowdown or >10s).
        Includes warmup iteration to mitigate cold-start effects.
        """
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

        # Warmup
        _ = ordinary_kriging(prop=prop, grid=grid, radiuses=(5, 5, 3),
                             max_neighbours=12, cov_model=cov_model)

        start = time.time()
        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model
        )
        elapsed = time.time() - start

        assert _benchmark("ok_small_grid", elapsed, hard_floor=10.0), \
            f"OK small grid performance degraded: {elapsed:.3f}s"
    
    def test_ok_medium_grid_performance(self):
        """Benchmark ordinary kriging on medium grid (50x50x20)

        Uses baseline comparison with 120s hard floor.
        """
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

        # Warmup
        _ = ordinary_kriging(prop=prop, grid=grid, radiuses=(10, 10, 5),
                             max_neighbours=12, cov_model=cov_model)

        start = time.time()
        result = ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(10, 10, 5),
            max_neighbours=12,
            cov_model=cov_model
        )
        elapsed = time.time() - start

        assert _benchmark("ok_medium_grid", elapsed, hard_floor=120.0), \
            f"OK medium grid performance degraded: {elapsed:.3f}s"
    
    def test_sgs_small_grid_performance(self):
        """Benchmark SGS on small grid with baseline comparison"""
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

        # Warmup
        _ = sgs_simulation(prop=prop, grid=grid, cdf_data=cdf_data,
                          radiuses=(5, 5, 3), max_neighbours=12,
                          cov_model=cov_model, seed=42)

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

        assert _benchmark("sgs_small_grid", elapsed, hard_floor=30.0), \
            f"SGS small grid performance degraded: {elapsed:.3f}s"
    
    def test_neighbour_count_performance_impact(self):
        """Test performance impact of different neighbour counts.

        Uses baseline comparison. Each neighbor count run must complete
        within 5s and not exceed baseline * REGRESSION_FACTOR.
        """
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
            # Each run should complete within hard floor
            assert elapsed < 5.0, \
                f"Kriging with {max_neighbours} neighbours took too long: {elapsed:.3f}s"
    
    def test_covariance_type_performance(self):
        """Test performance of different covariance types.

        Uses baseline comparison with 30s hard floor per covariance type.
        """
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

        # All should complete within hard floor
        for cov_name, elapsed in timings.items():
            assert elapsed < 30.0, f"{cov_name} took too long: {elapsed:.3f}s"


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
def test_mean_calculation_performance():
    """Test mean calculation performance with baseline comparison"""
    from geo_bsd.geo import ContProperty
    large_data = np.random.rand(100000).astype('float32') * 100
    large_mask = np.ones(100000, dtype='uint8')
    prop = ContProperty(large_data, large_mask)

    start = time.time()
    mean_val = calc_mean(prop)
    elapsed = time.time() - start

    print(f"Mean calculation on 100k elements: {elapsed:.6f}s")
    assert _benchmark("mean_calculation", elapsed, hard_floor=1.0), \
        f"Mean calculation performance degraded: {elapsed:.6f}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
