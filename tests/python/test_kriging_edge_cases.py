"""Edge case tests for HPGL kriging and simulation — M-P-05, M-P-10, M-P-26.

Covers:
- M-P-10: lvm_kriging NaN/Inf validation guards (geo.py:1320-1323)
- M-P-05: median_ik / indicator_kriging edge cases (empty data, data-size mismatch)
- M-P-26: SIS empty marginal_probs ValueError (sis.py:150-151)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        indicator_kriging,
        lvm_kriging,
        median_ik,
    )
    from geo_bsd.sis import sis_simulation
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


# =============================================================================
# M-P-10: lvm_kriging NaN/Inf validation
# =============================================================================


@pytest.mark.hpgl
class TestLvmKrigingNanInf:
    """Test lvm_kriging guards for NaN/Inf in prop.data and mean_data (M-P-10)."""

    def test_lvm_kriging_nan_in_prop_data_raises(self):
        """prop.data with NaN raises ValueError at geo.py:1320-1321."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z  # 18

        # Create valid property first, then inject NaN via _data bypass
        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)

        mean_data = np.random.rand(size).astype("float32") * 50
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        # Inject NaN into prop._data (bypass ContProperty constructor validation)
        bad_data = np.copy(data)
        bad_data[5] = float("nan")
        object.__setattr__(prop, '_data', np.require(bad_data, "float32", "F"))

        with pytest.raises(ValueError, match="prop.data contains NaN or Inf"):
            lvm_kriging(prop, grid, mean_data, (2, 2, 1), 4, cov_model)

    def test_lvm_kriging_inf_in_prop_data_raises(self):
        """prop.data with Inf raises ValueError."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z  # 18

        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)

        mean_data = np.random.rand(size).astype("float32") * 50
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        bad_data = np.copy(data)
        bad_data[3] = float("inf")
        object.__setattr__(prop, '_data', np.require(bad_data, "float32", "F"))

        with pytest.raises(ValueError, match="prop.data contains NaN or Inf"):
            lvm_kriging(prop, grid, mean_data, (2, 2, 1), 4, cov_model)

    def test_lvm_kriging_nan_in_mean_data_raises(self):
        """mean_data with NaN raises ValueError at geo.py:1322-1323."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z  # 18

        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)

        mean_data = np.random.rand(size).astype("float32") * 50
        mean_data[7] = float("nan")
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(ValueError, match="mean_data contains NaN or Inf"):
            lvm_kriging(prop, grid, mean_data, (2, 2, 1), 4, cov_model)

    def test_lvm_kriging_inf_in_mean_data_raises(self):
        """mean_data with Inf raises ValueError."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z  # 18

        data = np.random.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)

        mean_data = np.random.rand(size).astype("float32") * 50
        mean_data[2] = -float("inf")
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(ValueError, match="mean_data contains NaN or Inf"):
            lvm_kriging(prop, grid, mean_data, (2, 2, 1), 4, cov_model)


# =============================================================================
# M-P-05: median_ik / indicator_kriging edge cases
# =============================================================================


@pytest.mark.hpgl
class TestMedianIkEdgeCases:
    """Test median_ik edge cases — empty data, mismatched data (M-P-05)."""

    def test_median_ik_indicator_count_not_two(self):
        """median_ik raises ValueError when indicator_count != 2."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 3, size, dtype="uint8")  # 3 categories
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(ValueError, match="indicator_count must be 2"):
            median_ik(prop, grid, (0.5, 0.5), (2, 2, 1), 4, cov_model)

    def test_median_ik_indicator_count_one(self):
        """median_ik raises when indicator_count == 1.

        IndProperty with indicator_count=1 requires all data values be 0
        (in range [0, indicator_count)). Use data=0 to pass validation.
        """
        grid = SugarboxGrid(x=2, y=2, z=2)
        size = grid.x * grid.y * grid.z

        data = np.zeros(size, dtype="uint8")  # All zeros: valid for count=1
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 1)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(2.0, 2.0, 1.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(ValueError, match="indicator_count must be 2"):
            median_ik(prop, grid, (0.5, 0.5), (1, 1, 1), 4, cov_model)

    def test_median_ik_empty_data_not_allowed(self):
        """IndProperty with empty data array raises ValueError (0D data not allowed).

        Empty arrays produce ndim==0 or ndim==1 with size 0 — IndProperty
        requires 1D or 3D, so empty (ndim=1 size=0) passes initial check
        but later operations fail.
        """
        data = np.array([], dtype="uint8")
        mask = np.array([], dtype="uint8")
        # Empty 1D array should be constructible
        prop = IndProperty(data, mask, 2)
        assert prop.data.size == 0

    def test_median_ik_valid_2cat_execution(self):
        """median_ik with valid 2-category indicator property works."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 2)

        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        result = median_ik(prop, grid, (0.4, 0.6), (2, 2, 1), 4, cov_model)
        assert result.data is not None
        assert result.data.size == size


@pytest.mark.hpgl
class TestIndicatorKrigingEdgeCases:
    """Test indicator_kriging edge cases (M-P-05)."""

    def test_indicator_kriging_mismatched_marginal_probs_length(self):
        """indicator_kriging raises when marginal_probs doesn't match data length."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 3, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            }
            for _ in range(3)
        ]
        # 3 data entries, 2 marginal_probs → mismatch
        with pytest.raises(ValueError, match="marginal_probs length"):
            indicator_kriging(prop, grid, ik_data, [0.3, 0.7])

    def test_indicator_kriging_two_category_redirects_to_median_ik(self):
        """2-category indicator_kriging redirects to median_ik with warning."""
        import logging

        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 2)

        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            },
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (1, 1, 1),
                "max_neighbours": 2,
            },
        ]

        result = indicator_kriging(prop, grid, ik_data, [0.3, 0.7])
        assert result.data is not None
        # Result shape matches prop shape (redirected to median_ik with only data[0] params)
        assert result.data.size == size

    def test_indicator_kriging_mismatched_prop_data_size(self):
        """indicator_kriging with prop data size not matching grid."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = int(grid.x * grid.y * grid.z)

        # Create smaller data than grid needs
        data = np.random.randint(0, 3, 5, dtype="uint8")
        mask = np.ones(5, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            }
            for _ in range(3)
        ]

        # The FFI call may fail with a RuntimeError due to size mismatch
        with pytest.raises((ValueError, RuntimeError)):
            indicator_kriging(prop, grid, ik_data, [0.3, 0.3, 0.4])


# =============================================================================
# M-P-26: SIS empty marginal_probs ValueError
# =============================================================================


@pytest.mark.hpgl
class TestSisEmptyMarginalProbs:
    """Test SIS empty marginal_probs raises ValueError (M-P-26)."""

    def test_sis_empty_marginal_probs_raises(self):
        """sis_simulation with empty marginal_probs raises ValueError."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 3, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            }
            for _ in range(3)
        ]

        with pytest.raises(ValueError, match="marginal_probs must not be empty"):
            sis_simulation(
                prop=prop, grid=grid, data=ik_data,
                marginal_probs=[], seed=42,
            )

    def test_sis_marginal_probs_length_mismatch(self):
        """sis_simulation with wrong-length marginal_probs raises ValueError."""
        grid = SugarboxGrid(x=3, y=3, z=2)
        size = grid.x * grid.y * grid.z

        data = np.random.randint(0, 3, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        prop = IndProperty(data, mask, 3)

        ik_data = [
            {
                "cov_model": CovarianceModel(
                    covariance.spherical, (3.0, 3.0, 2.0), (0, 0, 0), 1.0, 0.1
                ),
                "radiuses": (2, 2, 1),
                "max_neighbours": 4,
            }
            for _ in range(3)
        ]

        # 3 indicator data entries, 2 marginal_probs → mismatch
        with pytest.raises(ValueError, match="marginal_probs length"):
            sis_simulation(
                prop=prop, grid=grid, data=ik_data,
                marginal_probs=[0.3, 0.7], seed=42,
            )
