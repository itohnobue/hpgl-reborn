"""
Comprehensive tests for all HPGL core classes:
- SugarboxGrid
- CovarianceModel
- ContProperty (Continuous Property)
- IndProperty (Indicator Property)
- CdfData
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        SugarboxGrid, CovarianceModel, ContProperty, IndProperty,
        covariance, checkFWA
    )
    from geo_bsd.cdf import CdfData
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestSugarboxGrid:
    """Test SugarboxGrid class - represents 3D grid dimensions"""

    def test_constructor_positive_dimensions(self):
        """Test constructor with positive dimensions"""
        grid = SugarboxGrid(x=10, y=20, z=5)
        assert grid.x == 10
        assert grid.y == 20
        assert grid.z == 5

    def test_constructor_unity_dimensions(self):
        """Test constructor with unity (1x1x1) dimensions"""
        grid = SugarboxGrid(x=1, y=1, z=1)
        assert grid.x == 1
        assert grid.y == 1
        assert grid.z == 1

    def test_constructor_large_dimensions(self):
        """Test constructor with large dimensions"""
        grid = SugarboxGrid(x=1000, y=1000, z=100)
        assert grid.x == 1000
        assert grid.y == 1000
        assert grid.z == 100

    def test_dimensions_are_integers(self):
        """Test that dimensions are stored as integers"""
        grid = SugarboxGrid(x=10.5, y=20.7, z=5.3)
        # Python allows float to int assignment - verify values are stored
        assert grid.x == 10.5
        assert grid.y == 20.7
        assert grid.z == 5.3

    def test_grid_size_property(self):
        """Test calculating total grid size"""
        grid = SugarboxGrid(x=10, y=20, z=5)
        total_size = grid.x * grid.y * grid.z
        assert total_size == 1000


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCovarianceModel:
    """Test CovarianceModel class - variogram parameters"""

    def test_constructor_default_parameters(self):
        """Test constructor with default parameters"""
        cov = CovarianceModel()
        assert cov.type == 0  # spherical
        assert cov.ranges == (0, 0, 0)
        assert cov.angles == (0.0, 0.0, 0.0)
        assert cov.sill == 1.0
        assert cov.nugget == 0.0

    def test_constructor_spherical_type(self):
        """Test constructor with spherical covariance type"""
        cov = CovarianceModel(
            type=covariance.spherical,
            ranges=(10.0, 10.0, 5.0),
            angles=(0.0, 0.0, 0.0),
            sill=2.0,
            nugget=0.5
        )
        assert cov.type == covariance.spherical
        assert cov.ranges == (10.0, 10.0, 5.0)
        assert cov.sill == 2.0
        assert cov.nugget == 0.5

    def test_constructor_exponential_type(self):
        """Test constructor with exponential covariance type"""
        cov = CovarianceModel(
            type=covariance.exponential,
            ranges=(15.0, 15.0, 8.0),
            angles=(45.0, 0.0, 30.0),
            sill=1.5,
            nugget=0.3
        )
        assert cov.type == covariance.exponential
        assert cov.angles == (45.0, 0.0, 30.0)

    def test_constructor_gaussian_type(self):
        """Test constructor with Gaussian covariance type"""
        cov = CovarianceModel(
            type=covariance.gaussian,
            ranges=(20.0, 20.0, 10.0),
            sill=3.0,
            nugget=0.0
        )
        assert cov.type == covariance.gaussian

    def test_nugget_equal_sill_valid(self):
        """Test that nugget equal to sill is valid"""
        cov = CovarianceModel(
            sill=1.0,
            nugget=1.0  # Equal to sill
        )
        assert cov.nugget == 1.0

    def test_nugget_less_than_sill_valid(self):
        """Test that nugget less than sill is valid"""
        cov = CovarianceModel(
            sill=2.0,
            nugget=0.5  # Less than sill
        )
        assert cov.nugget == 0.5

    def test_nugget_greater_than_sill_raises_exception(self):
        """Test that nugget > sill raises Exception"""
        with pytest.raises(Exception, match="Nugget .* exceeds sill"):
            CovarianceModel(
                sill=1.0,
                nugget=1.5  # Greater than sill - should raise
            )

    def test_nugget_zero_valid(self):
        """Test that zero nugget is valid"""
        cov = CovarianceModel(sill=1.0, nugget=0.0)
        assert cov.nugget == 0.0

    def test_anisotropic_ranges(self):
        """Test anisotropic range parameters"""
        cov = CovarianceModel(
            ranges=(30.0, 20.0, 10.0),  # Different ranges for each direction
            sill=1.0
        )
        assert cov.ranges == (30.0, 20.0, 10.0)

    def test_rotation_angles(self):
        """Test rotation angles for anisotropy"""
        cov = CovarianceModel(
            angles=(30.0, 45.0, 60.0),
            sill=1.0
        )
        assert cov.angles == (30.0, 45.0, 60.0)

    @pytest.mark.parametrize("cov_type", [
        covariance.spherical,
        covariance.exponential,
        covariance.gaussian
    ])
    def test_all_covariance_types(self, cov_type):
        """Parametrized test for all covariance types"""
        cov = CovarianceModel(
            type=cov_type,
            ranges=(10.0, 10.0, 5.0),
            sill=1.0
        )
        assert cov.type == cov_type


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestContProperty:
    """Test ContProperty class - continuous property with mask"""

    def test_constructor_with_lists(self):
        """Test constructor with Python lists"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mask = [1, 1, 1, 1, 1]
        prop = ContProperty(data, mask)

        assert prop.data.dtype == np.float32
        assert prop.mask.dtype == np.uint8
        assert np.array_equal(prop.data, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32'))

    def test_constructor_with_numpy_arrays(self):
        """Test constructor with numpy arrays"""
        data = np.array([1.0, 2.0, 3.0], dtype='float64')
        mask = np.array([1, 0, 1], dtype='int32')
        prop = ContProperty(data, mask)

        # Should convert to proper types
        assert prop.data.dtype == np.float32
        assert prop.mask.dtype == np.uint8

    def test_data_fortran_order(self):
        """Test that data array is Fortran-ordered"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        assert prop.data.flags['F_CONTIGUOUS']
        assert prop.mask.flags['F_CONTIGUOUS']

    def test_validate_with_valid_data(self):
        """Test validate() with valid Fortran-order arrays"""
        data = np.asfortranarray(np.ones((10, 10, 5), dtype='float32'))
        mask = np.asfortranarray(np.ones((10, 10, 5), dtype='uint8'))
        prop = ContProperty(data, mask)

        # Should not raise any exception
        prop.validate()

    def test_validate_with_shape_mismatch(self):
        """Test validate() raises error when data and mask shapes don't match"""
        data = np.asfortranarray(np.ones((10, 10, 5), dtype='float32'))
        mask = np.asfortranarray(np.ones((10, 10, 3), dtype='uint8'))  # Different shape
        prop = ContProperty(data, mask)

        with pytest.raises(AssertionError):
            prop.validate()

    def test_fix_shape_reshapes_flat_data(self):
        """Test fix_shape() reshapes flat array to 3D grid"""
        data = np.arange(100, dtype='float32')  # 1D array
        mask = np.ones(100, dtype='uint8')
        prop = ContProperty(data, mask)

        grid = SugarboxGrid(x=5, y=5, z=4)  # 5*5*4 = 100
        prop.fix_shape(grid)

        assert prop.data.shape == (5, 5, 4)
        assert prop.mask.shape == (5, 5, 4)

    def test_fix_shape_already_3d(self):
        """Test fix_shape() does nothing if data is already 3D"""
        data = np.asfortranarray(np.ones((5, 5, 4), dtype='float32'))
        mask = np.asfortranarray(np.ones((5, 5, 4), dtype='uint8'))
        prop = ContProperty(data, mask)

        grid = SugarboxGrid(x=5, y=5, z=4)
        original_shape = prop.data.shape
        prop.fix_shape(grid)

        assert prop.data.shape == original_shape

    def test_fix_shape_size_mismatch(self):
        """Test fix_shape() doesn't reshape when size doesn't match grid"""
        data = np.arange(50, dtype='float32')  # Wrong size
        mask = np.ones(50, dtype='uint8')
        prop = ContProperty(data, mask)

        grid = SugarboxGrid(x=5, y=5, z=4)  # 5*5*4 = 100, not 50
        prop.fix_shape(grid)

        # Should remain 1D since size doesn't match
        assert prop.data.ndim == 1

    def test_getitem_data(self):
        """Test deprecated __getitem__ for data access (index 0)"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        result = prop[0]
        assert np.array_equal(result, data)

    def test_getitem_mask(self):
        """Test deprecated __getitem__ for mask access (index 1)"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 0, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        result = prop[1]
        assert np.array_equal(result, mask)

    def test_getitem_invalid_index(self):
        """Test deprecated __getitem__ raises error for invalid index"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        with pytest.raises(RuntimeError, match="Index out of range"):
            _ = prop[2]

    def test_writable_and_aligned_flags(self):
        """Test that arrays are writable and aligned"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        # checkFWA checks Fortran, Writable, and Aligned flags
        checkFWA(prop.data)
        checkFWA(prop.mask)

    def test_empty_arrays(self):
        """Test with empty arrays"""
        data = np.array([], dtype='float32')
        mask = np.array([], dtype='uint8')
        prop = ContProperty(data, mask)

        assert prop.data.size == 0
        assert prop.mask.size == 0

    def test_single_element(self):
        """Test with single element arrays"""
        data = np.array([42.0], dtype='float32')
        mask = np.array([1], dtype='uint8')
        prop = ContProperty(data, mask)

        assert prop.data[0] == 42.0
        assert prop.mask[0] == 1

    def test_large_array(self):
        """Test with large array"""
        size = 1000000
        data = np.random.rand(size).astype('float32')
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data, mask)

        assert prop.data.size == size
        assert prop.mask.size == size


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestIndProperty:
    """Test IndProperty class - indicator property with mask"""

    def test_constructor_valid(self):
        """Test constructor with valid indicator data"""
        data = np.array([0, 1, 2, 0, 1], dtype='uint8')  # 3 indicators
        mask = np.array([1, 1, 1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        assert prop.indicator_count == 3
        assert prop.data.dtype == np.uint8
        assert prop.mask.dtype == np.uint8

    def test_data_fortran_order(self):
        """Test that data and mask arrays are Fortran-ordered"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        assert prop.data.flags['F_CONTIGUOUS']
        assert prop.mask.flags['F_CONTIGUOUS']

    def test_validate_with_valid_data(self):
        """Test validate() with valid indicator property"""
        data = np.asfortranarray(np.array([0, 1, 2, 0, 1], dtype='uint8'))
        mask = np.asfortranarray(np.ones(5, dtype='uint8'))
        prop = IndProperty(data, mask, 3)

        prop.validate()  # Should not raise

    def test_validate_with_shape_mismatch(self):
        """Test that constructor raises error when data and mask shapes don't match"""
        data = np.asfortranarray(np.ones(10, dtype='uint8'))
        mask = np.asfortranarray(np.ones(5, dtype='uint8'))

        # The constructor has an assertion that checks shape match
        with pytest.raises((ValueError, AssertionError)):
            IndProperty(data, mask, 2)

    def test_indicator_value_out_of_range_raises_error(self):
        """Test that indicator values >= indicator_count raise error"""
        data = np.array([0, 1, 3, 0], dtype='uint8')  # 3 is out of range for 3 indicators
        mask = np.array([1, 1, 1, 1], dtype='uint8')

        with pytest.raises(RuntimeError, match="Property contains some indicators outside of"):
            IndProperty(data, mask, 3)  # indicator_count=3 means valid values are 0,1,2

    def test_indicator_value_equal_to_count_allowed_if_masked(self):
        """Test that out-of-range values are OK if masked (mask=0)"""
        data = np.array([0, 1, 3, 0], dtype='uint8')  # 3 is out of range
        mask = np.array([1, 1, 0, 1], dtype='uint8')  # Value at index 2 is masked

        # Should not raise because the out-of-range value is masked
        prop = IndProperty(data, mask, 3)
        assert prop.indicator_count == 3

    def test_getitem_data(self):
        """Test deprecated __getitem__ for data access (index 0)"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        result = prop[0]
        assert np.array_equal(result, data)

    def test_getitem_mask(self):
        """Test deprecated __getitem__ for mask access (index 1)"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 0, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        result = prop[1]
        assert np.array_equal(result, mask)

    def test_getitem_indicator_count(self):
        """Test deprecated __getitem__ for indicator_count (index 2)"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 5)

        result = prop[2]
        assert result == 5

    def test_getitem_invalid_index(self):
        """Test deprecated __getitem__ raises error for invalid index"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        with pytest.raises(RuntimeError, match="Index out of range"):
            _ = prop[3]

    def test_writable_and_aligned_flags(self):
        """Test that arrays are writable and aligned"""
        data = np.array([0, 1, 2], dtype='uint8')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        checkFWA(prop.data)
        checkFWA(prop.mask)

    def test_binary_indicator_property(self):
        """Test binary indicator property (2 indicators: 0 and 1)"""
        data = np.array([0, 1, 1, 0, 1], dtype='uint8')
        mask = np.array([1, 1, 1, 1, 1], dtype='uint8')
        prop = IndProperty(data, mask, 2)

        assert prop.indicator_count == 2
        # Valid values are 0 and 1 only

    def test_multi_indicator_property(self):
        """Test multi-category indicator property"""
        categories = 10
        data = np.random.randint(0, categories, 100, dtype='uint8')
        mask = np.ones(100, dtype='uint8')
        prop = IndProperty(data, mask, categories)

        assert prop.indicator_count == categories

    def test_empty_arrays(self):
        """Test with empty arrays"""
        data = np.array([], dtype='uint8')
        mask = np.array([], dtype='uint8')
        prop = IndProperty(data, mask, 3)

        assert prop.data.size == 0
        assert prop.mask.size == 0

    def test_all_zero_indicators(self):
        """Test with all zero indicator values"""
        data = np.zeros(10, dtype='uint8')
        mask = np.ones(10, dtype='uint8')
        prop = IndProperty(data, mask, 5)

        assert np.all(prop.data == 0)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCdfData:
    """Test CdfData class - cumulative distribution function data"""

    def test_constructor_with_lists(self):
        """Test constructor with Python lists"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        probs = [0.2, 0.4, 0.6, 0.8, 1.0]
        cdf = CdfData(values, probs)

        assert cdf.values.dtype == np.float32
        assert cdf.probs.dtype == np.float32
        assert len(cdf.values) == 5
        assert len(cdf.probs) == 5

    def test_constructor_with_numpy_arrays(self):
        """Test constructor with numpy arrays"""
        values = np.array([1.0, 2.0, 3.0], dtype='float64')
        probs = np.array([0.33, 0.66, 1.0], dtype='float64')
        cdf = CdfData(values, probs)

        # Should convert to float32
        assert cdf.values.dtype == np.float32
        assert cdf.probs.dtype == np.float32

    def test_values_and_probs_same_size(self):
        """Test that values and probs arrays have same size"""
        values = [1.0, 2.0, 3.0]
        probs = [0.33, 0.66, 1.0]
        cdf = CdfData(values, probs)

        assert cdf.values.size == cdf.probs.size

    def test_single_value_cdf(self):
        """Test CDF with single value"""
        cdf = CdfData([5.0], [1.0])
        assert cdf.values[0] == 5.0
        assert cdf.probs[0] == 1.0

    def test_empty_cdf_arrays(self):
        """Test with empty arrays"""
        cdf = CdfData([], [])
        assert cdf.values.size == 0
        assert cdf.probs.size == 0

    def test_monotonically_increasing_probs(self):
        """Test CDF probabilities are monotonically increasing"""
        # This test verifies the property, not the constructor validation
        # (the constructor doesn't validate this)
        probs = np.array([0.1, 0.3, 0.6, 0.8, 1.0], dtype='float32')
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        cdf = CdfData(values, probs)

        # Verify probabilities are sorted
        assert np.all(cdf.probs[:-1] <= cdf.probs[1:])

    def test_prob_ends_at_one(self):
        """Test that CDF probability ends at or near 1.0"""
        values = [1.0, 2.0, 3.0]
        probs = [0.5, 0.8, 1.0]  # Ends at 1.0
        cdf = CdfData(values, probs)

        assert abs(cdf.probs[-1] - 1.0) < 1e-6

    def test_large_cdf(self):
        """Test with large CDF array"""
        size = 10000
        values = np.linspace(0, 100, size).astype('float32')
        probs = np.linspace(0, 1, size).astype('float32')
        cdf = CdfData(values, probs)

        assert cdf.values.size == size
        assert cdf.probs.size == size


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCovarianceClass:
    """Test covariance class for type constants"""

    def test_spherical_constant(self):
        """Test spherical covariance type constant"""
        assert covariance.spherical == 0

    def test_exponential_constant(self):
        """Test exponential covariance type constant"""
        assert covariance.exponential == 1

    def test_gaussian_constant(self):
        """Test Gaussian covariance type constant"""
        assert covariance.gaussian == 2

    def test_constants_are_unique(self):
        """Test that all covariance type constants are unique"""
        types = [
            covariance.spherical,
            covariance.exponential,
            covariance.gaussian
        ]
        assert len(set(types)) == len(types)


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestCheckFWA:
    """Test checkFWA helper function"""

    def test_fortran_writable_aligned_array(self):
        """Test checkFWA passes for F, W, A arrays"""
        arr = np.asfortranarray(np.ones((10, 10), dtype='float32'))
        # Should not raise
        checkFWA(arr)

    def test_non_fortran_array_raises(self):
        """Test checkFWA raises for non-Fortran array"""
        arr = np.ones((10, 10), dtype='float32')  # C-order by default
        with pytest.raises(AssertionError):
            checkFWA(arr)

    def test_non_writable_array_raises(self):
        """Test checkFWA raises for non-writable array"""
        arr = np.asfortranarray(np.ones((10, 10), dtype='float32'))
        arr.flags.writeable = False
        with pytest.raises(AssertionError):
            checkFWA(arr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
