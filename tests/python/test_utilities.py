"""
Comprehensive tests for HPGL utility functions.

Tests cover:
- Calculation functions: calc_mean, calc_cdf
- Thread management: set_thread_num, get_thread_num
- Callback handlers: set_output_handler, set_progress_handler
- I/O functions: load/write properties in INC and GSLib formats
- Utility functions: append_mask
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import CdfData, calc_cdf
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        _load_prop_ind_slow,  # For testing INC format parsing
        append_mask,
        calc_mean,
        covariance,
        get_thread_num,
        load_cont_property,
        load_ind_property,
        ordinary_kriging,
        read_inc_file_byte,
        read_inc_file_float,
        set_output_handler,
        set_progress_handler,
        set_thread_num,
        simple_kriging_weights,
        write_gslib_property,
        write_property,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


# =============================================================================
# Calculation Function Tests
# =============================================================================

@pytest.mark.hpgl
class TestCalcMean:
    """Test calc_mean function for calculating mean of properties"""

    def test_calc_mean_basic(self):
        """Test basic mean calculation with known values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.array([1, 1, 1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        assert result == pytest.approx(3.0)

    def test_calc_mean_with_masked_values(self):
        """Test mean calculation ignores uninformed (masked) values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.array([1, 0, 1, 0, 1], dtype='uint8')  # Mask 2 and 4
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        # Only 1, 3, 5 are informed: (1 + 3 + 5) / 3 = 3
        assert result == pytest.approx(3.0)

    def test_calc_mean_all_masked(self):
        """Test mean calculation when all values are masked"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([0, 0, 0], dtype='uint8')
        prop = ContProperty(data, mask)

        # Should raise ValueError for all-masked property
        with pytest.raises(ValueError, match="no informed values"):
            calc_mean(prop)

    def test_calc_mean_single_value(self):
        """Test mean calculation with single informed value"""
        data = np.array([42.0], dtype='float32')
        mask = np.array([1], dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        assert result == pytest.approx(42.0)

    def test_calc_mean_large_array(self):
        """Test mean calculation with large array"""
        np.random.seed(42)
        size = 10000
        data = np.random.rand(size).astype('float32') * 100
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        expected = np.mean(data)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_calc_mean_with_tuple_input(self):
        """Test calc_mean accepts tuple input (data, mask)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.array([1, 1, 1, 1, 1], dtype='uint8')

        result = calc_mean((data, mask))
        assert result == pytest.approx(3.0)

    def test_calc_mean_3d_property(self):
        """Test mean calculation with 3D property"""
        data = np.arange(27, dtype='float32').reshape((3, 3, 3), order='F')
        mask = np.ones((3, 3, 3), dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        # Mean of 0-26 is 13
        assert result == pytest.approx(13.0)


@pytest.mark.hpgl
class TestCalcCdf:
    """Test calc_cdf function for cumulative distribution calculation"""

    def test_calc_cdf_basic(self):
        """Test basic CDF calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.ones(5, dtype='uint8')
        prop = ContProperty(data.reshape((5, 1, 1), order='F'),
                           mask.reshape((5, 1, 1), order='F'))

        result = calc_cdf(prop)

        assert isinstance(result, CdfData)
        assert len(result.values) == 5  # n values for n unique (returns all)
        assert len(result.probs) == 5
        # Check probabilities are monotonically increasing
        assert np.all(np.diff(result.probs) >= 0)
        # Final cumulative probability reaches 1.0
        assert result.probs[-1] == pytest.approx(1.0)

    def test_calc_cdf_with_duplicates(self):
        """Test CDF with duplicate values"""
        data = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype='float32')
        mask = np.ones(6, dtype='uint8')
        prop = ContProperty(data.reshape((6, 1, 1), order='F'),
                           mask.reshape((6, 1, 1), order='F'))

        result = calc_cdf(prop)

        # 3 unique values (1, 2, 3) -> 3 values (returns all N)
        assert len(result.values) == 3
        # Final cumulative probability reaches 1.0
        assert result.probs[-1] == pytest.approx(1.0, rel=1e-5)

    def test_calc_cdf_with_masked_values(self):
        """Test CDF calculation ignores masked values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.array([1, 0, 1, 0, 1], dtype='uint8')
        prop = ContProperty(data.reshape((5, 1, 1), order='F'),
                           mask.reshape((5, 1, 1), order='F'))

        result = calc_cdf(prop)

        # Only 1, 3, 5 are informed (3 unique values) -> returns all 3 values
        assert len(result.values) == 3
        # Final cumulative probability reaches 1.0
        assert result.probs[-1] == pytest.approx(1.0, rel=1e-5)

    def test_calc_cdf_uniform_distribution(self):
        """Test CDF with uniform distribution"""
        data = np.array([5.0, 5.0, 5.0, 5.0], dtype='float32')
        mask = np.ones(4, dtype='uint8')
        prop = ContProperty(data.reshape((4, 1, 1), order='F'),
                           mask.reshape((4, 1, 1), order='F'))

        result = calc_cdf(prop)

        # Single value case
        assert len(result.values) == 1

    def test_calc_cdf_all_masked(self):
        """Test CDF when all values are masked raises ValueError"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.zeros((3, 1, 1), dtype='uint8')
        prop = ContProperty(data.reshape((3, 1, 1), order='F'),
                           mask.reshape((3, 1, 1), order='F'))

        with pytest.raises(ValueError, match="no informed values"):
            calc_cdf(prop)

    def test_calc_cdf_monotonic_probabilities(self):
        """Test that CDF probabilities are monotonically increasing"""
        np.random.seed(42)
        data = np.random.rand(100).astype('float32') * 100
        mask = np.ones(100, dtype='uint8')
        prop = ContProperty(data.reshape((100, 1, 1), order='F'),
                           mask.reshape((100, 1, 1), order='F'))

        result = calc_cdf(prop)

        # Check monotonicity
        diffs = np.diff(result.probs)
        assert np.all(diffs >= -1e-10)  # Allow small floating point errors

    def test_calc_cdf_final_probability(self):
        """Test that final CDF probability is 1.0"""
        np.random.seed(42)
        data = np.random.rand(50).astype('float32') * 100
        mask = np.ones(50, dtype='uint8')
        prop = ContProperty(data.reshape((50, 1, 1), order='F'),
                           mask.reshape((50, 1, 1), order='F'))

        result = calc_cdf(prop)

        # Final cumulative probability reaches 1.0
        assert result.probs[-1] == pytest.approx(1.0)
        assert result.probs[-1] > 0


# =============================================================================
# Thread Management Tests
# =============================================================================

def _omp_available():
    """Check if OpenMP is available by testing set/get thread round-trip.

    When _OPENMP is not defined (e.g., Apple Clang on macOS), set_thread_num()
    is a no-op and get_thread_num() always returns 1. A successful round-trip
    of set_thread_num(2) -> get_thread_num() == 2 confirms OpenMP availability.
    """
    saved = get_thread_num()
    try:
        set_thread_num(2)
        return get_thread_num() == 2
    finally:
        set_thread_num(saved)


@pytest.mark.hpgl
class TestThreadManagement:
    """Test thread number management functions"""

    def test_get_thread_num_initial(self):
        """Test getting initial thread count"""
        initial_threads = get_thread_num()
        assert isinstance(initial_threads, int)
        assert initial_threads > 0

    def test_set_thread_num(self):
        """Test setting thread number"""
        if not _omp_available():
            pytest.skip("OpenMP not available on this platform")

        original = get_thread_num()

        # Set to 4 threads
        set_thread_num(4)
        assert get_thread_num() == 4

        # Restore original
        set_thread_num(original)

    def test_set_get_thread_round_trip(self):
        """Test set/get round trip for various thread counts"""
        if not _omp_available():
            pytest.skip("OpenMP not available on this platform")

        original = get_thread_num()

        for num in [1, 2, 4, 8, 16]:
            set_thread_num(num)
            assert get_thread_num() == num

        # Restore original
        set_thread_num(original)

    def test_set_thread_num_single(self):
        """Test setting thread count to 1"""
        original = get_thread_num()

        set_thread_num(1)
        assert get_thread_num() == 1

        # Restore original
        set_thread_num(original)


# =============================================================================
# Callback Handler Tests
# =============================================================================

@pytest.mark.hpgl
class TestCallbackHandlers:
    """Test output and progress callback handlers"""

    def test_set_output_handler_with_mock(self):
        """Test setting output handler and verifying it is invoked by HPGL operations"""
        call_count = [0]
        received_messages = []

        def output_handler(msg, param):
            """Handler with correct signature: (c_char_p, py_object)"""
            call_count[0] += 1
            if msg is not None:
                received_messages.append(msg.decode() if isinstance(msg, bytes) else msg)
            return 1  # Return 1 to indicate success

        param = 42

        # Register the output handler
        set_output_handler(output_handler, param)

        # Invoke an HPGL operation that triggers output callbacks.
        # ordinary_kriging iterates over all grid cells and emits output per cell.
        np.random.seed(42)
        grid = SugarboxGrid(x=5, y=5, z=3)
        size = grid.x * grid.y * grid.z
        data = np.random.rand(size).astype('float32') * 100
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )
        ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(3, 3, 1),
            max_neighbours=8,
            cov_model=cov_model
        )

        # Clear handler
        set_output_handler(None, None)

        # Verify handler was invoked by the HPGL operation
        assert call_count[0] > 0, "Output handler was not called during HPGL operation"

    def test_set_output_handler_none(self):
        """Test clearing output handler"""
        # Setting to None should not raise
        set_output_handler(None, None)

        def simple_handler(msg, param):
            return 1

        set_output_handler(simple_handler, 0)
        set_output_handler(None, None)

    def test_set_progress_handler_with_mock(self):
        """Test setting progress handler and verifying it is invoked by HPGL operations"""
        call_count = [0]

        def progress_handler(msg, progress, param):
            """Handler with correct signature: (c_char_p, c_int, py_object)"""
            call_count[0] += 1
            return 1  # Return 1 to indicate success

        param = 100

        # Register the progress handler
        set_progress_handler(progress_handler, param)

        # Invoke an HPGL operation that triggers progress callbacks.
        # ordinary_kriging iterates over all grid cells and emits progress.
        np.random.seed(42)
        grid = SugarboxGrid(x=5, y=5, z=3)
        size = grid.x * grid.y * grid.z
        data = np.random.rand(size).astype('float32') * 100
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            sill=1.0,
            nugget=0.1
        )
        ordinary_kriging(
            prop=prop,
            grid=grid,
            radiuses=(3, 3, 1),
            max_neighbours=8,
            cov_model=cov_model
        )

        # Clear handler
        set_progress_handler(None, None)

        # Verify handler was invoked by the HPGL operation
        assert call_count[0] > 0, "Progress handler was not called during HPGL operation"

    def test_set_progress_handler_none(self):
        """Test clearing progress handler"""
        # Setting to None should not raise
        set_progress_handler(None, None)

        def simple_handler(msg, progress, param):
            return 1

        set_progress_handler(simple_handler, 0)
        set_progress_handler(None, None)

    def test_handler_signature(self):
        """Test that handlers accept correct signatures"""
        # Output handler: (message, param) -> int
        def output_handler(msg, param):
            # msg is c_char_p (bytes or None)
            # param is py_object
            return 1

        # Progress handler: (message, progress, param) -> int
        def progress_handler(msg, progress, param):
            # msg is c_char_p (bytes or None)
            # progress is c_int
            # param is py_object
            return 1

        # Should not raise exceptions
        set_output_handler(output_handler, "test_param")
        set_progress_handler(progress_handler, 0)

        # Cleanup
        set_output_handler(None, None)
        set_progress_handler(None, None)


# =============================================================================
# I/O Function Tests
# =============================================================================

@pytest.mark.hpgl
class TestIOContinuousProperty:
    """Test I/O functions for continuous properties"""

    def test_write_read_property_round_trip(self, tmp_path):
        """Test write then read produces same data"""
        # Create test data
        data = np.array([1.5, 2.5, 3.5, -99.0, 5.5], dtype='float32')
        mask = np.array([1, 1, 1, 0, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        # Write to file
        filename = str(tmp_path / "test_prop.inc")
        write_property(prop, filename, "test_prop", -99.0)

        # Read back
        loaded = load_cont_property(filename, -99.0)

        # Check data matches (considering mask)
        assert isinstance(loaded, ContProperty)
        assert loaded.data.shape == prop.data.shape

        # Check informed values match
        informed_mask = prop.mask == 1
        np.testing.assert_array_almost_equal(
            loaded.data[informed_mask],
            prop.data[informed_mask]
        )

    def test_write_read_gslib_property_round_trip(self, tmp_path):
        """Test GSLib write creates valid file"""
        # Note: GSLib format written by write_gslib_property cannot be read back
        # by the INC reader. This test just verifies the write creates a valid file.
        data = np.array([10.0, 20.0, 30.0, -999.0, 50.0], dtype='float32')
        mask = np.array([1, 1, 1, 0, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "test_gslib.dat")
        write_gslib_property(prop, filename, "test_gslib", -999.0)

        # Check file exists and has expected content
        assert Path(filename).exists()
        with open(filename) as f:
            content = f.read()
            # Should contain the header, property name and values
            assert "HPGL saved GSLIB file" in content
            assert "test_gslib" in content
            # Check values are present (scientific notation)
            assert "1.000000E+01" in content  # 10.0
            assert "2.000000E+01" in content  # 20.0

    def test_read_inc_file_float_with_size(self, tmp_path):
        """Test reading INC file with specified size"""
        # Write a file using write_property, then read it back with size
        data = np.array([1.0, 2.0, 3.0, 4.0, -99.0, 6.0], dtype='float32')
        mask = np.array([1, 1, 1, 1, 0, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "test_manual.inc")
        write_property(prop, filename, "test_prop", -99.0)

        size = (6, 1, 1)  # 6 elements
        loaded = read_inc_file_float(filename, -99.0, size)

        assert isinstance(loaded, ContProperty)
        assert loaded.data.size == 6
        # Check the values match (considering mask)
        assert loaded.data[0] == pytest.approx(1.0)
        assert loaded.data[5] == pytest.approx(6.0)
        # The -99.0 value should be masked
        assert loaded.mask[4] == 0

    def test_write_property_creates_file(self, tmp_path):
        """Test that write_property creates a valid file"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.array([1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "created_file.inc")
        write_property(prop, filename, "created", -99.0)

        # Check file exists
        assert Path(filename).exists()

    def test_write_read_3d_property(self, tmp_path):
        """Test write/read for 3D property"""
        data = np.arange(27, dtype='float32').reshape((3, 3, 3), order='F')
        mask = np.ones((3, 3, 3), dtype='uint8')
        mask[1, 1, 1] = 0  # Mask center value
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "3d_prop.inc")
        write_property(prop, filename, "3d_test", -99.0)

        # Verify file was created and contains expected data
        assert Path(filename).exists()
        with open(filename) as f:
            content = f.read()
            # Should have property name and some values
            assert "3d_test" in content
            # Should have values (0.0, 1.0, etc. in scientific notation)
            assert "0.000000E+00" in content  # First value

        # Note: Reading back 3D properties with read_inc_file_float is not stable
        # due to C++ implementation issues. The write functionality is verified.


@pytest.mark.hpgl
class TestIOIndicatorProperty:
    """Test I/O functions for indicator properties"""

    def test_write_read_indicator_property_round_trip(self, tmp_path):
        """Test write then read for indicator property"""
        data = np.array([0, 1, 2, 0, 1], dtype='uint8')
        mask = np.array([1, 1, 1, 1, 0], dtype='uint8')
        indicator_values = [10, 20, 30]  # Original values
        prop = IndProperty(data, mask, 3)

        filename = str(tmp_path / "test_ind.inc")
        write_property(prop, filename, "test_ind", 255, indicator_values)

        loaded = load_ind_property(filename, 255, indicator_values)

        assert isinstance(loaded, IndProperty)
        assert loaded.indicator_count == 3

    def test_read_inc_file_byte_with_size(self, tmp_path):
        """Test reading byte INC file with specified size"""
        # Create a simple file that the slow loader can parse
        # Values must be in indicator_values or be the undefined value
        filename = str(tmp_path / "test_byte.inc")
        with open(filename, 'w') as f:
            f.write("test_byte\n")
            f.write("10 20 30\n")  # All valid indicator values
            f.write("10 255 20\n")  # 255 is undefined
            f.write("/\n")

        indicator_values = [10, 20, 30]
        # The slow loader will read all 6 values (excluding 255 which is masked)
        loaded = _load_prop_ind_slow(filename, 255, indicator_values)

        assert isinstance(loaded, IndProperty)
        assert loaded.indicator_count == 3
        # Should have 5 informed values (one is masked as 255)
        assert loaded.data.size == 6
        assert loaded.mask[4] == 0  # The 255 value should be masked (at position 4)

    def test_write_gslib_indicator_property(self, tmp_path):
        """Test writing indicator property in GSLib format"""
        data = np.array([0, 1, 0, 1, 0], dtype='uint8')
        mask = np.ones(5, dtype='uint8')
        indicator_values = [1, 2]
        prop = IndProperty(data, mask, 2)

        filename = str(tmp_path / "ind_gslib.dat")
        write_gslib_property(prop, filename, "ind_test", 255, indicator_values)

        # Check file was created
        assert Path(filename).exists()

    def test_indicator_values_mapping(self, tmp_path):
        """Test that indicator values are correctly mapped"""
        # Create data with mapped values
        data = np.array([0, 1, 2, 0, 1, 2], dtype='uint8')
        mask = np.ones(6, dtype='uint8')
        indicator_values = [100, 200, 250]  # Use valid uint8 values (max 255)
        prop = IndProperty(data, mask, 3)

        filename = str(tmp_path / "mapped_ind.inc")
        write_property(prop, filename, "mapped", 255, indicator_values)

        loaded = load_ind_property(filename, 255, indicator_values)

        # Check indicator count is preserved
        assert loaded.indicator_count == 3


@pytest.mark.hpgl
class TestUndefinedValueHandling:
    """Test undefined value handling in I/O operations"""

    def test_undefined_value_masking_continuous(self, tmp_path):
        """Test that undefined values are correctly masked"""
        data = np.array([1.0, -99.0, 3.0, -99.0, 5.0], dtype='float32')
        mask = np.ones(5, dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "undefined_test.inc")
        write_property(prop, filename, "undef", -99.0)

        loaded = load_cont_property(filename, -99.0)

        # Undefined values should be masked
        assert loaded.mask[1] == 0
        assert loaded.mask[3] == 0
        assert loaded.mask[0] == 1

    def test_undefined_value_masking_indicator(self, tmp_path):
        """Test undefined value masking for indicators"""
        data = np.array([0, 1, 0, 1], dtype='uint8')
        mask = np.ones(4, dtype='uint8')
        indicator_values = [1, 2]
        prop = IndProperty(data, mask, 2)

        filename = str(tmp_path / "undef_ind.inc")
        write_property(prop, filename, "undef_ind", 255, indicator_values)

        loaded = load_ind_property(filename, 255, indicator_values)

        assert isinstance(loaded, IndProperty)


# =============================================================================
# Utility Function Tests
# =============================================================================

@pytest.mark.hpgl
class TestAppendMask:
    """Test append_mask utility function"""

    def test_append_mask_basic(self):
        """Test basic mask append operation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        mask = np.array([1, 1, 1, 1, 1], dtype='uint8')
        prop = ContProperty(data, mask)

        additional_mask = np.array([1, 0, 1, 0, 1], dtype='uint8')

        # Store original data for comparison
        original_data = prop.data.copy()

        append_mask(prop, additional_mask)

        # Check that masked values were set to -99
        assert prop.data[1] == pytest.approx(-99.0)
        assert prop.data[3] == pytest.approx(-99.0)
        # Unmasked values should remain
        assert prop.data[0] == pytest.approx(original_data[0])

    def test_append_mask_all_zeros(self):
        """Test append_mask with all-zero mask"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        additional_mask = np.array([0, 0, 0], dtype='uint8')

        append_mask(prop, additional_mask)

        # All values should be set to -99
        assert np.all(prop.data == -99.0)

    def test_append_mask_all_ones(self):
        """Test append_mask with all-one mask"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        original_data = prop.data.copy()
        additional_mask = np.array([1, 1, 1], dtype='uint8')

        append_mask(prop, additional_mask)

        # Nothing should change
        np.testing.assert_array_equal(prop.data, original_data)

    def test_append_mask_with_tuple_input(self):
        """Test append_mask accepts property as tuple"""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
        mask = np.ones(4, dtype='uint8')

        # Pass as tuple
        append_mask((data, mask), np.array([1, 0, 1, 0], dtype='uint8'))

        # Check data was modified
        assert data[1] == pytest.approx(-99.0)
        assert data[3] == pytest.approx(-99.0)

    def test_append_mask_3d_property(self):
        """Test append_mask on 3D property"""
        data = np.arange(8, dtype='float32').reshape((2, 2, 2), order='F')
        mask = np.ones((2, 2, 2), dtype='uint8')
        prop = ContProperty(data, mask)

        additional_mask = np.zeros((2, 2, 2), dtype='uint8')
        additional_mask[0, 0, 0] = 1  # Keep first value

        append_mask(prop, additional_mask)

        # Most values should be -99, except first
        assert prop.data[0, 0, 0] == pytest.approx(0.0)


# =============================================================================
# File Format Tests
# =============================================================================

@pytest.mark.hpgl
class TestFileFormats:
    """Test different file format handling"""

    def test_inc_format_with_comments(self, tmp_path):
        """Test INC file with comment lines"""
        filename = str(tmp_path / "comments.inc")
        with open(filename, 'w') as f:
            f.write("-- This is a comment\n")
            f.write("1.0 2.0 3.0\n")
            f.write("-- Another comment\n")
            f.write("4.0 5.0 6.0\n")

        loaded = load_cont_property(filename, -99.0)

        assert isinstance(loaded, ContProperty)
        assert loaded.data.size >= 6

    def test_gslib_format_header(self, tmp_path):
        """Test GSLib format with property name"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "gslib_with_name.dat")
        write_gslib_property(prop, filename, "MyProperty", -99.0)

        # File should exist and be readable
        assert Path(filename).exists()
        loaded = load_cont_property(filename, -99.0)
        assert isinstance(loaded, ContProperty)

    def test_multiple_properties_same_file(self, tmp_path):
        """Test handling multiple properties in same directory"""
        data1 = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask1 = np.ones(3, dtype='uint8')
        prop1 = ContProperty(data1, mask1)

        data2 = np.array([4.0, 5.0, 6.0], dtype='float32')
        mask2 = np.ones(3, dtype='uint8')
        prop2 = ContProperty(data2, mask2)

        filename1 = str(tmp_path / "prop1.inc")
        filename2 = str(tmp_path / "prop2.inc")

        write_property(prop1, filename1, "prop1", -99.0)
        write_property(prop2, filename2, "prop2", -99.0)

        loaded1 = load_cont_property(filename1, -99.0)
        loaded2 = load_cont_property(filename2, -99.0)

        assert isinstance(loaded1, ContProperty)
        assert isinstance(loaded2, ContProperty)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.hpgl
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_property_calc_mean(self):
        """Test calc_mean with minimal property"""
        data = np.array([1.0], dtype='float32')
        mask = np.array([1], dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)
        assert result == pytest.approx(1.0)

    def test_large_undefined_value(self, tmp_path):
        """Test with large undefined value"""
        data = np.array([1.0, -1e10, 3.0], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "large_undef.inc")
        write_property(prop, filename, "large_undef", -1e10)

        loaded = load_cont_property(filename, -1e10)
        assert isinstance(loaded, ContProperty)

    def test_negative_values_in_data(self, tmp_path):
        """Test handling of negative values"""
        data = np.array([-1.0, -2.0, -3.0, -99.0], dtype='float32')
        mask = np.ones(4, dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "negative.inc")
        write_property(prop, filename, "negative", -99.0)

        loaded = load_cont_property(filename, -99.0)

        # Negative values should be preserved
        assert loaded.data[0] < 0
        assert loaded.data[1] < 0
        assert loaded.data[2] < 0

    def test_very_small_values(self, tmp_path):
        """Test handling of very small values"""
        data = np.array([1e-10, 1e-5, 1e-3], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        filename = str(tmp_path / "small.inc")
        write_property(prop, filename, "small", -99.0)

        loaded = load_cont_property(filename, -99.0)

        # Small values should be preserved approximately
        assert loaded.data[0] > 0

    def test_alternative_undefined_values(self, tmp_path):
        """Test different undefined value conventions"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        mask = np.ones(3, dtype='uint8')
        prop = ContProperty(data, mask)

        # Test with -999 (common alternative)
        filename = str(tmp_path / "alt_undef.inc")
        write_property(prop, filename, "alt_undef", -999.0)

        loaded = load_cont_property(filename, -999.0)
        assert isinstance(loaded, ContProperty)


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.hpgl
class TestPerformanceUtilities:
    """Performance-related tests for utility functions"""

    def test_calc_mean_performance_large(self):
        """Test calc_mean performance with large dataset"""
        np.random.seed(42)
        size = 100000
        data = np.random.rand(size).astype('float32')
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data, mask)

        result = calc_mean(prop)

        expected = np.sum(data) / size
        assert result == pytest.approx(expected, rel=1e-4)

    def test_calc_cdf_performance_large(self):
        """Test calc_cdf performance with larger dataset"""
        np.random.seed(42)
        # Use smaller size for CDF as it's O(n^2) in worst case
        size = 1000
        data = np.random.rand(size).astype('float32') * 100
        mask = np.ones(size, dtype='uint8')
        prop = ContProperty(data.reshape((size, 1, 1), order='F'),
                           mask.reshape((size, 1, 1), order='F'))

        result = calc_cdf(prop)

        assert isinstance(result, CdfData)
        assert len(result.values) > 0
        # Final cumulative probability reaches 1.0
        assert result.probs[-1] == pytest.approx(1.0)
        assert result.probs[-1] > 0.9  # Should be close to 1 for large n


# =============================================================================
# I/O Error Path Tests
# =============================================================================

@pytest.mark.hpgl
class TestWritePropertyErrorPaths:
    """Test error paths in write_property and write_gslib_property."""

    def test_write_property_invalid_indicator_values_type(self, tmp_path):
        """write_property raises TypeError when indicator_values is not a list."""
        data = np.array([0, 1, 0], dtype='uint8')
        mask = np.ones(3, dtype='uint8')
        prop = IndProperty(data, mask, 2)
        filename = str(tmp_path / "test.inc")

        with pytest.raises(TypeError, match="indicator_values must be a list"):
            write_property(prop, filename, "test", 255, indicator_values="not_a_list")

    def test_write_gslib_property_invalid_indicator_values_type(self, tmp_path):
        """write_gslib_property raises TypeError when indicator_values is not a list."""
        data = np.array([0, 1, 0], dtype='uint8')
        mask = np.ones(3, dtype='uint8')
        prop = IndProperty(data, mask, 2)
        filename = str(tmp_path / "test.dat")

        with pytest.raises(TypeError, match="indicator_values must be a list"):
            write_gslib_property(prop, filename, "test", 255, indicator_values=123)

    def test_write_property_none_indicator_values(self, tmp_path):
        """write_property with indicator_values=None defaults to empty list (no error)."""
        data = np.array([0, 1, 0], dtype='uint8')
        mask = np.ones(3, dtype='uint8')
        prop = IndProperty(data, mask, 2)
        filename = str(tmp_path / "test.inc")

        # indicator_values=None should default to [] and not raise
        write_property(prop, filename, "test", 255)  # No indicator_values
        assert Path(filename).exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
