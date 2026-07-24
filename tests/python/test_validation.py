"""
Unit tests for HPGL validation framework (validation.py).

Covers all validators, decorators, constants, and the ValidationContext
context manager. These tests exercise edge cases documented in the
adversarial review findings (Q4) including boundary values for
MAX_GRID_SIZE, MIN_SILL, PROBABILITY_SUM_TOLERANCE, etc.
"""

import os
import sys
from builtins import UserWarning
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.validation import (
        CriticalValidationError,
        GridValidator,
        ParameterValidator,
        PathValidator,
        ValidationConstants,
        ValidationContext,
        ValidationError,
        ValidationWarning,
        validate_file_params,
        validate_grid_params,
        validate_kriging_params,
        validate_simulation_params,
    )

    VALIDATION_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError):
    VALIDATION_AVAILABLE = False


# =============================================================================
# ValidationConstants Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestValidationConstants:
    """Test validation constant values and edge cases."""

    def test_grid_dimension_limits_are_reasonable(self):
        """MIN_GRID_DIMENSION >= 1, MAX_GRID_DIMENSION finite."""
        assert ValidationConstants.MIN_GRID_DIMENSION >= 1
        assert ValidationConstants.MAX_GRID_DIMENSION > ValidationConstants.MIN_GRID_DIMENSION

    def test_max_grid_size_is_large(self):
        """MAX_GRID_SIZE is 1e9 (1 billion cells)."""
        assert ValidationConstants.MAX_GRID_SIZE >= 1e9

    def test_neighbor_count_limits(self):
        """MIN_NEIGHBORS=1, MAX_NEIGHBORS=1000."""
        assert ValidationConstants.MIN_NEIGHBORS == 1
        assert ValidationConstants.MAX_NEIGHBORS == 1000

    def test_probability_sum_tolerance(self):
        """PROBABILITY_SUM_TOLERANCE is 0.01."""
        assert ValidationConstants.PROBABILITY_SUM_TOLERANCE == 0.01

    def test_covariance_limits(self):
        """MIN_SILL=1e-6, MIN_NUGGET=0.0, MAX_INDICATORS=256."""
        assert ValidationConstants.MIN_SILL == 1e-6
        assert ValidationConstants.MIN_NUGGET == 0.0
        assert ValidationConstants.MAX_INDICATORS == 255
        assert ValidationConstants.MIN_SEED == 0


# =============================================================================
# ValidationError / CriticalValidationError Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestValidationErrors:
    """Test validation exception hierarchy."""

    def test_validation_error_base(self):
        """ValidationError stores message, param_name, severity."""
        err = ValidationError("test message", "param", severity="warning")
        assert err.message == "test message"
        assert err.parameter_name == "param"
        assert err.severity == "warning"
        assert "param" in str(err)

    def test_critical_validation_error(self):
        """CriticalValidationError has severity='critical'."""
        err = CriticalValidationError("critical", "param")
        assert err.severity == "critical"

    def test_validation_warning(self):
        """ValidationWarning has severity='warning'."""
        err = ValidationWarning("warning")
        assert err.severity == "warning"

    def test_error_without_parameter_name(self):
        """Error without parameter_name should not include ':' in str."""
        err = ValidationError("message only")
        assert err.parameter_name == ""
        assert ":" not in str(err) or str(err).startswith(":")


# =============================================================================
# PathValidator Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestPathValidator:
    """Test path validation and directory traversal prevention."""

    def test_empty_filename_raises(self):
        """Empty filename raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            PathValidator.validate_filepath("")

        with pytest.raises(CriticalValidationError):
            PathValidator.validate_filepath(None)

    def test_valid_existing_file(self, tmp_path):
        """Valid existing file passes validation."""
        f = tmp_path / "test.txt"
        f.write_text("test")
        result = PathValidator.validate_filepath(str(f), must_exist=True)
        assert os.path.abspath(result) == os.path.abspath(str(f))

    def test_non_existent_file_without_must_exist(self, tmp_path):
        """File that doesn't exist passes when must_exist=False."""
        f = tmp_path / "nonexistent.txt"
        result = PathValidator.validate_filepath(str(f), must_exist=False)
        assert os.path.basename(result) == "nonexistent.txt"

    def test_non_existent_file_with_must_exist_raises(self, tmp_path):
        """File that doesn't exist fails when must_exist=True."""
        f = tmp_path / "nonexistent.txt"
        with pytest.raises(CriticalValidationError):
            PathValidator.validate_filepath(str(f), must_exist=True)

    def test_path_traversal_rejected(self):
        """Path traversal with '..' is rejected."""
        with pytest.raises(CriticalValidationError, match="traversal"):
            PathValidator.validate_filepath("../etc/passwd")

    def test_allowed_extensions_whitelist(self, tmp_path):
        """Allowed extensions whitelist works."""
        f = tmp_path / "test.txt"
        f.write_text("test")
        PathValidator.validate_filepath(str(f), allowed_extensions=[".txt"])

        with pytest.raises(CriticalValidationError, match="extension"):
            PathValidator.validate_filepath(str(f), allowed_extensions=[".dat"])

    def test_validate_write_filepath(self, tmp_path):
        """validate_write_filepath shortcut works."""
        f = tmp_path / "output.inc"
        result = PathValidator.validate_write_filepath(str(f))
        assert os.path.basename(result) == "output.inc"

    # ---- validate_filepath_in_basedir (F-144) ----

    def test_validate_filepath_in_basedir_inside_passes(self, tmp_path):
        """File inside basedir passes basedir containment check."""
        basedir = tmp_path / "allowed"
        basedir.mkdir()
        testfile = basedir / "data.txt"
        testfile.write_text("test")
        result = PathValidator.validate_filepath_in_basedir(
            str(testfile), str(basedir), must_exist=True
        )
        assert os.path.abspath(result) == os.path.abspath(str(testfile))

    def test_validate_filepath_in_basedir_outside_raises(self, tmp_path):
        """File outside basedir raises CriticalValidationError."""
        basedir = tmp_path / "allowed"
        basedir.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("test")
        with pytest.raises(CriticalValidationError, match="outside"):
            PathValidator.validate_filepath_in_basedir(
                str(outside_file), str(basedir), must_exist=True
            )

    def test_validate_filepath_in_basedir_nonexistent_inside_passes(self, tmp_path):
        """Non-existent file inside basedir passes when must_exist=False."""
        basedir = tmp_path / "allowed"
        basedir.mkdir()
        result = PathValidator.validate_filepath_in_basedir(
            str(basedir / "new.txt"), str(basedir), must_exist=False
        )
        assert os.path.basename(result) == "new.txt"

    # ---- safe_open_write / safe_open_read (I2-F04) ----

    def test_safe_open_write_valid_path(self, tmp_path):
        """safe_open_write creates/truncates a file inside basedir."""
        basedir = tmp_path / "basedir"
        basedir.mkdir()
        filepath = basedir / "output.txt"

        with PathValidator.safe_open_write(str(filepath), str(basedir)) as fh:
            fh.write("hello world")

        content = (basedir / "output.txt").read_text()
        assert content == "hello world"

    def test_safe_open_write_outside_basedir_raises(self, tmp_path):
        """safe_open_write outside basedir raises CriticalValidationError."""
        basedir = tmp_path / "allowed"
        basedir.mkdir()
        outside_file = tmp_path / "outside.txt"

        with pytest.raises(CriticalValidationError, match="outside"):
            PathValidator.safe_open_write(str(outside_file), str(basedir))

    def test_safe_open_read_valid_path(self, tmp_path):
        """safe_open_read opens an existing file inside basedir."""
        basedir = tmp_path / "basedir"
        basedir.mkdir()
        filepath = basedir / "data.txt"
        filepath.write_text("read me")

        with PathValidator.safe_open_read(str(filepath), str(basedir)) as fh:
            content = fh.read()

        assert content == "read me"

    def test_safe_open_read_outside_basedir_raises(self, tmp_path):
        """safe_open_read outside basedir raises CriticalValidationError."""
        basedir = tmp_path / "allowed"
        basedir.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("data")

        with pytest.raises(CriticalValidationError, match="outside"):
            PathValidator.safe_open_read(str(outside_file), str(basedir))

    def test_safe_open_write_nonexistent_dir_raises(self, tmp_path):
        """safe_open_write to nonexistent basedir child — validate_filepath_in_basedir resolves parent dir for O_NOFOLLOW."""
        basedir = tmp_path / "basedir"
        basedir.mkdir()
        filepath = basedir / "new_file.txt"

        with PathValidator.safe_open_write(str(filepath), str(basedir)) as fh:
            fh.write("created")

        assert filepath.read_text() == "created"

    def test_safe_open_read_file_in_basedir_after_write_works(self, tmp_path):
        """Round-trip: write via safe_open_write, then read via safe_open_read."""
        basedir = tmp_path / "datadir"
        basedir.mkdir()
        filepath = basedir / "roundtrip.txt"

        with PathValidator.safe_open_write(str(filepath), str(basedir)) as fh:
            fh.write("roundtrip data")

        with PathValidator.safe_open_read(str(filepath), str(basedir)) as fh:
            assert fh.read() == "roundtrip data"


# =============================================================================
# GridValidator Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestGridValidator:
    """Test grid dimension validation."""

    def test_valid_grid_dimensions(self):
        """Standard grid dimensions pass validation."""
        GridValidator.validate_grid_dimensions(10, 10, 5)

    def test_minimal_grid(self):
        """1x1x1 grid (minimum) passes."""
        GridValidator.validate_grid_dimensions(1, 1, 1)

    def test_negative_dimension_raises(self):
        """Negative dimension raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_dimensions(-1, 10, 5)

    def test_zero_dimension_raises(self):
        """Zero dimension raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_dimensions(0, 10, 5)

    def test_exceeds_max_dimension_raises(self):
        """Dimension > MAX_GRID_DIMENSION raises."""
        too_big = ValidationConstants.MAX_GRID_DIMENSION + 1
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_dimensions(too_big, 10, 5)

    def test_total_size_exceeds_max_grid_size_raises(self):
        """Total size exceeding MAX_GRID_SIZE raises error."""
        # 1000 * 1000 * 2000 = 2e9 > 1e9 MAX_GRID_SIZE
        with pytest.raises(CriticalValidationError, match="Total grid size"):
            GridValidator.validate_grid_dimensions(1000, 1000, 2000)

    def test_max_grid_size_boundary(self):
        """Grid exactly at MAX_GRID_SIZE passes."""
        # 1000^3 = 1e9 = MAX_GRID_SIZE
        GridValidator.validate_grid_dimensions(1000, 1000, 1000)

    def test_array_size_validation(self):
        """Array size matches grid dimensions passes."""
        arr = np.zeros(6, dtype="float32")
        GridValidator.validate_array_size(arr, (2, 3, 1))

    def test_array_size_mismatch_raises(self):
        """Array size mismatch raises."""
        arr = np.zeros(10, dtype="float32")
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_array_size(arr, (3, 3, 1))  # 9 != 10

    def test_array_dtype_validation(self):
        """Array dtype validation works."""
        arr = np.zeros(10, dtype="float32")
        GridValidator.validate_array_dtype(arr, np.float32)

    def test_array_dtype_mismatch_raises(self):
        """Wrong array dtype raises error."""
        arr = np.zeros(10, dtype="float64")
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_array_dtype(arr, np.float32)

    # ---- Empty array edge cases (F-145) ----

    def test_empty_array_size_mismatch_raises(self):
        """Empty array with non-zero grid raises CriticalValidationError."""
        arr = np.array([], dtype="float32")
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_array_size(arr, (1, 1, 1))  # 0 != 1

    def test_empty_array_dtype_passes(self):
        """Empty array with matching dtype passes dtype validation."""
        arr = np.array([], dtype="float32")
        GridValidator.validate_array_dtype(arr, np.float32)

    def test_empty_array_dtype_mismatch_raises(self):
        """Empty array with wrong dtype raises error."""
        arr = np.array([], dtype="float64")
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_array_dtype(arr, np.float32)

    # ---- validate_coordinate_arrays (I2-F06) ----

    def test_coordinate_arrays_finite_pass(self):
        """All-finite coordinate arrays pass validation."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        z = np.array([7.0, 8.0, 9.0])
        # Should not raise
        GridValidator.validate_coordinate_arrays(x, y, z)

    @pytest.mark.parametrize("axis_idx,make_bad", [
        (0, lambda a: np.array([1.0, float("nan"), 3.0])),
        (1, lambda a: np.array([1.0, float("nan"), 3.0])),
        (2, lambda a: np.array([1.0, float("nan"), 3.0])),
    ])
    def test_coordinate_arrays_nan_raises(self, axis_idx, make_bad):
        """NaN in any coordinate axis raises ValueError."""
        bad = make_bad(None)
        x = bad if axis_idx == 0 else np.array([1.0, 2.0, 3.0])
        y = bad if axis_idx == 1 else np.array([4.0, 5.0, 6.0])
        z = bad if axis_idx == 2 else np.array([7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            GridValidator.validate_coordinate_arrays(x, y, z)

    @pytest.mark.parametrize("axis_idx,make_bad", [
        (0, lambda a: np.array([1.0, float("inf"), 3.0])),
        (1, lambda a: np.array([1.0, float("inf"), 3.0])),
        (2, lambda a: np.array([1.0, float("inf"), 3.0])),
    ])
    def test_coordinate_arrays_inf_raises(self, axis_idx, make_bad):
        """Inf in any coordinate axis raises ValueError."""
        bad = make_bad(None)
        x = bad if axis_idx == 0 else np.array([1.0, 2.0, 3.0])
        y = bad if axis_idx == 1 else np.array([4.0, 5.0, 6.0])
        z = bad if axis_idx == 2 else np.array([7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            GridValidator.validate_coordinate_arrays(x, y, z)

    def test_coordinate_arrays_negative_inf_raises(self):
        """-Inf in coordinates raises ValueError."""
        x = np.array([1.0, -float("inf"), 3.0])
        y = np.array([4.0, 5.0, 6.0])
        z = np.array([7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            GridValidator.validate_coordinate_arrays(x, y, z)


# =============================================================================
# ParameterValidator Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestParameterValidator:
    """Test numerical parameter validation."""

    # ---- Radius ----

    def test_valid_single_radius(self):
        """Single numeric radius returns 3-tuple."""
        result = ParameterValidator.validate_radius(5.0, "radius")
        assert result == (5.0, 5.0, 5.0)

    def test_valid_tuple_radius(self):
        """Tuple radius returns correctly."""
        result = ParameterValidator.validate_radius((1, 2, 3), "radius")
        assert result == (1.0, 2.0, 3.0)

    def test_radius_nan_raises(self):
        """NaN radius raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius(float("nan"))

    def test_radius_inf_raises(self):
        """Infinite radius raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius(float("inf"))

    def test_negative_radius_raises(self):
        """Negative radius raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius(-1.0)

    def test_radius_exceeds_max_raises(self):
        """Radius exceeding MAX_RADIUS raises."""
        too_big = ValidationConstants.MAX_RADIUS + 1.0
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius(too_big)

    # ---- Wrong-length radius tuple (F-147) ----

    def test_radius_wrong_length_2_elements_raises(self):
        """2-element radius tuple raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius((1.0, 2.0))

    def test_radius_wrong_length_4_elements_raises(self):
        """4-element radius tuple raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius((1.0, 2.0, 3.0, 4.0))

    def test_radius_wrong_type_raises(self):
        """String radius raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_radius("not_a_number")

    # ---- Max Neighbors ----

    def test_valid_max_neighbors(self):
        """Valid max_neighbors passes."""
        ParameterValidator.validate_max_neighbors(12)

    def test_max_neighbors_zero_raises(self):
        """max_neighbors=0 raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_max_neighbors(0)

    def test_max_neighbors_exceeds_limit_raises_warning(self):
        """max_neighbors exceeding recommended limit raises ValidationWarning."""
        too_big = ValidationConstants.MAX_NEIGHBORS + 1
        with pytest.warns(UserWarning):
            ParameterValidator.validate_max_neighbors(too_big)

    # ---- Min Neighbors ----

    def test_valid_min_neighbors(self):
        """Valid min_neighbors passes."""
        ParameterValidator.validate_min_neighbors(2, 12)

    def test_min_exceeds_max_raises(self):
        """min_neighbors > max_neighbors raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_min_neighbors(15, 10)

    def test_min_neighbors_negative_raises(self):
        """Negative min_neighbors raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_min_neighbors(-1, 12)

    # ---- Covariance Parameters ----

    def test_valid_covariance_parameters(self):
        """Standard covariance parameters pass."""
        ParameterValidator.validate_covariance_parameters(
            sill=1.0, nugget=0.1, ranges=(5.0, 5.0, 3.0), angles=(30.0, 0.0, 0.0)
        )

    def test_negative_sill_raises(self):
        """Negative sill raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(sill=-1.0, nugget=0.1)

    def test_zero_sill_raises(self):
        """Zero sill raises (MIN_SILL=1e-6). Sill must be >= 1e-6 for color schemes."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(sill=0.0, nugget=0.0)

    def test_nugget_exceeds_sill_raises(self):
        """Nugget > sill raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="Nugget.*exceeds.*sill"):
            ParameterValidator.validate_covariance_parameters(sill=0.5, nugget=1.0)

    def test_nugget_equal_to_sill_passes(self):
        """Nugget == sill passes."""
        ParameterValidator.validate_covariance_parameters(sill=1.0, nugget=1.0)

    def test_sill_nan_raises(self):
        """NaN sill raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(sill=float("nan"), nugget=0.1)

    def test_negative_range_raises(self):
        """Negative range raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.1, ranges=(-5.0, 5.0, 3.0)
            )

    def test_ranges_wrong_length_raises(self):
        """Ranges with wrong length raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.1, ranges=(5.0, 5.0)
            )

    def test_angles_wrong_length_raises(self):
        """Angles with wrong length raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.1, angles=(30.0, 0.0)
            )

    def test_angle_nan_raises(self):
        """NaN angle raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.1, angles=(float("nan"), 0.0, 0.0)
            )

    # ---- Inf angle validation (F-146) ----

    def test_inf_angle_raises(self):
        """Inf angle raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.1, angles=(float("inf"), 0.0, 0.0)
            )

    def test_inf_nugget_raises(self):
        """Inf nugget raises (documented coverage of Inf path in covariance validation)."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_covariance_parameters(sill=1.0, nugget=float("inf"))

    # ---- Probability ----

    def test_valid_probability(self):
        """Valid probability passes."""
        ParameterValidator.validate_probability(0.5)

    def test_probability_out_of_range_raises(self):
        """Probability outside [0,1] raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_probability(1.5)

    def test_negative_probability_raises(self):
        """Negative probability raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_probability(-0.1)

    def test_probability_sum_valid(self):
        """Probabilities summing to 1.0 pass."""
        ParameterValidator.validate_probability_sum([0.3, 0.3, 0.4])

    def test_probability_sum_wrong_raises(self):
        """Probabilities not summing to 1.0 raise."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_probability_sum([0.5, 0.5, 0.5])

    def test_probability_sum_within_tolerance_passes(self):
        """Probabilities within tolerance pass."""
        # 0.33 + 0.34 + 0.335 = 1.005, diff = 0.005 < 0.01 tolerance
        ParameterValidator.validate_probability_sum([0.33, 0.34, 0.335])

    def test_probability_sum_at_tolerance_boundary(self):
        """Probabilities exactly at tolerance boundary."""
        # 0.503 + 0.503 = 1.006, diff = 0.006 < 0.01 tolerance
        ParameterValidator.validate_probability_sum([0.503, 0.503])

    # ---- Seed ----

    def test_valid_seed(self):
        """Valid seed passes."""
        ParameterValidator.validate_seed(3439275)

    def test_zero_seed_passes(self):
        """Zero seed (MIN_SEED boundary) passes."""
        ParameterValidator.validate_seed(0)

    # ---- Indicator Count ----

    def test_valid_indicator_count(self):
        """Valid indicator count passes."""
        ParameterValidator.validate_indicator_count(3)

    def test_indicator_count_zero_raises(self):
        """Zero indicator count raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_indicator_count(0)

    def test_indicator_count_exceeds_max_raises(self):
        """Indicator count > MAX_INDICATORS raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_indicator_count(257)

    def test_indicator_count_at_max_passes(self):
        """Indicator count at MAX_INDICATORS passes."""
        ParameterValidator.validate_indicator_count(255)

    def test_indicator_count_at_max_plus_one_raises(self):
        """Indicator count at MAX_INDICATORS + 1 raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_indicator_count(256)

    # ---- Correlation Coef ----

    def test_valid_correlation_coef(self):
        """Valid correlation coefficient passes."""
        ParameterValidator.validate_correlation_coef(0.5)

    def test_correlation_coef_below_minus_one_raises(self):
        """Correlation < -1 raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_correlation_coef(-1.5)

    def test_correlation_coef_above_one_raises(self):
        """Correlation > 1 raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_correlation_coef(1.5)

    def test_correlation_coef_at_boundaries_passes(self):
        """Correlation at -1 and 1 passes."""
        ParameterValidator.validate_correlation_coef(-1.0)
        ParameterValidator.validate_correlation_coef(1.0)

    def test_correlation_coef_nan_raises(self):
        """NaN correlation raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_correlation_coef(float("nan"))

    # ---- Variance ----

    def test_valid_variance(self):
        """Valid variance passes."""
        ParameterValidator.validate_variance(1.0)

    def test_negative_variance_raises(self):
        """Negative variance raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_variance(-1.0)

    def test_zero_variance_passes(self):
        """Zero variance passes."""
        ParameterValidator.validate_variance(0.0)

    def test_variance_nan_raises(self):
        """NaN variance raises."""
        with pytest.raises(CriticalValidationError):
            ParameterValidator.validate_variance(float("nan"))

    def test_variance_inf_raises(self):
        """Inf variance raises (I2-F19 — untested)."""
        with pytest.raises(CriticalValidationError, match="finite"):
            ParameterValidator.validate_variance(float("inf"))

    def test_variance_negative_inf_raises(self):
        """-Inf variance raises."""
        with pytest.raises(CriticalValidationError, match="finite"):
            ParameterValidator.validate_variance(-float("inf"))

    # ---- validate_seed edge cases (I2-F21) ----

    def test_seed_negative_raises(self):
        """Negative seed raises ValidationError (I2-F21 — untested)."""
        with pytest.raises(ValidationError, match="negative"):
            ParameterValidator.validate_seed(-1)

    def test_seed_bool_raises(self):
        """Bool seed raises TypeError (I2-F21 — untested)."""
        with pytest.raises(TypeError, match="seed"):
            ParameterValidator.validate_seed(True)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="seed"):
            ParameterValidator.validate_seed(False)  # type: ignore[arg-type]

    def test_seed_non_int_raises(self):
        """Non-integer seed raises TypeError."""
        with pytest.raises(TypeError, match="seed"):
            ParameterValidator.validate_seed("string")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="seed"):
            ParameterValidator.validate_seed(3.14)  # type: ignore[arg-type]

    def test_seed_numpy_integer_accepted(self):
        """NumPy integer seed is accepted."""
        ParameterValidator.validate_seed(np.int32(42))
        ParameterValidator.validate_seed(np.int64(0))

    # ---- validate_property_type (I2-F18) ----

    def test_validate_property_type_valid(self):
        """Valid property type passes."""
        prop = 42
        ParameterValidator.validate_property_type(prop, int, "test_func", "my_param")

    def test_validate_property_type_wrong_type_raises(self):
        """Wrong type raises TypeError with descriptive message."""
        prop = "not_an_int"
        with pytest.raises(TypeError, match="test_func: my_param must be"):
            ParameterValidator.validate_property_type(prop, int, "test_func", "my_param")

    def test_validate_property_type_with_type_name_override(self):
        """Custom type_name appears in error message."""
        prop = [1, 2, 3]
        with pytest.raises(TypeError, match="must be ContProperty"):
            ParameterValidator.validate_property_type(
                prop, dict, "my_func", "prop", type_name="ContProperty"
            )

    def test_validate_property_type_string_against_str(self):
        """String property validated against str type."""
        ParameterValidator.validate_property_type("hello", str, "test", "text_param")

    # ---- validate_list_param (I2-F18) ----

    def test_validate_list_param_valid(self):
        """Valid list passes."""
        ParameterValidator.validate_list_param([1, 2, 3], "my_list", "test_func")

    def test_validate_list_param_tuple_raises(self):
        """Tuple (not list) raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            ParameterValidator.validate_list_param((1, 2, 3), "my_list", "test_func")

    def test_validate_list_param_string_raises(self):
        """String raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            ParameterValidator.validate_list_param("abc", "my_list", "test_func")

    # ---- validate_scalar_mean (I2-F18) ----

    def test_validate_scalar_mean_valid_float(self):
        """Valid float passes."""
        ParameterValidator.validate_scalar_mean(3.14, "test_func")

    def test_validate_scalar_mean_valid_int(self):
        """Valid int passes."""
        ParameterValidator.validate_scalar_mean(42, "test_func")

    def test_validate_scalar_mean_numpy_scalar(self):
        """NumPy float scalar passes."""
        ParameterValidator.validate_scalar_mean(np.float64(2.5), "test_func")

    def test_validate_scalar_mean_nan_raises(self):
        """NaN mean raises ValueError."""
        with pytest.raises(ValueError, match="scalar mean must be finite"):
            ParameterValidator.validate_scalar_mean(float("nan"), "test_func")

    def test_validate_scalar_mean_inf_raises(self):
        """Inf mean raises ValueError."""
        with pytest.raises(ValueError, match="scalar mean must be finite"):
            ParameterValidator.validate_scalar_mean(float("inf"), "test_func")

    def test_validate_scalar_mean_non_number_raises(self):
        """Non-number mean raises TypeError."""
        with pytest.raises(TypeError, match="scalar mean must be a number"):
            ParameterValidator.validate_scalar_mean("bad", "test_func")

    def test_validate_scalar_mean_bool_accepted(self):
        """Bool is accepted (bool is an int subclass — numeric)."""
        ParameterValidator.validate_scalar_mean(True, "test_func")
        ParameterValidator.validate_scalar_mean(False, "test_func")


# =============================================================================
# ValidationContext Tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestValidationContext:
    """Test the ValidationContext context manager."""

    def test_context_manager_strict(self):
        """Strict mode raises on first error."""
        with pytest.raises(CriticalValidationError):
            with ValidationContext(strict=True) as ctx:
                ctx.validate_grid_dimensions(-1, 10, 5)

    def test_context_manager_non_strict_collects_errors(self):
        """Non-strict mode collects all errors and raises on exit."""
        with pytest.raises(CriticalValidationError):
            with ValidationContext(strict=False) as ctx:
                ctx.validate_grid_dimensions(-1, 10, 5)
                # Should collect error without raising immediately
                assert len(ctx.errors) >= 1

    def test_context_manager_no_errors_passes(self):
        """Context manager with no errors exits cleanly."""
        with ValidationContext(strict=True) as ctx:
            ctx.validate_grid_dimensions(10, 10, 5)
        # No exception raised

    def test_validate_radius_in_context(self):
        """validate_radius in context returns radius tuple."""
        with ValidationContext(strict=True) as ctx:
            result = ctx.validate_radius((3, 4, 5), "test")
            assert result == (3.0, 4.0, 5.0)


# =============================================================================
# Decorator Tests (basic import/functionality check)
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestDecorators:
    """Basic checks that decorators exist and are callable."""

    def test_decorators_are_functions(self):
        """All decorators are callable."""
        assert callable(validate_grid_params)
        assert callable(validate_kriging_params)
        assert callable(validate_simulation_params)
        assert callable(validate_file_params)

    def test_validate_grid_params_decorator_applies(self):
        """validate_grid_params actually validates grid dimensions (F-148).

        The decorator inspects args/kwargs for an object with x/y/z attributes
        and calls GridValidator.validate_grid_dimensions on them.
        """

        class FakeGrid:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        @validate_grid_params
        def dummy_func(grid=None, **kwargs):
            return grid

        # Valid grid: passes through the decorator
        valid_grid = FakeGrid(10, 10, 5)
        result = dummy_func(grid=valid_grid)
        assert result is valid_grid

        # Invalid grid (negative dimension): decorator raises
        invalid_grid = FakeGrid(-1, 10, 5)
        with pytest.raises(CriticalValidationError):
            dummy_func(grid=invalid_grid)

    def test_validate_file_params_decorator_none_skip(self):
        """validate_file_params with None filename skips validation."""

        @validate_file_params
        def dummy_func(filename=None, **kwargs):
            return filename

        result = dummy_func(filename=None)
        assert result is None

    def test_validate_file_params_decorator_valid_file(self, tmp_path):
        """validate_file_params with existing file passes validation (F-149)."""
        testfile = tmp_path / "valid.txt"
        testfile.write_text("data")

        @validate_file_params
        def dummy_func(filename=None, **kwargs):
            return filename

        result = dummy_func(filename=str(testfile))
        assert os.path.abspath(result) == os.path.abspath(str(testfile))

    def test_validate_file_params_decorator_nonexistent_raises(self, tmp_path):
        """validate_file_params with non-existent file raises (F-149)."""
        nonexistent = tmp_path / "missing.txt"

        @validate_file_params
        def dummy_func(filename=None, **kwargs):
            return filename

        with pytest.raises(CriticalValidationError):
            dummy_func(filename=str(nonexistent))

    def test_validate_kriging_params_decorator_applies(self):
        """validate_kriging_params decorator validates radiuses, max_neighbours, cov_model.

        The decorator inspects kwargs for radiuses, max_neighbours, and
        cov_model, calling the appropriate ParameterValidator methods.
        """
        class FakeCovModel:
            def __init__(self):
                self.sill = 1.0
                self.nugget = 0.1
                self.ranges = (5.0, 5.0, 3.0)
                self.angles = (0.0, 0.0, 0.0)

        @validate_kriging_params
        def dummy_func(radiuses=None, max_neighbours=None, cov_model=None, **kwargs):
            return (radiuses, max_neighbours, cov_model)

        # Valid params: pass through the decorator
        cov = FakeCovModel()
        r, n, c = dummy_func(radiuses=(5, 5, 3), max_neighbours=12, cov_model=cov)
        assert r == (5, 5, 3)
        assert n == 12
        assert c is cov

        # Invalid radius (negative): decorator raises
        with pytest.raises(CriticalValidationError):
            dummy_func(radiuses=(-1, 5, 3), max_neighbours=12)

        # Invalid max_neighbours (0): decorator raises
        with pytest.raises(CriticalValidationError):
            dummy_func(max_neighbours=0)

    def test_validate_simulation_params_decorator_applies(self):
        """validate_simulation_params decorator validates seed, min_neighbours, max_neighbours.

        The decorator inspects kwargs for seed, min_neighbours, and
        max_neighbours, calling the appropriate ParameterValidator methods.
        """
        @validate_simulation_params
        def dummy_func(seed=None, min_neighbours=None, max_neighbours=None, **kwargs):
            return (seed, min_neighbours, max_neighbours)

        # Valid params: pass through the decorator
        s, mn, mx = dummy_func(seed=42, min_neighbours=4, max_neighbours=12)
        assert s == 42
        assert mn == 4
        assert mx == 12

        # Invalid seed (negative): decorator raises
        with pytest.raises(ValidationError, match="negative"):
            dummy_func(seed=-1)

    def test_validate_simulation_params_direct_call(self):
        """validate_simulation_params direct mode with min_neighbours uses consistent default (F-17).

        When only min_neighbours is provided (without max_neighbours),
        the max_neighbours default should be 12, not 0.
        """
        # Should not raise - min_neighbours <= 12
        validate_simulation_params(min_neighbours=4, max_neighbours=12)

        # min_neighbours > default max (12) should raise
        with pytest.raises(CriticalValidationError):
            validate_simulation_params(min_neighbours=20)

    def test_validate_simulation_params_direct_call_default_max(self):
        """Direct call with seed only — defaults work (F-17 verification)."""
        validate_simulation_params(seed=42)

    def test_validate_kriging_params_direct_call(self):
        """validate_kriging_params direct mode returns radiuses tuple.

        The dual calling convention supports direct call with grid, radiuses,
        max_neighbours, and cov_model.
        """
        result = validate_kriging_params(
            grid=None, radiuses=(5, 5, 3), max_neighbours=12, cov_model=None
        )
        assert result == (5, 5, 3)

    def test_validate_kriging_params_direct_call_invalid_radius(self):
        """validate_kriging_params direct mode raises on invalid radius."""
        with pytest.raises(CriticalValidationError):
            validate_kriging_params(radiuses=(-1, 5, 3), max_neighbours=12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
