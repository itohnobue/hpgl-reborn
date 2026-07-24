"""Tests for HPGL validation coverage gaps — M-P-36, M-P-37, M-P-38, M-P-39.

Covers:
- M-P-36: validate_grid_size_param direct call
- M-P-37: Non-integer radius (5.5, 1.3) → ValueError
- M-P-38: NaN/Inf in validate_probability
- M-P-39: NaN/Inf in validate_probability_sum
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.validation import (
        CriticalValidationError,
        GridValidator,
        ParameterValidator,
    )
    VALIDATION_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError):
    VALIDATION_AVAILABLE = False


# =============================================================================
# M-P-36: validate_grid_size_param direct tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestValidateGridSizeParam:
    """Direct test coverage for validate_grid_size_param (M-P-36)."""

    def test_valid_3tuple_passes(self):
        """3-element tuple passes validation."""
        GridValidator.validate_grid_size_param((10, 10, 5))
        GridValidator.validate_grid_size_param([3, 4, 2])

    def test_scalar_int_passes(self):
        """Scalar int passes (downstream caller enforces caps)."""
        GridValidator.validate_grid_size_param(42, "test_func")
        GridValidator.validate_grid_size_param(1)

    def test_invalid_3tuple_raises_on_validate(self):
        """3-tuple with negative dimension raises via validate_grid_dimensions."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_size_param((-1, 10, 5))

    def test_invalid_3tuple_zero_raises(self):
        """3-tuple with zero dimension raises."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_size_param((0, 10, 5))

    def test_non_tuple_non_int_skips_validation(self):
        """Non-3-tuple, non-int values skip validation (no error)."""
        # A string or float that is not a 3-length tuple/list → skip
        GridValidator.validate_grid_size_param("not_grid")


# =============================================================================
# M-P-37: Non-integer radius tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestNonIntegerRadius:
    """Test non-integer radius raises CriticalValidationError (M-P-37)."""

    def test_float_radius_55_raises(self):
        """Radius 5.5 is not an integer — raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius(5.5, "radius")

    def test_float_radius_13_raises(self):
        """Radius 1.3 is not an integer — raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius(1.3)

    def test_tuple_with_float_55_raises(self):
        """Tuple radius with non-integer value 5.5 raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius((5, 5.5, 3))

    def test_tuple_with_float_13_raises(self):
        """Tuple radius with non-integer value 1.3 raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius((1.3, 2, 3))

    def test_numpy_float_radius_55_raises(self):
        """Numpy float64 5.5 radius raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius(np.float64(5.5))

    def test_integer_as_float_passes(self):
        """Radius 5.0 (integer as float) passes."""
        result = ParameterValidator.validate_radius(5.0)
        assert result == (5, 5, 5)

    def test_tuple_with_mixed_int_float_passes(self):
        """Tuple with all integer-valued numbers passes."""
        result = ParameterValidator.validate_radius((1.0, 2.0, 3.0))
        assert result == (1, 2, 3)


# =============================================================================
# M-P-38: NaN/Inf in validate_probability
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestNanInfInValidateProbability:
    """Test NaN/Inf in validate_probability (M-P-38)."""

    def test_nan_probability_raises(self):
        """NaN probability raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(float("nan"), "prob")

    def test_inf_probability_raises(self):
        """Inf probability raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(float("inf"), "prob")

    def test_negative_inf_probability_raises(self):
        """-Inf probability raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(-float("inf"), "prob")

    def test_numpy_nan_probability_raises(self):
        """NumPy NaN probability raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(np.float64(float("nan")))

    def test_numpy_inf_probability_raises(self):
        """NumPy Inf probability raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(np.float32(float("inf")))


# =============================================================================
# M-P-39: NaN/Inf in validate_probability_sum
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestNanInfInValidateProbabilitySum:
    """Test NaN/Inf in validate_probability_sum (M-P-39)."""

    def test_nan_in_probability_list_raises(self):
        """NaN in probability list raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([0.5, float("nan"), 0.5])

    def test_inf_in_probability_list_raises(self):
        """Inf in probability list raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([0.3, float("inf"), 0.7])

    def test_negative_inf_in_probability_list_raises(self):
        """-Inf in probability list raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([-float("inf"), 1.0])

    def test_all_nan_list_raises(self):
        """All-NaN list raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([float("nan"), float("nan")])

    def test_single_inf_raises(self):
        """Single Inf value raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([float("inf")])
