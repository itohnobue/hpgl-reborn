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

    def test_non_3tuple_scalar_or_string_skips_validation(self):
        """Scalar int / string / non-3-length values skip validation (no error).

        Both the scalar-int (:43) and string (:58) variants hit the same
        ``isinstance(size, (tuple, list)) and len == 3`` no-op branch — merged.
        """
        GridValidator.validate_grid_size_param(42, "test_func")
        GridValidator.validate_grid_size_param(1)
        GridValidator.validate_grid_size_param("not_grid")

    def test_invalid_3tuple_raises_on_validate(self):
        """3-tuple with negative dimension raises via validate_grid_dimensions."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_size_param((-1, 10, 5))

    def test_invalid_3tuple_zero_raises(self):
        """3-tuple with zero dimension raises."""
        with pytest.raises(CriticalValidationError):
            GridValidator.validate_grid_size_param((0, 10, 5))


# =============================================================================
# M-P-37: Non-integer radius tests
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestNonIntegerRadius:
    """Test non-integer radius raises CriticalValidationError (M-P-37).

    The 5 raise-variants hit the same ``not float(r).is_integer()`` branch
    (validation.py:679) — consolidated to 3 shape variants (scalar, tuple,
    numpy) plus the 2 discriminating pass-tests.
    """

    @pytest.mark.parametrize("bad_radius", [5.5, 1.3])
    def test_scalar_float_radius_raises(self, bad_radius):
        """Scalar non-integer radius raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius(bad_radius, "radius")

    @pytest.mark.parametrize("bad_tuple", [(5, 5.5, 3), (1.3, 2, 3)])
    def test_tuple_float_radius_raises(self, bad_tuple):
        """Tuple radius with non-integer value raises."""
        with pytest.raises(CriticalValidationError, match="not an integer"):
            ParameterValidator.validate_radius(bad_tuple)

    def test_numpy_float_radius_raises(self):
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
    """Test NaN/Inf in validate_probability (M-P-38).

    nan/inf/-inf/np.nan/np.inf all hit the single
    ``numpy.isnan(prob) or numpy.isinf(prob)`` branch (validation.py:891-892)
    — consolidated to one parametrized scalar case + one numpy variant.
    """

    @pytest.mark.parametrize("bad_prob", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_probability_raises(self, bad_prob):
        """Non-finite probability raises CriticalValidationError."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(bad_prob, "prob")

    def test_numpy_non_finite_probability_raises(self):
        """NumPy floating non-finite probability raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability(np.float64(float("nan")))


# =============================================================================
# M-P-39: NaN/Inf in validate_probability_sum
# =============================================================================


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation module not available")
class TestNanInfInValidateProbabilitySum:
    """Test NaN/Inf in validate_probability_sum (M-P-39).

    nan/inf/-inf/all-nan/single-inf all hit the same per-element
    ``numpy.isnan or numpy.isinf`` check (validation.py:914-915) —
    consolidated to one parametrized list-family + one all-NaN case.
    """

    @pytest.mark.parametrize(
        "bad_list",
        [
            [0.5, float("nan"), 0.5],
            [0.3, float("inf"), 0.7],
            [-float("inf"), 1.0],
            [float("inf")],
        ],
    )
    def test_non_finite_in_probability_list_raises(self, bad_list):
        """Non-finite value in probability list raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum(bad_list)

    def test_all_nan_list_raises(self):
        """All-NaN list raises."""
        with pytest.raises(CriticalValidationError, match="NaN or infinite"):
            ParameterValidator.validate_probability_sum([float("nan"), float("nan")])
