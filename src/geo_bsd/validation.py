"""
Input Validation Framework for HPGL Python Interface
Addresses vulnerability IV-001 (CVSS 7.5) - Insufficient Input Validation

This module provides comprehensive input validation for all HPGL Python functions.
"""

import os
import pathlib
import numpy
import logging
from typing import Union, Tuple, Optional, List, Any, Dict
from functools import wraps

# Configure validation logger
validation_logger = logging.getLogger('hpgl.validation')


# ============================================================================
# Validation Constants
# ============================================================================

class ValidationConstants:
    """Constants for validation limits"""

    # Grid dimension limits
    MIN_GRID_DIMENSION = 1
    MAX_GRID_DIMENSION = 10000000
    MAX_GRID_SIZE = 1000000000  # 1 billion cells

    # Neighbor count limits
    MIN_NEIGHBORS = 1
    MAX_NEIGHBORS = 1000
    DEFAULT_MAX_NEIGHBORS = 12

    # Radius limits
    MIN_RADIUS = 0.0
    MAX_RADIUS = 1000000.0

    # Covariance parameter limits
    MIN_SILL = 0.0
    MAX_SILL = 1e10
    MIN_NUGGET = 0.0
    MAX_NUGGET = 1e10
    MIN_RANGE = 0.0
    MAX_RANGE = 1e10

    # Angle limits (in degrees)
    MIN_ANGLE = 0.0
    MAX_ANGLE = 360.0

    # Probability limits
    MIN_PROBABILITY = 0.0
    MAX_PROBABILITY = 1.0
    PROBABILITY_SUM_TOLERANCE = 0.01  # Allow 1% tolerance for floating point errors

    # Indicator limits
    MAX_INDICATORS = 256

    # Seed limits
    MIN_SEED = 0


# ============================================================================
# Validation Exceptions
# ============================================================================

class ValidationError(Exception):
    """Base class for validation errors"""
    def __init__(self, message: str, parameter_name: str = "", severity: str = "error"):
        self.message = message
        self.parameter_name = parameter_name
        self.severity = severity
        if parameter_name:
            super().__init__(f"{parameter_name}: {message}")
        else:
            super().__init__(message)


class CriticalValidationError(ValidationError):
    """Critical validation error that prevents operation"""
    def __init__(self, message: str, parameter_name: str = ""):
        super().__init__(message, parameter_name, "critical")


class ValidationWarning(ValidationError):
    """Validation warning that doesn't prevent operation"""
    def __init__(self, message: str, parameter_name: str = ""):
        super().__init__(message, parameter_name, "warning")


# ============================================================================
# Path Validation
# ============================================================================

class PathValidator:
    """Validates file paths to prevent directory traversal attacks"""

    @staticmethod
    def validate_filepath(
        filename: Union[str, pathlib.Path],
        must_exist: bool = False,
        allow_directories: bool = False,
        allowed_extensions: Optional[List[str]] = None
    ) -> str:
        """
        Validates and sanitizes file paths to prevent directory traversal attacks.

        Args:
            filename: The file path to validate
            must_exist: Whether the file must exist (for read operations)
            allow_directories: Whether to allow directory paths
            allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.data'])

        Returns:
            Absolute, normalized path as string

        Raises:
            CriticalValidationError: If path contains directory traversal attempts
                                    or points outside allowed directories
        """
        if not filename:
            raise CriticalValidationError("Filename cannot be empty", "filename")

        # Convert to Path object for robust handling
        path = pathlib.Path(filename)

        # Check for path traversal attempts in the original string
        # This catches cases where resolve() might behave unexpectedly
        path_str = os.path.normpath(str(filename))
        if '..' in path_str.split(os.sep) or '../' in str(filename) or '..\\' in str(filename):
            raise CriticalValidationError(
                f"Path traversal detected in filename: {filename}",
                "filename"
            )

        # Resolve to absolute path and normalize (removes ../ segments)
        try:
            resolved_path = path.resolve(strict=must_exist)
        except (OSError, RuntimeError) as e:
            if must_exist:
                raise CriticalValidationError(
                    f"File does not exist: {filename}",
                    "filename"
                ) from e
            # For non-existent files, resolve without strict check
            try:
                resolved_path = path.resolve()
            except Exception:
                raise CriticalValidationError(
                    f"Invalid path: {filename}",
                    "filename"
                ) from e

        # Check extension if specified
        if allowed_extensions is not None:
            if resolved_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise CriticalValidationError(
                    f"File extension '{resolved_path.suffix}' not allowed. "
                    f"Allowed extensions: {allowed_extensions}",
                    "filename"
                )

        return str(resolved_path)

    @staticmethod
    def validate_write_filepath(filename: Union[str, pathlib.Path]) -> str:
        """
        Validates a file path for writing operations.

        Args:
            filename: The file path to validate

        Returns:
            Absolute, normalized path as string

        Raises:
            CriticalValidationError: If path is invalid
        """
        return PathValidator.validate_filepath(
            filename,
            must_exist=False,
            allow_directories=False
        )


# ============================================================================
# Grid and Array Validation
# ============================================================================

class GridValidator:
    """Validates grid dimensions and array sizes"""

    @staticmethod
    def validate_grid_dimensions(x: int, y: int, z: int) -> None:
        """
        Validates grid dimensions.

        Args:
            x: Grid X dimension
            y: Grid Y dimension
            z: Grid Z dimension

        Raises:
            CriticalValidationError: If dimensions are invalid
        """
        if x < ValidationConstants.MIN_GRID_DIMENSION or x > ValidationConstants.MAX_GRID_DIMENSION:
            raise CriticalValidationError(
                f"Grid X dimension {x} outside valid range "
                f"[{ValidationConstants.MIN_GRID_DIMENSION}, {ValidationConstants.MAX_GRID_DIMENSION}]",
                "grid_x"
            )

        if y < ValidationConstants.MIN_GRID_DIMENSION or y > ValidationConstants.MAX_GRID_DIMENSION:
            raise CriticalValidationError(
                f"Grid Y dimension {y} outside valid range "
                f"[{ValidationConstants.MIN_GRID_DIMENSION}, {ValidationConstants.MAX_GRID_DIMENSION}]",
                "grid_y"
            )

        if z < ValidationConstants.MIN_GRID_DIMENSION or z > ValidationConstants.MAX_GRID_DIMENSION:
            raise CriticalValidationError(
                f"Grid Z dimension {z} outside valid range "
                f"[{ValidationConstants.MIN_GRID_DIMENSION}, {ValidationConstants.MAX_GRID_DIMENSION}]",
                "grid_z"
            )

        # Check total grid size
        total_size = x * y * z
        if total_size > ValidationConstants.MAX_GRID_SIZE:
            raise CriticalValidationError(
                f"Total grid size {total_size} exceeds maximum of {ValidationConstants.MAX_GRID_SIZE}",
                "grid_size"
            )

    @staticmethod
    def validate_array_size(array: numpy.ndarray, grid: Tuple[int, int, int]) -> None:
        """
        Validates that array size matches grid dimensions.

        Args:
            array: NumPy array to validate
            grid: Tuple of (x, y, z) grid dimensions

        Raises:
            CriticalValidationError: If array size doesn't match grid
        """
        expected_size = grid[0] * grid[1] * grid[2]
        actual_size = array.size

        if actual_size != expected_size:
            raise CriticalValidationError(
                f"Array size {actual_size} does not match grid size {expected_size}",
                "array_size"
            )

    @staticmethod
    def validate_array_dtype(array: numpy.ndarray, expected_dtype: numpy.dtype) -> None:
        """
        Validates array data type.

        Args:
            array: NumPy array to validate
            expected_dtype: Expected data type

        Raises:
            CriticalValidationError: If array has wrong data type
        """
        if array.dtype != expected_dtype:
            raise CriticalValidationError(
                f"Array has dtype {array.dtype}, expected {expected_dtype}",
                "array_dtype"
            )


# ============================================================================
# Parameter Validation
# ============================================================================

class ParameterValidator:
    """Validates numerical parameters for geostatistical operations"""

    @staticmethod
    def validate_radius(radius: Union[float, int, Tuple], name: str = "radius") -> Tuple[float, float, float]:
        """
        Validates radius parameters.

        Args:
            radius: Single value or tuple of (rx, ry, rz)
            name: Parameter name for error messages

        Returns:
            Tuple of (rx, ry, rz) - preserves int type when input is int

        Raises:
            CriticalValidationError: If radius is invalid
        """
        # Track if input was integer to preserve type
        is_int_input = isinstance(radius, int) or (
            isinstance(radius, (tuple, list)) and
            len(radius) == 3 and
            all(isinstance(r, int) for r in radius)
        )

        if isinstance(radius, (int, float)):
            rx = ry = rz = float(radius) if not isinstance(radius, int) else float(radius)
        elif isinstance(radius, (tuple, list)) and len(radius) == 3:
            rx, ry, rz = map(float, radius)
        else:
            raise CriticalValidationError(
                f"Radius must be a number or tuple of 3 numbers, got {type(radius)}",
                name
            )

        for i, r in enumerate((rx, ry, rz)):
            if numpy.isnan(r) or numpy.isinf(r):
                raise CriticalValidationError(
                    f"{name}[{i}] is NaN or infinite",
                    name
                )

        # Convert back to int if input was int and values are whole numbers
        if is_int_input:
            rx = int(rx) if rx.is_integer() else rx
            ry = int(ry) if ry.is_integer() else ry
            rz = int(rz) if rz.is_integer() else rz

        for i, r in enumerate((rx, ry, rz)):
            r_val = float(r)
            if r_val < ValidationConstants.MIN_RADIUS:
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} is less than minimum {ValidationConstants.MIN_RADIUS}",
                    name
                )
            if r_val > ValidationConstants.MAX_RADIUS:
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} exceeds maximum {ValidationConstants.MAX_RADIUS}",
                    name
                )

        return (rx, ry, rz)

    @staticmethod
    def validate_max_neighbors(max_neighbors: int) -> None:
        """
        Validates maximum number of neighbors.

        Args:
            max_neighbors: Maximum number of neighbors

        Raises:
            CriticalValidationError: If max_neighbors is invalid
            ValidationWarning: If max_neighbors is unusually large
        """
        if max_neighbors < ValidationConstants.MIN_NEIGHBORS:
            raise CriticalValidationError(
                f"Max neighbors {max_neighbors} is less than minimum {ValidationConstants.MIN_NEIGHBORS}",
                "max_neighbors"
            )

        if max_neighbors > ValidationConstants.MAX_NEIGHBORS:
            raise ValidationWarning(
                f"Max neighbors {max_neighbors} exceeds recommended maximum {ValidationConstants.MAX_NEIGHBORS}. "
                "Performance may be degraded.",
                "max_neighbors"
            )
            validation_logger.warning(
                f"Max neighbors {max_neighbors} exceeds recommended maximum "
                f"{ValidationConstants.MAX_NEIGHBORS}"
            )

    @staticmethod
    def validate_min_neighbors(min_neighbors: int, max_neighbors: int) -> None:
        """
        Validates minimum number of neighbors.

        Args:
            min_neighbors: Minimum number of neighbors
            max_neighbors: Maximum number of neighbors

        Raises:
            CriticalValidationError: If min_neighbors is invalid
        """
        if min_neighbors > max_neighbors:
            raise CriticalValidationError(
                f"Min neighbors {min_neighbors} exceeds max neighbors {max_neighbors}",
                "min_neighbors"
            )

        if min_neighbors < 0:
            raise CriticalValidationError(
                f"Min neighbors {min_neighbors} is negative",
                "min_neighbors"
            )

    @staticmethod
    def validate_covariance_parameters(
        sill: float,
        nugget: float,
        ranges: Optional[Tuple] = None,
        angles: Optional[Tuple] = None
    ) -> None:
        """
        Validates covariance model parameters.

        Args:
            sill: Sill value
            nugget: Nugget value
            ranges: Optional tuple of (range1, range2, range3)
            angles: Optional tuple of (angle1, angle2, angle3)

        Raises:
            CriticalValidationError: If parameters are invalid
        """
        # Validate sill
        if numpy.isnan(sill) or numpy.isinf(sill):
            raise CriticalValidationError("Sill is NaN or infinite", "sill")

        if sill < ValidationConstants.MIN_SILL:
            raise CriticalValidationError(
                f"Sill {sill} is less than minimum {ValidationConstants.MIN_SILL}",
                "sill"
            )

        if sill > ValidationConstants.MAX_SILL:
            raise CriticalValidationError(
                f"Sill {sill} exceeds maximum {ValidationConstants.MAX_SILL}",
                "sill"
            )

        # Validate nugget
        if numpy.isnan(nugget) or numpy.isinf(nugget):
            raise CriticalValidationError("Nugget is NaN or infinite", "nugget")

        if nugget < ValidationConstants.MIN_NUGGET:
            raise CriticalValidationError(
                f"Nugget {nugget} is less than minimum {ValidationConstants.MIN_NUGGET}",
                "nugget"
            )

        if nugget > ValidationConstants.MAX_NUGGET:
            raise CriticalValidationError(
                f"Nugget {nugget} exceeds maximum {ValidationConstants.MAX_NUGGET}",
                "nugget"
            )

        # Critical: Nugget should not exceed sill
        if nugget > sill:
            raise CriticalValidationError(
                f"Nugget {nugget} exceeds sill {sill} (nugget must be <= sill)",
                "nugget"
            )

        # Validate ranges if provided
        if ranges is not None:
            if len(ranges) != 3:
                raise CriticalValidationError(
                    f"Ranges must have 3 values, got {len(ranges)}",
                    "ranges"
                )

            for i, r in enumerate(ranges):
                if numpy.isnan(r) or numpy.isinf(r):
                    raise CriticalValidationError(
                        f"Range[{i}] is NaN or infinite",
                        "ranges"
                    )
                if r < ValidationConstants.MIN_RANGE:
                    raise CriticalValidationError(
                        f"Range[{i}] = {r} is less than minimum {ValidationConstants.MIN_RANGE}",
                        "ranges"
                    )
                if r > ValidationConstants.MAX_RANGE:
                    raise CriticalValidationError(
                        f"Range[{i}] = {r} exceeds maximum {ValidationConstants.MAX_RANGE}",
                        "ranges"
                    )

        # Validate angles if provided
        if angles is not None:
            if len(angles) != 3:
                raise CriticalValidationError(
                    f"Angles must have 3 values, got {len(angles)}",
                    "angles"
                )

            for i, a in enumerate(angles):
                if numpy.isnan(a) or numpy.isinf(a):
                    raise CriticalValidationError(
                        f"Angle[{i}] is NaN or infinite",
                        "angles"
                    )
                # Warn if angle is outside typical range
                if a < ValidationConstants.MIN_ANGLE or a > ValidationConstants.MAX_ANGLE:
                    validation_logger.warning(
                        f"Angle[{i}] = {a} is outside typical range "
                        f"[{ValidationConstants.MIN_ANGLE}, {ValidationConstants.MAX_ANGLE}]"
                    )

    @staticmethod
    def validate_probability(prob: float, name: str = "probability") -> None:
        """
        Validates a probability value.

        Args:
            prob: Probability value
            name: Parameter name for error messages

        Raises:
            CriticalValidationError: If probability is invalid
        """
        if numpy.isnan(prob) or numpy.isinf(prob):
            raise CriticalValidationError(f"{name} is NaN or infinite", name)

        if prob < ValidationConstants.MIN_PROBABILITY or prob > ValidationConstants.MAX_PROBABILITY:
            raise CriticalValidationError(
                f"{name} = {prob} outside valid range "
                f"[{ValidationConstants.MIN_PROBABILITY}, {ValidationConstants.MAX_PROBABILITY}]",
                name
            )

    @staticmethod
    def validate_probability_sum(probs: List[float]) -> None:
        """
        Validates that probabilities sum to approximately 1.0.

        Args:
            probs: List of probability values

        Raises:
            CriticalValidationError: If probabilities don't sum to 1.0
        """
        prob_sum = sum(probs)

        if numpy.isnan(prob_sum) or numpy.isinf(prob_sum):
            raise CriticalValidationError("Probability sum is NaN or infinite", "probabilities")

        diff = abs(prob_sum - 1.0)
        if diff > ValidationConstants.PROBABILITY_SUM_TOLERANCE:
            raise CriticalValidationError(
                f"Probabilities sum to {prob_sum}, expected 1.0 (difference: {diff})",
                "probabilities"
            )

    @staticmethod
    def validate_seed(seed: int) -> None:
        """
        Validates seed value for random number generation.

        Args:
            seed: Seed value

        Raises:
            ValidationWarning: If seed is negative (unusual but not necessarily wrong)
        """
        if seed < ValidationConstants.MIN_SEED:
            validation_logger.warning(f"Seed value {seed} is negative")

    @staticmethod
    def validate_indicator_count(count: int) -> None:
        """
        Validates indicator count.

        Args:
            count: Number of indicators

        Raises:
            CriticalValidationError: If count is invalid
        """
        if count <= 0:
            raise CriticalValidationError(
                f"Indicator count must be positive, got {count}",
                "indicator_count"
            )

        if count > ValidationConstants.MAX_INDICATORS:
            raise CriticalValidationError(
                f"Indicator count {count} exceeds maximum {ValidationConstants.MAX_INDICATORS}",
                "indicator_count"
            )

    @staticmethod
    def validate_correlation_coef(coef, name: str = "correlation_coef") -> None:
        """
        Validates correlation coefficient is in [-1, 1] range.

        Args:
            coef: Correlation coefficient value
            name: Parameter name for error messages

        Raises:
            CriticalValidationError: If coefficient is out of range or invalid
        """
        import math
        if not isinstance(coef, (int, float)):
            raise CriticalValidationError(
                f"{name} must be a number, got {type(coef).__name__}",
                name
            )
        if math.isnan(coef) or math.isinf(coef):
            raise CriticalValidationError(
                f"{name} must be finite, got {coef}",
                name
            )
        if coef < -1.0 or coef > 1.0:
            raise CriticalValidationError(
                f"{name} must be in [-1, 1] range, got {coef}",
                name
            )

    @staticmethod
    def validate_variance(variance, name: str = "variance") -> None:
        """
        Validates variance is non-negative and finite.

        Args:
            variance: Variance value
            name: Parameter name for error messages

        Raises:
            CriticalValidationError: If variance is negative or invalid
        """
        import math
        if not isinstance(variance, (int, float)):
            raise CriticalValidationError(
                f"{name} must be a number, got {type(variance).__name__}",
                name
            )
        if math.isnan(variance) or math.isinf(variance):
            raise CriticalValidationError(
                f"{name} must be finite, got {variance}",
                name
            )
        if variance < 0:
            raise CriticalValidationError(
                f"{name} must be non-negative, got {variance}",
                name
            )


# ============================================================================
# Decorators for Function Validation
# ============================================================================

def validate_grid_params(func):
    """Decorator to validate grid parameters"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find grid parameter
        grid = None
        if 'grid' in kwargs:
            grid = kwargs['grid']
        else:
            # Try to find grid in positional arguments
            for arg in args:
                if hasattr(arg, 'x') and hasattr(arg, 'y') and hasattr(arg, 'z'):
                    grid = arg
                    break

        if grid is not None:
            GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

        return func(*args, **kwargs)
    return wrapper


def validate_kriging_params(func):
    """Decorator to validate kriging parameters"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate radiuses if provided
        if 'radiuses' in kwargs:
            ParameterValidator.validate_radius(kwargs['radiuses'], 'radiuses')

        # Validate max_neighbours if provided
        if 'max_neighbours' in kwargs or 'max_neighbors' in kwargs:
            max_neigh = kwargs.get('max_neighbours', kwargs.get('max_neighbors'))
            if max_neigh is not None:
                ParameterValidator.validate_max_neighbors(max_neigh)

        # Validate covariance model if provided
        if 'cov_model' in kwargs:
            cov_model = kwargs['cov_model']
            ParameterValidator.validate_covariance_parameters(
                cov_model.sill,
                cov_model.nugget,
                cov_model.ranges,
                cov_model.angles
            )

        return func(*args, **kwargs)
    return wrapper


def validate_simulation_params(func):
    """Decorator to validate simulation parameters"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate seed if provided
        if 'seed' in kwargs:
            ParameterValidator.validate_seed(kwargs['seed'])

        # Validate min_neighbours if provided
        if 'min_neighbours' in kwargs or 'min_neighbors' in kwargs:
            min_neigh = kwargs.get('min_neighbours', kwargs.get('min_neighbors'))
            max_neigh = kwargs.get('max_neighbours', kwargs.get('max_neighbors', 12))
            if min_neigh is not None:
                ParameterValidator.validate_min_neighbors(min_neigh, max_neigh)

        return func(*args, **kwargs)
    return wrapper


def validate_file_params(func):
    """Decorator to validate file parameters"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate filename for reading
        if 'filename' in kwargs:
            filename = kwargs['filename']
            if filename is not None:
                PathValidator.validate_filepath(filename, must_exist=True)

        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# Validation Context Manager
# ============================================================================

class ValidationContext:
    """
    Context manager for collecting validation results.

    Usage:
        with ValidationContext() as validator:
            validator.validate_grid_dimensions(x, y, z)
            validator.validate_radius(radius)
            # If any validation fails, exception is raised on exit
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, raise exception on first error. If False, collect all errors.
        """
        self.strict = strict
        self.errors = []
        self.warnings = []

    def validate_grid_dimensions(self, x: int, y: int, z: int) -> None:
        """Validate grid dimensions and record results"""
        try:
            GridValidator.validate_grid_dimensions(x, y, z)
        except ValidationError as e:
            if self.strict:
                raise
            self.errors.append(e)

    def validate_radius(self, radius: Union[float, int, Tuple], name: str = "radius") -> Tuple[float, float, float]:
        """Validate radius and record results"""
        try:
            return ParameterValidator.validate_radius(radius, name)
        except ValidationError as e:
            if self.strict:
                raise
            self.errors.append(e)
            return (0.0, 0.0, 0.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.errors and not self.strict:
            raise CriticalValidationError(
                f"Validation failed with {len(self.errors)} error(s)",
                ""
            )
        return False


# Export all public classes and functions
__all__ = [
    'ValidationConstants',
    'ValidationError',
    'CriticalValidationError',
    'ValidationWarning',
    'PathValidator',
    'GridValidator',
    'ParameterValidator',
    'validate_grid_params',
    'validate_kriging_params',
    'validate_simulation_params',
    'validate_file_params',
    'ValidationContext',
]
