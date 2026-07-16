# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""
Input Validation Framework for HPGL Python Interface
Addresses vulnerability IV-001 (CVSS 7.5) - Insufficient Input Validation

This module provides comprehensive input validation for all HPGL Python functions.
"""

from __future__ import annotations

import logging
import os
import pathlib
import warnings
from functools import wraps

import numpy

# Configure validation logger
validation_logger = logging.getLogger("hpgl.validation")


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
    MAX_INDICATORS = 255

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

    # Trusted base directory that callers SHOULD use instead of deriving
    # basedir from the filename (self-referential basedir defeats symlink
    # containment). Defaults to cwd realpath at import time.
    DEFAULT_BASE_DIR: str = os.path.realpath(os.getcwd())

    @staticmethod
    def validate_filepath(
        filename: str | pathlib.Path,
        must_exist: bool = False,
        allow_directories: bool = False,
        allowed_extensions: list[str] | None = None,
        basedir: str | pathlib.Path | None = None,
    ) -> str:
        """
        Validates and sanitizes file paths to prevent directory traversal attacks.

        Args:
            filename: The file path to validate
            must_exist: Whether the file must exist (for read operations)
            allow_directories: Whether to allow directory paths
            allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.data'])
            basedir: Optional base directory to restrict resolved paths within.
                     If provided, the resolved path must be a child of this directory.

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

        # Check for path traversal attempts in the ORIGINAL (unnormalized) string
        # BEFORE normalization, since normpath removes '..' components.
        # Split the original path string by the OS separator to detect bare '..'
        # even without trailing slashes.
        path_str = str(filename)
        parts = path_str.replace("\\", "/").split("/")
        if ".." in parts:
            raise CriticalValidationError(
                f"Path traversal detected in filename: {filename}", "filename"
            )

        # Resolve to absolute path and normalize (removes ../ segments)
        try:
            resolved_path = path.resolve(strict=must_exist)
        except (OSError, RuntimeError) as e:
            if must_exist:
                raise CriticalValidationError(f"File does not exist: {filename}", "filename") from e
            # For non-existent files, resolve without strict check
            try:
                resolved_path = path.resolve()
            except (OSError, RuntimeError, ValueError) as e2:
                raise CriticalValidationError(f"Invalid path: {filename}", "filename") from e2

        # If basedir is specified, verify the resolved path is within it.
        # This prevents symlink-based escapes (e.g., /tmp/link → /etc).
        if basedir is not None:
            basedir_resolved = pathlib.Path(basedir).resolve()
            try:
                resolved_path.relative_to(basedir_resolved)
            except ValueError as err:
                raise CriticalValidationError(
                    f"Path {resolved_path} is outside allowed base directory {basedir_resolved}",
                    "filename",
                ) from err

        # Check extension if specified
        if allowed_extensions is not None:
            if resolved_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise CriticalValidationError(
                    f"File extension '{resolved_path.suffix}' not allowed. "
                    f"Allowed extensions: {allowed_extensions}",
                    "filename",
                )

        # Check if path is a directory and directories are not allowed
        if not allow_directories and resolved_path.is_dir():
            raise CriticalValidationError(
                f"Path is a directory: {resolved_path}", "filename"
            )

        return str(resolved_path)

    @staticmethod
    def validate_write_filepath(filename: str | pathlib.Path) -> str:
        """
        Validates a file path for writing operations.

        Args:
            filename: The file path to validate

        Returns:
            Absolute, normalized path as string

        Raises:
            CriticalValidationError: If path is invalid
        """
        return PathValidator.validate_filepath(filename, must_exist=False, allow_directories=False)

    @staticmethod
    def validate_filepath_in_basedir(
        filename: str | pathlib.Path,
        basedir: str | pathlib.Path,
        must_exist: bool = False,
        allowed_extensions: list[str] | None = None,
    ) -> str:
        """
        Validates filepath with a REQUIRED basedir containment check.

        Wraps ``validate_filepath`` with the ``basedir`` parameter for callers
        that know the expected parent directory. The ``basedir`` MUST be provided;
        this is not optional like the base method.

        Args:
            filename: The file path to validate
            basedir: Base directory that the resolved path must be within
            must_exist: Whether the file must exist (for read operations)
            allowed_extensions: List of allowed file extensions

        Returns:
            Absolute, normalized path as string

        Raises:
            CriticalValidationError: If path is invalid or outside basedir
        """
        return PathValidator.validate_filepath(
            filename,
            must_exist=must_exist,
            allow_directories=False,
            allowed_extensions=allowed_extensions,
            basedir=basedir,
        )

    @staticmethod
    def safe_open_write(
        filename: str | pathlib.Path,
        basedir: str | pathlib.Path,
        encoding: str = "utf-8",
    ):
        """Validate path for writing and immediately open with O_NOFOLLOW.

        Validates the file path against a base directory, then opens it
        atomically with ``O_NOFOLLOW`` to prevent TOCTOU (symlink race)
        vulnerabilities. Returns a file object that must be closed by
        the caller (use as a context manager).

        Args:
            filename: The file path to validate and open for writing.
            basedir: Base directory for containment check (required).
            encoding: Text encoding for the opened file.

        Returns:
            A text-mode file object opened for writing.

        Raises:
            CriticalValidationError: If path is invalid or outside basedir.
            OSError: If the file cannot be opened.
        """
        safe_path = PathValidator.validate_filepath_in_basedir(
            filename, basedir=basedir, must_exist=False
        )
        fd = os.open(safe_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW)
        return os.fdopen(fd, "w", encoding=encoding)

    @staticmethod
    def safe_open_read(
        filename: str | pathlib.Path,
        basedir: str | pathlib.Path,
        encoding: str = "utf-8",
    ):
        """Validate path for reading and immediately open with O_NOFOLLOW.

        Validates the file path against a base directory, then opens it
        atomically with ``O_NOFOLLOW`` to prevent TOCTOU (symlink race)
        vulnerabilities. Returns a file object that must be closed by
        the caller (use as a context manager).

        Args:
            filename: The file path to validate and open for reading.
            basedir: Base directory for containment check (required).
            encoding: Text encoding for the opened file.

        Returns:
            A text-mode file object opened for reading.

        Raises:
            CriticalValidationError: If path is invalid or outside basedir.
            OSError: If the file cannot be opened.
        """
        safe_path = PathValidator.validate_filepath_in_basedir(
            filename, basedir=basedir, must_exist=True
        )
        fd = os.open(safe_path, os.O_RDONLY | os.O_NOFOLLOW)
        return os.fdopen(fd, "r", encoding=encoding)

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
                "grid_x",
            )

        if y < ValidationConstants.MIN_GRID_DIMENSION or y > ValidationConstants.MAX_GRID_DIMENSION:
            raise CriticalValidationError(
                f"Grid Y dimension {y} outside valid range "
                f"[{ValidationConstants.MIN_GRID_DIMENSION}, {ValidationConstants.MAX_GRID_DIMENSION}]",
                "grid_y",
            )

        if z < ValidationConstants.MIN_GRID_DIMENSION or z > ValidationConstants.MAX_GRID_DIMENSION:
            raise CriticalValidationError(
                f"Grid Z dimension {z} outside valid range "
                f"[{ValidationConstants.MIN_GRID_DIMENSION}, {ValidationConstants.MAX_GRID_DIMENSION}]",
                "grid_z",
            )

        # Check total grid size — cast to Python int to avoid numpy int32 overflow
        total_size = int(x) * int(y) * int(z)
        if total_size > ValidationConstants.MAX_GRID_SIZE:
            raise CriticalValidationError(
                f"Total grid size {total_size} exceeds maximum of {ValidationConstants.MAX_GRID_SIZE}",
                "grid_size",
            )

    @staticmethod
    def validate_array_size(array: numpy.ndarray, grid: tuple[int, int, int]) -> None:
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
                f"Array size {actual_size} does not match grid size {expected_size}", "array_size"
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
                f"Array has dtype {array.dtype}, expected {expected_dtype}", "array_dtype"
            )


# ============================================================================
# Parameter Validation
# ============================================================================


class ParameterValidator:
    """Validates numerical parameters for geostatistical operations"""

    @staticmethod
    def validate_radius(
        radius: float | int | tuple, name: str = "radius"
    ) -> tuple[float, float, float]:
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
        if isinstance(radius, (int, float, numpy.integer, numpy.floating)):
            vals = [float(radius)] * 3
        elif isinstance(radius, (tuple, list)) and len(radius) == 3:
            vals = list(map(float, radius))
        else:
            raise CriticalValidationError(
                f"Radius must be a number or tuple of 3 numbers, got {type(radius)}", name
            )

        for i, r in enumerate(vals):
            if numpy.isnan(r) or numpy.isinf(r):
                raise CriticalValidationError(f"{name}[{i}] is NaN or infinite", name)
            if r < ValidationConstants.MIN_RADIUS:
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} is less than minimum {ValidationConstants.MIN_RADIUS}", name
                )
            if r > ValidationConstants.MAX_RADIUS:
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} exceeds maximum {ValidationConstants.MAX_RADIUS}", name
                )
            if not float(r).is_integer():
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} is not an integer (radius must be a whole number of grid cells)",
                    name,
                )

        return (int(vals[0]), int(vals[1]), int(vals[2]))

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
                "max_neighbors",
            )

        if max_neighbors > ValidationConstants.MAX_NEIGHBORS:
            warnings.warn(
                f"Max neighbors {max_neighbors} exceeds recommended maximum {ValidationConstants.MAX_NEIGHBORS}. "
                "Performance may be degraded.",
                stacklevel=2,
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
                "min_neighbors",
            )

        if min_neighbors < 0:
            raise CriticalValidationError(
                f"Min neighbors {min_neighbors} is negative", "min_neighbors"
            )

    @staticmethod
    def validate_covariance_parameters(
        sill: float, nugget: float, ranges: tuple | None = None, angles: tuple | None = None
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
        if not isinstance(sill, (int, float, numpy.floating, numpy.integer)):
            raise CriticalValidationError(
                f"Sill must be a number, got {type(sill).__name__}", "sill"
            )
        if numpy.isnan(sill) or numpy.isinf(sill):
            raise CriticalValidationError("Sill is NaN or infinite", "sill")

        if sill < ValidationConstants.MIN_SILL:
            raise CriticalValidationError(
                f"Sill {sill} is less than minimum {ValidationConstants.MIN_SILL}", "sill"
            )

        if sill > ValidationConstants.MAX_SILL:
            raise CriticalValidationError(
                f"Sill {sill} exceeds maximum {ValidationConstants.MAX_SILL}", "sill"
            )

        # Validate nugget
        if not isinstance(nugget, (int, float, numpy.floating, numpy.integer)):
            raise CriticalValidationError(
                f"Nugget must be a number, got {type(nugget).__name__}", "nugget"
            )
        if numpy.isnan(nugget) or numpy.isinf(nugget):
            raise CriticalValidationError("Nugget is NaN or infinite", "nugget")

        if nugget < ValidationConstants.MIN_NUGGET:
            raise CriticalValidationError(
                f"Nugget {nugget} is less than minimum {ValidationConstants.MIN_NUGGET}", "nugget"
            )

        if nugget > ValidationConstants.MAX_NUGGET:
            raise CriticalValidationError(
                f"Nugget {nugget} exceeds maximum {ValidationConstants.MAX_NUGGET}", "nugget"
            )

        # Critical: Nugget should not exceed sill
        if nugget > sill:
            raise CriticalValidationError(
                f"Nugget {nugget} exceeds sill {sill} (nugget must be <= sill)", "nugget"
            )

        # Validate ranges if provided
        if ranges is not None:
            if len(ranges) != 3:
                raise CriticalValidationError(
                    f"Ranges must have 3 values, got {len(ranges)}", "ranges"
                )

            for i, r in enumerate(ranges):
                if numpy.isnan(r) or numpy.isinf(r):
                    raise CriticalValidationError(f"Range[{i}] is NaN or infinite", "ranges")
                if r < ValidationConstants.MIN_RANGE:
                    raise CriticalValidationError(
                        f"Range[{i}] = {r} is less than minimum {ValidationConstants.MIN_RANGE}",
                        "ranges",
                    )
                if r > ValidationConstants.MAX_RANGE:
                    raise CriticalValidationError(
                        f"Range[{i}] = {r} exceeds maximum {ValidationConstants.MAX_RANGE}",
                        "ranges",
                    )

        # Validate angles if provided
        if angles is not None:
            if len(angles) != 3:
                raise CriticalValidationError(
                    f"Angles must have 3 values, got {len(angles)}", "angles"
                )

            for i, a in enumerate(angles):
                if numpy.isnan(a) or numpy.isinf(a):
                    raise CriticalValidationError(f"Angle[{i}] is NaN or infinite", "angles")
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
        if not isinstance(prob, (int, float, numpy.floating, numpy.integer)):
            raise CriticalValidationError(
                f"{name} must be a number, got {type(prob).__name__}", name
            )
        if numpy.isnan(prob) or numpy.isinf(prob):
            raise CriticalValidationError(f"{name} is NaN or infinite", name)

        if prob < ValidationConstants.MIN_PROBABILITY or prob > ValidationConstants.MAX_PROBABILITY:
            raise CriticalValidationError(
                f"{name} = {prob} outside valid range "
                f"[{ValidationConstants.MIN_PROBABILITY}, {ValidationConstants.MAX_PROBABILITY}]",
                name,
            )

    @staticmethod
    def validate_probability_sum(probs: list[float]) -> None:
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
                "probabilities",
            )

    @staticmethod
    def validate_seed(seed: int) -> None:
        """
        Validates seed value for random number generation.

        Args:
            seed: Seed value

        Raises:
            ValidationError: If seed is negative (C++ contract requires non-negative).
        """
        if seed < ValidationConstants.MIN_SEED:
            raise ValidationError(
                f"Seed value {seed} is negative (must be non-negative)",
                "seed",
            )

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
                f"Indicator count must be positive, got {count}", "indicator_count"
            )

        if count > ValidationConstants.MAX_INDICATORS:
            raise CriticalValidationError(
                f"Indicator count {count} exceeds maximum {ValidationConstants.MAX_INDICATORS}",
                "indicator_count",
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

        if not isinstance(coef, (int, float, numpy.floating, numpy.integer)):
            raise CriticalValidationError(
                f"{name} must be a number, got {type(coef).__name__}", name
            )
        if math.isnan(coef) or math.isinf(coef):
            raise CriticalValidationError(f"{name} must be finite, got {coef}", name)
        if coef < -1.0 or coef > 1.0:
            raise CriticalValidationError(f"{name} must be in [-1, 1] range, got {coef}", name)

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

        if not isinstance(variance, (int, float, numpy.floating, numpy.integer)):
            raise CriticalValidationError(
                f"{name} must be a number, got {type(variance).__name__}", name
            )
        if math.isnan(variance) or math.isinf(variance):
            raise CriticalValidationError(f"{name} must be finite, got {variance}", name)
        if variance < 0:
            raise CriticalValidationError(f"{name} must be non-negative, got {variance}", name)


# ============================================================================
# Decorators for Function Validation
# ============================================================================


def validate_grid_params(func):
    """Decorator to validate grid parameters"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find grid parameter
        grid = None
        if "grid" in kwargs:
            grid = kwargs["grid"]
        else:
            # Try to find grid in positional arguments
            for arg in args:
                if hasattr(arg, "x") and hasattr(arg, "y") and hasattr(arg, "z"):
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
        if "radiuses" in kwargs:
            ParameterValidator.validate_radius(kwargs["radiuses"], "radiuses")

        # Validate max_neighbours if provided
        if "max_neighbours" in kwargs or "max_neighbors" in kwargs:
            max_neigh = kwargs.get("max_neighbours", kwargs.get("max_neighbors"))
            if max_neigh is not None:
                ParameterValidator.validate_max_neighbors(max_neigh)

        # Validate covariance model if provided
        if "cov_model" in kwargs:
            cov_model = kwargs["cov_model"]
            ParameterValidator.validate_covariance_parameters(
                cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
            )

        return func(*args, **kwargs)

    return wrapper


def validate_simulation_params(func):
    """Decorator to validate simulation parameters"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate seed if provided
        if "seed" in kwargs:
            ParameterValidator.validate_seed(kwargs["seed"])

        # Validate min_neighbours if provided
        if "min_neighbours" in kwargs or "min_neighbors" in kwargs:
            min_neigh = kwargs.get("min_neighbours", kwargs.get("min_neighbors"))
            max_neigh = kwargs.get("max_neighbours", kwargs.get("max_neighbors", 12))
            if min_neigh is not None:
                ParameterValidator.validate_min_neighbors(min_neigh, max_neigh)

        return func(*args, **kwargs)

    return wrapper


def validate_file_params(func):
    """Decorator to validate file parameters"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate filename for reading
        if "filename" in kwargs:
            filename = kwargs["filename"]
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
        self.errors: list[ValidationError] = []
        self.warnings: list[str] = []

    def validate_grid_dimensions(self, x: int, y: int, z: int) -> None:
        """Validate grid dimensions and record results"""
        try:
            GridValidator.validate_grid_dimensions(x, y, z)
        except ValidationError as e:
            if self.strict:
                raise
            self.errors.append(e)

    def validate_radius(
        self, radius: float | int | tuple, name: str = "radius"
    ) -> tuple[float, float, float]:
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
            raise CriticalValidationError(f"Validation failed with {len(self.errors)} error(s)", "")
        return False


# Export all public classes and functions
__all__ = [
    "ValidationConstants",
    "ValidationError",
    "CriticalValidationError",
    "ValidationWarning",
    "PathValidator",
    "GridValidator",
    "ParameterValidator",
    "validate_grid_params",
    "validate_kriging_params",
    "validate_simulation_params",
    "validate_file_params",
    "ValidationContext",
]
