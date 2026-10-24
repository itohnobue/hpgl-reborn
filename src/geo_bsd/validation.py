# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""
Input Validation Framework for HPGL Python Interface
Addresses vulnerability IV-001 (CVSS 7.5) - Insufficient Input Validation

This module provides comprehensive input validation for all HPGL Python functions.

Exception taxonomy (F-689): "wrong argument type" is signalled by two documented
exception types, both part of the public contract:

* ``TypeError`` from the dedicated type-checking helpers
  (``validate_property_type``, ``validate_list_param``, ``validate_scalar_mean``,
  ``validate_max_neighbors``, ``validate_seed``) and from the int gates in
  ``validate_min_neighbors`` / ``validate_indicator_count``.
* ``CriticalValidationError`` from the numeric validators
  (``validate_radius``, ``validate_covariance_parameters``,
  ``validate_probability``, ``validate_correlation_coef``,
  ``validate_variance``), whose ``Raises:`` blocks document it.

A caller that must catch a wrong-type error on a numeric parameter therefore has
to catch ``ValidationError`` as well as ``TypeError``. Unifying the two families
is a breaking public-API change (it rewrites documented ``Raises:`` contracts on
both sides) and is deliberately not done here.
"""

from __future__ import annotations

import logging
import os
import pathlib
import warnings
from contextlib import contextmanager
from functools import wraps

import numpy

# Configure validation logger
validation_logger = logging.getLogger("hpgl.validation")

# F-698: ``os.O_NOFOLLOW`` is documented as Unix-only — the Windows ``os``
# module has no such attribute, so the unguarded ``os.O_NOFOLLOW`` references in
# the file-IO helpers below raised ``AttributeError`` before the file was
# touched, taking every Python GSLIB load/save path down on a platform
# pyproject.toml declares as supported. Resolve the flag ONCE with a 0 fallback;
# on Windows the effective guards are the ``realpath`` containment check plus
# ``O_EXCL`` on the temp file (residual noted in each helper's docstring).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


# ============================================================================
# Validation Constants
# ============================================================================


class ValidationConstants:
    """Constants for validation limits"""

    # Grid dimension limits
    MIN_GRID_DIMENSION = 1
    MAX_GRID_DIMENSION = 10000000
    MAX_GRID_SIZE = 1000000000  # 1 billion cells

    # Total values allowed in a single pure-Python GSLIB load
    # (grid cells x property count). Bounds the worst case to the same
    # footprint a single MAX_GRID_SIZE property already permits.
    MAX_GSLIB_VALUES = 1000000000

    # Volume cap for the pure-Python moving-average path (F-M16). The
    # cubical-mask path is vectorized, but its mean-mask allocation is
    # radius-driven ((2r+1)^3 cells); both the vectorized and per-cell loop
    # paths share this bound to prevent a hang/OOM from oversized grids.
    MAX_MOVING_AVERAGE_VOLUME = 1000000

    # Work cap for the pure-Python moving-average per-cell (ellipse-mask)
    # path (F-30). The loop is O(N·V): every grid cell runs MeanCalc over
    # a (2rx+1)(2ry+1)(2rz+1) window. MAX_MOVING_AVERAGE_VOLUME bounds the grid
    # volume (memory) only; N × window_volume bounds the actual work.
    # 1e8 (~2.5 s worst case at measured pure-Python throughput) mirrors
    # the MAX_TOTAL_PAIR_LAG_WORK philosophy for the other pure-Python
    # hot loops (variogram.py).
    MAX_MOVING_AVERAGE_WORK = 100000000

    # Neighbor count limits
    MIN_NEIGHBORS = 1
    MAX_NEIGHBORS = 1000
    # Hard upper bound aligned with the C++ API (api.cpp
    # MAX_NEIGHBOURS_UPPER_BOUND = 10000 — the NEIGHBOUR_WORK_CAP the
    # lookup paths honor as the effective scan/result cap; E2-145/E-M85:
    # the previous 100000 admitted per-node matrices of 80-240 GB +
    # ~3.3e14 flop solves while the lookup silently capped the count at
    # 4913, and simple_cokriging_markI.cpp build_system guards at
    # 10001 = cap + 1 for the secondary equation). Values above this are
    # hard-rejected here so the Python surface fails fast with a clear
    # error instead of a late RuntimeError from the C++ engine (M-20).
    MAX_NEIGHBORS_HARD_LIMIT = 10000
    DEFAULT_MAX_NEIGHBORS = 12

    # Radius limits
    MIN_RADIUS = 0.0
    MAX_RADIUS = 1000000.0

    # F-699: the C++ engine caps the covariance box at
    # COVARIANCE_FIELD_VOLUME_CAP (covariance_field.h:140, 1e8 cells) and throws
    # there for a larger box, while the Python boundary admitted any per-axis
    # radius up to MAX_RADIUS — so inputs such as (232, 232, 232) passed
    # validation and failed late as a RuntimeError instead of a clean
    # CriticalValidationError. The cap applies to the VOLUME
    # (2rx+1)(2ry+1)(2rz+1), never to a single axis: a per-axis bound would
    # reject legal anisotropic radii such as (300, 10, 10) = 265,041 cells.
    MAX_COVARIANCE_FIELD_VOLUME = 100000000

    # Positive minimum search radius for the kriging path (SK/LVM/etc.).
    # Radius 0 is still accepted globally (MIN_RADIUS=0.0) because the
    # documented SGS zero-radius CDF-draw behavior depends on it; the
    # kriging path passes MIN_KRIGING_RADIUS to validate_radius so a
    # zero-radius kriging search (which silently mean-fills every node)
    # is rejected instead.
    MIN_KRIGING_RADIUS = 1.0

    # Covariance parameter limits
    MIN_SILL = 1e-6
    MAX_SILL = 1e10
    MIN_NUGGET = 0.0
    MAX_NUGGET = 1e10
    # F-39: the covariance range must be STRICTLY positive, matching the C++
    # contract (covariance_param.cpp set_ranges rejects range <= 0, and
    # create_transform rejects range=0). MIN_RANGE is kept at 0.0 so the
    # comparison below (`r <= MIN_RANGE`) rejects both zero and negative
    # values with a clean CriticalValidationError instead of deferring the
    # failure to a late RuntimeError inside the FFI call.
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
    # F-017: the seed crosses the C ABI as int64_t (api.h:96 for the SGS
    # params struct, and `int64_t seed` in the hpgl_sis_simulation /
    # hpgl_sis_simulation_lvm declarations; the ctypes side is ``C.c_int64``
    # at hpgl_wrap.py:131 and :695/:705). Values above this bound are not
    # representable: ctypes wraps them modulo 2**64, so 2**64+5 replays the
    # stream of 5, and a value in (2**63, 2**64) arrives in C++ as a negative
    # seed and is rejected with a message naming a seed the caller never
    # passed. This is the ONE upper bound every Python seed gate shares.
    # NOT applicable to the variogram seeded RNG, whose ABI field is uint64
    # (cvariogram.py passes ``C.c_uint64``) — a different width.
    MAX_SEED = 2**63 - 1


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


class ValidationWarning(UserWarning):
    """Validation warning that doesn't prevent operation (F-688/F-701).

    Derives from ``UserWarning`` (a ``Warning`` subclass), not from
    ``ValidationError``: ``warnings.warn`` requires its ``category`` argument to
    be a ``Warning`` subclass, so the previous ``ValidationError`` base made this
    class structurally incapable of ever being emitted as a warning while the
    ``validate_max_neighbors`` docstring documented it as the raised type.
    ``UserWarning`` (rather than bare ``Warning``) is the base so that existing
    filters and assertions written for the category the code actually emitted
    before this fix keep matching: ``warnings.warn`` with the default category
    raises ``UserWarning``. ``ValidationError`` remains the hierarchy for RAISED
    errors; this class is the warned category, and ``validate_max_neighbors``
    now passes it to ``warnings.warn``.

    ``severity`` / ``message`` / ``parameter_name`` are kept as public payload
    attributes for callers (and tests) written against the former shape.
    """

    severity = "warning"

    def __init__(self, message: str, parameter_name: str = ""):
        self.message = message
        self.parameter_name = parameter_name
        if parameter_name:
            super().__init__(f"{parameter_name}: {message}")
        else:
            super().__init__(message)


# ============================================================================
# Path Validation
# ============================================================================


class _PathValidatorMeta(type):
    """Metaclass providing a lazy ``DEFAULT_BASE_DIR`` class property (F-N21).

    The base directory is resolved from the current working directory at
    ACCESS time instead of being captured at import time. The import-time
    capture was stale after ``os.chdir()`` and, in service contexts, could
    pin an attacker-influenced import-time cwd as the trusted base.
    """

    @property
    def DEFAULT_BASE_DIR(cls) -> str:
        # G-21: os.getcwd() raises FileNotFoundError when the process cwd has
        # been unlinked (and OSError for other lookup failures). Escaping as a
        # bare FileNotFoundError from a path-validation property is
        # indistinguishable from a missing data file, so translate it into the
        # typed error the path API documents. There is no trustworthy base to
        # fall back to: substituting "/" would silently widen the containment
        # check that every caller depends on.
        try:
            return os.path.realpath(os.getcwd())
        except OSError as e:
            raise CriticalValidationError(
                "Cannot resolve the default base directory: the process working "
                "directory is unavailable",
                "basedir",
            ) from e


class PathValidator(metaclass=_PathValidatorMeta):
    """Validates file paths to prevent directory traversal attacks"""

    # ``DEFAULT_BASE_DIR`` is provided lazily by ``_PathValidatorMeta``:
    # the trusted base directory is resolved at access time from the
    # process cwd (F-N21), so ``os.chdir()`` after import is respected
    # instead of pinning the import-time cwd. Callers SHOULD use it
    # instead of deriving basedir from the filename (self-referential
    # basedir defeats symlink containment).

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

        The base directory is process-global: it is read from the cwd (or from
        ``basedir``) at call time, so per-thread isolation requires absolute
        filenames. One call resolves the base and the target from a SINGLE cwd
        instant (G-13).

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

        # G-13: resolve the trusted base FIRST and anchor a relative filename to
        # it, so ONE call reads the process cwd at one instant. Pre-fix the base
        # (``DEFAULT_BASE_DIR`` → os.getcwd()) and the target
        # (``Path.resolve`` → os.getcwd()) were read separately, so a concurrent
        # os.chdir() could make them disagree: a spurious
        # CriticalValidationError, or a silently mislocated write/read under a
        # different directory with a success return. An explicit ``basedir`` did
        # not mitigate it because only the target resolve consulted the cwd.
        basedir_resolved = None
        if basedir is not None:
            basedir_resolved = pathlib.Path(basedir).resolve()
            if not path.is_absolute():
                # A relative name is resolved UNDER the trusted base, which is
                # what the ``basedir`` contract says it is for; the pre-fix code
                # resolved it against the cwd instead.
                path = basedir_resolved / path

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
        if basedir_resolved is not None:
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

        Platform residual (F-698): ``O_NOFOLLOW`` is Unix-only. Where the flag is
        unavailable (Windows) it is omitted, so a symlink swapped in between
        ``resolve()`` and ``open()`` is not blocked by the open itself — the
        containment check still applies, and the descriptor is never created
        outside it.

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
        fd = os.open(safe_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW)
        try:
            return os.fdopen(fd, "w", encoding=encoding)
        except BaseException:
            # G-22: os.fdopen can fail after the descriptor exists (unknown
            # encoding, MemoryError). Close it explicitly instead of relying on
            # the interpreter's refcount cleanup.
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    @staticmethod
    @contextmanager
    def safe_atomic_open_write(
        filename: str | pathlib.Path,
        basedir: str | pathlib.Path,
        encoding: str = "utf-8",
    ):
        """Validate path for writing and write atomically via temp + rename (F-N18).

        Same path validation as ``safe_open_write`` (basedir containment +
        O_NOFOLLOW), but content is first written to a uniquely-named temp
        file in the same directory and then atomically ``os.replace``-ed onto
        the target on clean exit. A crash or exception mid-write leaves the
        previous target intact instead of a truncated file (the C++ writers
        use the same temp+rename pattern, property_writer.cpp F-52/F-M19
        lineage). The temp name embeds pid + a process-local counter and is
        created with O_CREAT|O_EXCL|O_NOFOLLOW + mode 0644, defeating
        hardlink pre-creation (F-M19/F-N23 parity).

        Args:
            filename: The file path to validate and write.
            basedir: Base directory for containment check (required).
            encoding: Text encoding for the opened temp file.

        Yields:
            A text-mode file object for the temp file. On clean exit the
            content is atomically renamed over ``filename``; on exception
            the temp file is removed.

        Raises:
            CriticalValidationError: If path is invalid or outside basedir.
            OSError: If the file cannot be opened or renamed.

        Platform residual (F-698): ``O_NOFOLLOW`` is Unix-only and is omitted
        where unavailable (Windows), so a symlink swapped in between
        ``resolve()`` and ``open()`` is not blocked by the open itself; the
        containment check and the ``O_EXCL`` temp creation still apply.
        """
        safe_path = PathValidator.validate_filepath_in_basedir(
            filename, basedir=basedir, must_exist=False
        )
        tmp_dir = os.path.dirname(safe_path) or "."
        tmp_base = os.path.basename(safe_path) + ".tmp"
        pid = os.getpid()
        counter = 0
        max_attempts = 64
        fd = -1
        tmp_path = None
        for _ in range(max_attempts):
            candidate = os.path.join(tmp_dir, f"{tmp_base}.{pid}.{counter}")
            counter += 1
            try:
                fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                    0o644,
                )
                tmp_path = candidate
                break
            except FileExistsError:
                continue  # stale temp from a crashed run — try next name
        if tmp_path is None:
            raise OSError(
                f"Could not create a unique temporary file for {safe_path}"
            )
        try:
            f = os.fdopen(fd, "w", encoding=encoding)
        except BaseException:
            # G-22: os.fdopen can fail after the descriptor and the temp file
            # exist. The cleanup handler below is armed only after fdopen
            # succeeds, so without this the descriptor stays open and the temp
            # file is left on disk.
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        try:
            yield f
            f.flush()
            f.close()
            os.replace(tmp_path, safe_path)
        except BaseException:
            try:
                f.close()
            except OSError:
                pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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

        Platform residual (F-698): ``O_NOFOLLOW`` is Unix-only and is omitted
        where unavailable (Windows), so a symlink swapped in between
        ``resolve()`` and ``open()`` is not blocked by the open itself; the
        containment check still applies.
        """
        safe_path = PathValidator.validate_filepath_in_basedir(
            filename, basedir=basedir, must_exist=True
        )
        fd = os.open(safe_path, os.O_RDONLY | _O_NOFOLLOW)
        try:
            return os.fdopen(fd, "r", encoding=encoding)
        except BaseException:
            # G-22: os.fdopen can fail after the descriptor exists (unknown
            # encoding, MemoryError). Close it explicitly.
            try:
                os.close(fd)
            except OSError:
                pass
            raise

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

    @staticmethod
    def validate_grid_size_param(
        size: int | tuple | list, func_name: str = ""
    ) -> None:
        """Validate a grid size parameter that may be a 3-tuple/list or scalar int.

        When ``size`` is a 3-element tuple/list, validates each dimension via
        ``validate_grid_dimensions``.  When ``size`` is a scalar ``int`` only
        the total-element cap is enforced downstream by the caller.
        """
        if isinstance(size, (tuple, list)) and len(size) == 3:
            GridValidator.validate_grid_dimensions(
                int(size[0]), int(size[1]), int(size[2])
            )

    @staticmethod
    def validate_coordinate_arrays(
        x: numpy.ndarray,
        y: numpy.ndarray,
        z: numpy.ndarray,
        name: str = "coordinates",
    ) -> None:
        """Validate that coordinate arrays contain only finite values (F-046).

        Raises ``ValueError`` if any coordinate array has NaN or Inf entries.
        """
        for label, arr in [("X", x), ("Y", y), ("Z", z)]:
            if not numpy.all(numpy.isfinite(arr)):
                raise ValueError(
                    f"{name}: {label} coordinate array contains NaN or Inf values"
                )


# ============================================================================
# Parameter Validation
# ============================================================================


class ParameterValidator:
    """Validates numerical parameters for geostatistical operations"""

    # ---------- type-checking helpers (centralized isinstance) ----------

    @staticmethod
    def validate_property_type(
        prop: object,
        expected_type: type,
        func_name: str,
        param_name: str = "prop",
        type_name: str | None = None,
    ) -> None:
        """Validate that ``prop`` is an instance of ``expected_type``.

        Raises ``TypeError`` with a consistent ``"<func>: <param> must be <type>, got <actual>"``
        message that matches the existing kriging/simulation error texts.

        Args:
            prop: The object to check.
            expected_type: The type ``prop`` must be an instance of.
            func_name: Function name for the error message prefix.
            param_name: Parameter name to include in the message.
            type_name: Override for the type name in the error message
                       (default: ``expected_type.__name__``).
        """
        if not isinstance(prop, expected_type):
            raise TypeError(
                f"{func_name}: {param_name} must be "
                f"{type_name if type_name is not None else expected_type.__name__}, "
                f"got {type(prop).__name__}"
            )

    @staticmethod
    def validate_list_param(value, name: str, func_name: str) -> None:
        """Validate that ``value`` is a ``list``, raising ``TypeError`` otherwise."""
        if not isinstance(value, list):
            raise TypeError(
                f"{func_name}: {name} must be a list, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def validate_scalar_mean(mean: object, func_name: str) -> None:
        """Validate that a scalar mean is a finite number."""
        if not isinstance(mean, (int, float, numpy.floating, numpy.integer)):
            raise TypeError(
                f"{func_name}: scalar mean must be a number, "
                f"got {type(mean).__name__}"
            )
        if numpy.isnan(mean) or numpy.isinf(mean):
            raise ValueError(
                f"{func_name}: scalar mean must be finite, got {mean}"
            )

    # ---------- numerical validators (existing, unchanged) ----------

    @staticmethod
    def validate_radius(
        radius: float | int | tuple,
        name: str = "radius",
        min_radius: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Validates radius parameters.

        The covariance-box volume a radius builds, ``(2rx+1)(2ry+1)(2rz+1)``,
        must not exceed ``MAX_COVARIANCE_FIELD_VOLUME`` — the engine's
        ``COVARIANCE_FIELD_VOLUME_CAP`` (covariance_field.h:140). The check is
        on the volume, not on any single axis (F-699), so anisotropic radii
        such as ``(300, 10, 10)`` stay legal.

        Args:
            radius: Single value or tuple of (rx, ry, rz)
            name: Parameter name for error messages
            min_radius: Optional lower bound override. When ``None``,
                ``ValidationConstants.MIN_RADIUS`` is used. The kriging
                path passes ``MIN_KRIGING_RADIUS`` so a zero-radius
                search (which silently mean-fills) is rejected while the
                SGS zero-radius CDF-draw behavior keeps working.

        Returns:
            Tuple of (rx, ry, rz) - preserves int type when input is int

        Raises:
            CriticalValidationError: If radius is invalid
        """
        # M-10: bool is a subclass of int and would otherwise be accepted as
        # radius 1 (True). The sibling field gates exclude it; validate_radius
        # must too.
        if isinstance(radius, (int, float, numpy.integer, numpy.floating)) and not isinstance(radius, bool):
            vals = [float(radius)] * 3
        elif isinstance(radius, (tuple, list)) and len(radius) == 3:
            vals = list(map(float, radius))
        else:
            raise CriticalValidationError(
                f"Radius must be a number or tuple of 3 numbers, got {type(radius)}", name
            )

        lower_bound = ValidationConstants.MIN_RADIUS if min_radius is None else min_radius
        for i, r in enumerate(vals):
            if numpy.isnan(r) or numpy.isinf(r):
                raise CriticalValidationError(f"{name}[{i}] is NaN or infinite", name)
            if r < lower_bound:
                raise CriticalValidationError(
                    f"{name}[{i}] = {r} is less than minimum {lower_bound}", name
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

        # F-699: mirror the engine's covariance-box volume cap
        # (covariance_field.h:140, COVARIANCE_FIELD_VOLUME_CAP = 1e8 cells;
        # reached from every kriging/SGS/SGS-LVM/SIS covariance-box
        # construction). Without it a legal-looking per-axis radius such as
        # (232, 232, 232) passed Python and failed late as a RuntimeError from
        # deep inside the FFI call. The predicate is the VOLUME, never a
        # per-axis bound: (300, 10, 10) is 265,041 cells and is legal.
        sx, sy, sz = (2 * int(v) + 1 for v in vals)
        box_volume = sx * sy * sz
        if box_volume > ValidationConstants.MAX_COVARIANCE_FIELD_VOLUME:
            raise CriticalValidationError(
                f"{name} = {tuple(int(v) for v in vals)} builds a covariance box "
                f"of {box_volume} cells ((2x+1)(2y+1)(2z+1)), exceeding the "
                f"maximum {ValidationConstants.MAX_COVARIANCE_FIELD_VOLUME} cells "
                f"(aligned with the C++ engine limit); reduce the search radii",
                name,
            )

        return (int(vals[0]), int(vals[1]), int(vals[2]))

    @staticmethod
    def validate_max_neighbors(
        max_neighbors: int, min_neighbors: int | None = None
    ) -> None:
        """
        Validates maximum number of neighbors.

        Args:
            max_neighbors: Maximum number of neighbors
            min_neighbors: Optional lower-bound override. Defaults to
                ``ValidationConstants.MIN_NEIGHBORS`` (1). Kriging entry
                points keep the default (a zero-neighbour kriging search
                is an empty system), while the simulation entry points
                (SGS/SIS) pass ``0`` to admit the C++-documented
                "unconditional simulation" mode (api.cpp:144-167,
                validate_max_neighbours_or_throw allows 0; kriging entries
                use a separate >=1 gate).

        Raises:
            TypeError: If max_neighbors is not an int (bool excluded).
            CriticalValidationError: If max_neighbors is invalid.

        Warns:
            ValidationWarning: If max_neighbors is unusually large (above
                ``MAX_NEIGHBORS`` but within the hard engine limit). The
                warning category is the exported ``ValidationWarning`` type, so
                ``pytest.warns(ValidationWarning)`` or
                ``warnings.filterwarnings("error", category=ValidationWarning)``
                applies to it (F-688/F-701).
        """
        if not isinstance(max_neighbors, (int, numpy.integer)) or isinstance(max_neighbors, bool):
            raise TypeError(
                f"max_neighbors must be an int, got {type(max_neighbors).__name__}"
            )

        if min_neighbors is None:
            min_neighbors = ValidationConstants.MIN_NEIGHBORS
        if max_neighbors < min_neighbors:
            raise CriticalValidationError(
                f"Max neighbors {max_neighbors} is less than minimum {min_neighbors}",
                "max_neighbors",
            )

        # M-20: hard-reject above the C++ engine's upper bound. The C++ API
        # rejects m_max_neighbours > 10000 at every kriging/simulation entry
        # point (api.cpp MAX_NEIGHBOURS_UPPER_BOUND — the lookup's
        # NEIGHBOUR_WORK_CAP), so accepting such values in Python only defers
        # the failure to a late, harder-to-read RuntimeError from deep inside
        # the C++ call. Reject here with the same bound and a clear message.
        # MAX_NEIGHBORS (1000) remains a performance-guidance warning
        # threshold.
        if max_neighbors > ValidationConstants.MAX_NEIGHBORS_HARD_LIMIT:
            raise CriticalValidationError(
                f"Max neighbors {max_neighbors} exceeds the maximum allowed "
                f"{ValidationConstants.MAX_NEIGHBORS_HARD_LIMIT} (aligned with the "
                f"C++ engine limit).",
                "max_neighbors",
            )

        if max_neighbors > ValidationConstants.MAX_NEIGHBORS:
            warnings.warn(
                f"Max neighbors {max_neighbors} exceeds recommended maximum {ValidationConstants.MAX_NEIGHBORS}. "
                "Performance may be degraded.",
                ValidationWarning,
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
            TypeError: If ``min_neighbors`` or ``max_neighbors`` is not an int
                (bool excluded). Pre-fix both comparisons ran on unchecked
                values, so ``NaN``/``2.7`` passed and a string escaped as a raw
                ``'>' not supported between instances of 'str' and 'int'``
                naming no parameter (F-690); the value then failed three layers
                down at the ctypes ``c_int`` struct assignment.
            CriticalValidationError: If min_neighbors is invalid
        """
        # F-690: the family's type prelude, mirroring validate_max_neighbors,
        # validate_seed and SGSConfig.__post_init__. 0 stays legal — the SGS
        # entry documents it as the unconditioned-simulation mode.
        if not isinstance(min_neighbors, (int, numpy.integer)) or isinstance(
            min_neighbors, bool
        ):
            raise TypeError(
                f"min_neighbors must be an int, got {type(min_neighbors).__name__}"
            )
        if not isinstance(max_neighbors, (int, numpy.integer)) or isinstance(
            max_neighbors, bool
        ):
            raise TypeError(
                f"max_neighbors must be an int, got {type(max_neighbors).__name__}"
            )

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
                # F-39: reject range <= 0 at the Python boundary. Pre-fix the
                # comparison was `r < MIN_RANGE` with MIN_RANGE=0.0, so a zero
                # covariance range passed Python validation and only failed
                # late inside the FFI call — C++ set_ranges throws
                # hpgl_exception → RuntimeError instead of a clean
                # CriticalValidationError (live probe, boundary-found F-39).
                # C++ rejects range <= 0 (covariance_param.cpp:27-32), so
                # Python must too. The strict-inequality message mirrors the
                # C++ "range must be > 0" text.
                if r <= ValidationConstants.MIN_RANGE:
                    raise CriticalValidationError(
                        f"Range[{i}] = {r} must be > 0 (C++ rejects range <= 0). "
                        f"Minimum {ValidationConstants.MIN_RANGE} is not inclusive.",
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
            TypeError: If seed is not an integer.
            CriticalValidationError: If seed is negative (C++ contract requires
                non-negative) or exceeds ``ValidationConstants.MAX_SEED`` —
                the int64 width of the C ABI field (F-017). A larger value is
                not representable: ctypes wraps it modulo 2**64, so two
                distinct seeds silently replay one stream. Matches the
                documented contract in the sgs/sis entry points and the
                sibling validators in this chain (2-M-11).
        """
        if not isinstance(seed, (int, numpy.integer)) or isinstance(seed, bool):
            raise TypeError(
                f"seed must be an int, got {type(seed).__name__}"
            )
        if seed < ValidationConstants.MIN_SEED:
            raise CriticalValidationError(
                f"Seed value {seed} is negative (must be non-negative)",
                "seed",
            )
        if seed > ValidationConstants.MAX_SEED:
            raise CriticalValidationError(
                f"Seed value {seed} exceeds the int64 range accepted by the C "
                f"ABI (0..{ValidationConstants.MAX_SEED}); larger values wrap "
                f"modulo 2**64 and replay another seed's random stream",
                "seed",
            )

    @staticmethod
    def validate_indicator_count(count: int) -> None:
        """
        Validates indicator count.

        Args:
            count: Number of indicators

        Raises:
            TypeError: If ``count`` is not an int (bool excluded) — F-690, the
                same missing gate as ``validate_min_neighbors``; pre-fix
                ``NaN``/``2.5``/``True`` passed every comparison.
            CriticalValidationError: If count is invalid
        """
        if not isinstance(count, (int, numpy.integer)) or isinstance(count, bool):
            raise TypeError(
                f"indicator_count must be an int, got {type(count).__name__}"
            )

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
# Centralized Validation Helpers (decorator-compatible)
# ============================================================================
#
# Each function below supports a **dual calling convention**: direct call
# (production code uses this exclusively) and decorator (test code uses
# this).  Both conventions are part of the stable public API.
#
# **Direct call (production):**  validate_xxx(grid, radiuses, ...)
#   → Used by sgs_simulation, ordinary_kriging, simple_kriging, etc.
#
# **Decorator (tests):**  @validate_xxx
#   → Used by test_validation.py to assert that the validation logic
#     fires on decorated functions.  The decorator path is NOT used
#     in production code but is tested for regression safety.
#
# These functions are NOT dead code — they are the canonical entry
# points for all HPGL parameter validation and are kept as public API
# with both conventions for backward compatibility.
#
# Convention for every ``validate_xxx`` below:
#   * Single non-callable arg  →  direct validation (prod code)
#   * Single callable arg       →  decorator (test code)
#   * Multiple args / kwargs    →  direct validation (prod code)
#
# The ``callable(arg)`` dispatch works because every production argument
# (SugarboxGrid, tuple, int, float, str, pathlib.Path) is NOT callable,
# while every decorated-function argument IS callable.


def validate_grid_params(*args, **kwargs):
    """Validate grid dimensions — dual: decorator or direct call.

    **Direct call (production):** ``validate_grid_params(grid)`` where
    ``grid`` has ``.x`` / ``.y`` / ``.z`` attributes.

    **Decorator (tests):** ``@validate_grid_params``, the decorated
    function's first argument with ``x/y/z`` attrs or ``grid=`` kwarg
    is validated.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # --- decorator mode ---
        func = args[0]

        @wraps(func)
        def wrapper(*a, **kw):
            grid = kw.get("grid")
            if grid is None:
                for arg in a:
                    if hasattr(arg, "x") and hasattr(arg, "y") and hasattr(arg, "z"):
                        grid = arg
                        break
            if grid is not None:
                GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)
            return func(*a, **kw)

        return wrapper

    # --- direct call mode ---
    grid = kwargs.get("grid", args[0] if args else None)
    if grid is not None:
        GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)


def validate_kriging_params(*args, **kwargs):
    """Validate kriging params — dual: decorator or direct call.

    **Direct call (production):**
    ``validate_kriging_params(grid, radiuses, max_neighbours, cov_model)``
    returns the validated ``(int, int, int)`` radiuses tuple.

    **Decorator (tests):** ``@validate_kriging_params``.

    An optional ``min_radius`` keyword is forwarded to
    ``ParameterValidator.validate_radius`` — the kriging path passes
    ``ValidationConstants.MIN_KRIGING_RADIUS`` so zero-radius kriging
    (which silently mean-fills) is rejected while SGS zero-radius
    CDF-draw keeps working.

    An optional ``min_neighbors`` keyword is forwarded to
    ``ParameterValidator.validate_max_neighbors``. Defaults to
    ``ValidationConstants.MIN_NEIGHBORS`` (1) — the kriging entries keep
    the default; the SGS simulation entry passes ``0`` so the
    C++-documented "unconditional simulation" mode (api.cpp:144-167) is
    reachable (E-M9).
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # --- decorator mode ---
        func = args[0]

        @wraps(func)
        def wrapper(*a, **kw):
            if "radiuses" in kw:
                ParameterValidator.validate_radius(
                    kw["radiuses"], "radiuses",
                    min_radius=kw.get("min_radius"),
                )
            if "max_neighbours" in kw or "max_neighbors" in kw:
                n = kw.get("max_neighbours", kw.get("max_neighbors"))
                if n is not None:
                    ParameterValidator.validate_max_neighbors(
                        n, min_neighbors=kw.get("min_neighbors")
                    )
            if "cov_model" in kw:
                cm = kw["cov_model"]
                ParameterValidator.validate_covariance_parameters(
                    cm.sill, cm.nugget, cm.ranges, cm.angles
                )
            return func(*a, **kw)

        return wrapper

    # --- direct call mode ---
    grid = kwargs.get("grid", args[0] if len(args) > 0 else None)
    radiuses = kwargs.get("radiuses", args[1] if len(args) > 1 else None)
    max_neighbours = kwargs.get("max_neighbours", args[2] if len(args) > 2 else None)
    cov_model = kwargs.get("cov_model", args[3] if len(args) > 3 else None)
    min_radius = kwargs.get("min_radius", None)
    min_neighbors = kwargs.get("min_neighbors", None)

    if grid is not None:
        validate_grid_params(grid)
    valid_radiuses = None
    if radiuses is not None:
        valid_radiuses = ParameterValidator.validate_radius(
            radiuses, "radiuses", min_radius=min_radius
        )
    if max_neighbours is not None:
        ParameterValidator.validate_max_neighbors(
            max_neighbours, min_neighbors=min_neighbors
        )
    if cov_model is not None:
        ParameterValidator.validate_covariance_parameters(
            cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
        )
    return valid_radiuses


def validate_simulation_params(*args, **kwargs):
    """Validate simulation params — dual: decorator or direct call.

    **Direct call (production):**
    ``validate_simulation_params(seed=seed, min_neighbours=..., max_neighbours=...)``

    **Decorator (tests):** ``@validate_simulation_params``.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # --- decorator mode ---
        func = args[0]

        @wraps(func)
        def wrapper(*a, **kw):
            if "seed" in kw:
                ParameterValidator.validate_seed(kw["seed"])
            if "min_neighbours" in kw or "min_neighbors" in kw:
                mn = kw.get("min_neighbours", kw.get("min_neighbors"))
                mx = kw.get("max_neighbours", kw.get("max_neighbors", 12))
                if mn is not None:
                    ParameterValidator.validate_min_neighbors(mn, mx)
            return func(*a, **kw)

        return wrapper

    # --- direct call mode ---
    seed = kwargs.get("seed", args[0] if len(args) > 0 else None)
    min_neighbours = kwargs.get("min_neighbours", None)
    max_neighbours = kwargs.get("max_neighbours", None)

    if seed is not None:
        ParameterValidator.validate_seed(seed)
    if min_neighbours is not None:
        ParameterValidator.validate_min_neighbors(
            min_neighbours, max_neighbours if max_neighbours is not None else 12
        )


def validate_file_params(*args, **kwargs):
    """Validate file path — dual: decorator or direct call.

    **Direct call (production):**
    ``validate_file_params(filename, must_exist=True)``

    **Decorator (tests):** ``@validate_file_params``.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # --- decorator mode ---
        func = args[0]

        @wraps(func)
        def wrapper(*a, **kw):
            if "filename" in kw:
                filename = kw["filename"]
                if filename is not None:
                    PathValidator.validate_filepath(filename, must_exist=True)
            return func(*a, **kw)

        return wrapper

    # --- direct call mode ---
    filename = kwargs.get("filename", args[0] if args else None)
    must_exist = kwargs.get("must_exist", True)
    if filename is not None:
        PathValidator.validate_filepath(filename, must_exist=must_exist)


def validate_property_name(name: str, func_name: str = "validate_property_name") -> str:
    """Validate a property name for INC/GSLIB header writes (F-N16/F-N17).

    Shared contract with the C++ writers (property_writer.cpp
    ``validate_property_name``, property_writer.cpp:34-52). A name is
    VALID iff:
      - non-empty;
      - contains no control characters (bytes 0x00-0x1F or 0x7F) — in
        particular no ``\\n``/``\\r`` (would inject phantom header lines),
        no ``\\t``;
      - no leading or trailing whitespace (readers skip whitespace-leading
        lines, silently shifting the data off-by-one);
      - does not start with ``"--"`` (comment marker skipped by readers) or
        ``"/"`` (INC end-of-data marker);
      - at most 1024 UTF-8 bytes (MAX_PROP_NAME_LENGTH in the C++ fast
        reader, ``read_prop_name`` / load_property_from_file.cpp) — the
        reader throws on longer names, so the writer must not produce
        files the reader rejects (E-M72). Byte length (not code points)
        matches the C++ ``name.size()`` cap and the FFI UTF-8 encoding.
    Internal spaces and non-ASCII UTF-8 are allowed.

    Args:
        name: Property name to validate.
        func_name: Caller name for the error message prefix.

    Returns:
        The validated ``name`` (unchanged).

    Raises:
        ValueError: If ``name`` violates the contract.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"{func_name}: property name must be a non-empty string")

    # E-M72: length cap consistent with the C++ fast reader's
    # MAX_PROP_NAME_LENGTH (1024 bytes, load_property_from_file.cpp
    # read_prop_name). The FFI encodes the name as UTF-8, so the cap is
    # measured in bytes to match the C++ name.size() check exactly.
    if len(name.encode("utf-8")) > 1024:
        raise ValueError(
            f"{func_name}: property name {name!r} exceeds maximum length "
            f"(1024 bytes)"
        )

    if name.startswith("--"):
        raise ValueError(
            f"{func_name}: property name {name!r} must not start with \"--\""
        )
    if name[0] == "/":
        raise ValueError(
            f"{func_name}: property name {name!r} must not start with '/'"
        )
    if name[0].isspace() or name[-1].isspace():
        raise ValueError(
            f"{func_name}: property name {name!r} must not have leading or "
            f"trailing whitespace"
        )
    for ch in name:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            raise ValueError(
                f"{func_name}: property name {name!r} must not contain control "
                f"characters"
            )
    return name


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
    "validate_property_name",
    "ValidationContext",
]
