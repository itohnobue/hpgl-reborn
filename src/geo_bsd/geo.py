# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
# III-24: postponed evaluation of annotations is required for Python 3.9
# (declared-supported, requires-python >=3.9). Without it the module-level
# PEP 604 union below (`_last_kriging_stats: dict | None = None`) is
# evaluated at import time and raises TypeError on 3.9 (type.__or__ landed
# in 3.10). Other package modules (config.py, ffi_adapter.py, validation.py)
# already carry this future import.
from __future__ import annotations

import functools
import logging
import math
import os
import re
from typing import Any

import numpy

# Import validation framework
from . import validation
from .ffi_adapter import (
    # Re-exports from hpgl_wrap (struct types, library, callbacks)
    _HPGL_MEDIAN_IK_PARAMS,
    _HPGL_OK_PARAMS,
    _HPGL_SK_PARAMS,
    __hpgl_cockriging_m1_params_t,
    __hpgl_cockriging_m2_params_t,
    __hpgl_cov_params_t,
    # ctypes struct construction helpers
    _c_array,
    _create_hpgl_shape,
    _error_local,  # noqa: F401 — re-exported for contract test compatibility
    # Error checking infrastructure (moved to ffi_adapter)
    _hpgl_call_lock,
    # ctypes type aliases for use with _c_array()
    c_double,
    c_float,
    c_int,
    c_ubyte,
    call_get_last_exception_message,
    call_get_thread_num,
    call_indicator_kriging,
    call_lvm_kriging,
    call_median_ik,
    # C API wrapper functions (one per C API call)
    call_ordinary_kriging,
    call_read_inc_file_byte,
    call_read_inc_file_float,
    call_set_output_handler,
    call_set_progress_handler,
    call_set_thread_num,
    call_simple_cokriging_mark1,
    call_simple_cokriging_mark2,
    call_simple_kriging,
    call_simple_kriging_weights,
    call_write_gslib_byte_property,
    call_write_gslib_cont_property,
    call_write_inc_file_byte,
    call_write_inc_file_float,
    # Numpy array validation
    checkFWA,
    # Kriging stats (diagnostic information from C++ kriging calls)
    get_kriging_stats,
    hpgl_output_handler,
    hpgl_progress_handler,
)
from .ffi_adapter import (
    checked_create as __checked_create,
)
from .ffi_adapter import (
    create_cont_masked_array as _create_hpgl_cont_masked_array,
)
from .ffi_adapter import (
    create_ik_params as __create_hpgl_ik_params,
)
from .ffi_adapter import (
    create_ind_masked_array as _create_hpgl_ind_masked_array,
)
from .gslib_ref import (
    GSLIB_SENTINEL_WINDOW,
    is_gslib_missing_sentinel,
)
from .validation import (
    GridValidator,
    ParameterValidator,
    PathValidator,
    ValidationConstants,
    validate_kriging_params,
    validate_property_name,
)

logger = logging.getLogger(__name__)

# Maximum number of elements for slow Python-based file parsers.
# Files exceeding this limit should use the fast C++ reader instead
# (specify `size` parameter in load_cont_property / load_ind_property).
_MAX_SLOW_PARSER_ELEMENTS = 10_000_000

# Maximum number of bytes in a single line for slow Python-based file
# parsers (2-M-16). A crafted newline-free line with millions of tokens
# would otherwise force line.split() to materialize a multi-GB token list
# (and a multi-GB strip() copy) before the element cap could fire, OOM-ing
# the process. HPGL-written files emit one value per line; GSLIB-style
# files use short lines — this bound only rejects pathological single
# lines far above any legitimate format.
_MAX_SLOW_PARSER_LINE_BYTES = 1_000_000

# Compiled token pattern for the slow parsers' streaming tokenizer (2-M-16).
_SLOW_TOKEN_RE = re.compile(r"\S+")

# E2-06/R-06: undefined_value re-masking on the Python slow paths must
# agree with the C++ fast reader, which masks with EXACT float32 equality
# (read_inc_file.cpp:379: `v == undefined_value` on float32 values). The
# INC writer emits %.9E (property_writer.cpp:121) — 10 significant
# digits, >= FLT_DECIMAL_DIG=9 — so any float32 value round-trips the
# writer exactly and exact equality after a float32 conversion is both
# sufficient and required. The earlier tolerance band (1e-6 relative) was
# calibrated to the OLD %E-6 writer (7 significant digits), which could
# not round-trip fractional undefined_values with >7 significant digits;
# with the fixed writer that band over-masks near-sentinel real data —
# e.g. -99.00005 next to sentinel -99.0 is 9.9e-5 inside the old band
# (relative 5.05e-7 < 1e-6) yet the C++ reader leaves it informed
# (float32(-99.00005) != float32(-99.0)). Comparing in float32 — the same
# conversion the C++ reader applies before its == — reproduces the C++
# mask on every value.


def _is_undefined_value(val, undefined_value):
    """Exact float32 undefined_value match (E2-06/R-06).

    Returns True when ``val`` is indistinguishable from ``undefined_value``
    after an HPGL INC writer round-trip: converting both to float32 — the
    storage type of ContProperty data and the parse type of the C++ fast
    reader (read_inc_file.cpp:379) — and comparing for equality. The
    writer's %.9E output round-trips float32 exactly, so any value the
    writer emitted as the sentinel re-masks and any real value near (but
    not equal to) the sentinel stays informed, exactly like the C++ side.
    NaN handling stays exact (NaN == NaN is False; a NaN sentinel only
    matches a NaN undefined_value). Values outside the float32 range
    cannot equal undefined_value after the round-trip and are handled by
    the GSLIB ±1.0e21 window check at the call sites; the finiteness/range
    guard avoids a numpy overflow-cast warning for them.
    """
    if math.isnan(undefined_value):
        return math.isnan(val)
    # float32 max as a Python float: comparing a Python float directly
    # against the numpy float32 scalar would itself overflow-cast.
    _f32_max = float(numpy.finfo(numpy.float32).max)
    if not math.isfinite(val) or abs(val) > _f32_max:
        return False
    return numpy.float32(val) == numpy.float32(undefined_value)


def _iter_slow_tokens(line, func_name):
    """Yield whitespace-separated tokens from ``line`` one at a time (2-M-16).

    ``line.split()`` materializes the full token list before the caller's
    element cap can be checked — a crafted single line with billions of
    tokens would OOM the process. This generator iterates regex matches
    lazily and enforces the per-line token bound during tokenization, so
    the parse aborts before any large list is built.

    Args:
        line: A single line read from the INC file.
        func_name: Name of the calling parser for error messages.

    Yields:
        Each whitespace-separated token of ``line``.
    """
    count = 0
    for m in _SLOW_TOKEN_RE.finditer(line):
        if count >= _MAX_SLOW_PARSER_ELEMENTS:
            raise MemoryError(
                f"{func_name}: single line exceeds {_MAX_SLOW_PARSER_ELEMENTS} "
                f"tokens. Use fast C++ reader by specifying `size` parameter."
            )
        count += 1
        yield m.group()


# Module-level handler references initialized to None
_h = None
_progress_handler = None
# Store param references to prevent garbage collection while C++ holds void*.
# Without these, the caller dropping param would create a dangling pointer.
_output_handler_param = None
_progress_handler_param = None

# Kriging diagnostic stats from the most recent kriging/simulation call.
# SENTINEL CONTRACT: every kriging/simulation wrapper (ordinary_kriging,
# simple_kriging, lvm_kriging, median_ik, indicator_kriging,
# simple_cokriging_markI/II, sgs_simulation, sis_simulation) resets this
# to None BEFORE its FFI call, so stale stats from a prior call never
# survive an exception raised by the C++ call. On the C++ side every
# kriging/simulation entry point except simple_kriging_weights (a
# reset-only, stats-less path — api.cpp:746) calls set_kriging_stats() —
# OK/SK/LVM, cokriging markI/II
# (simple_cokriging_markI.cpp:439/:498), median_ik
# (median_ik.cpp:201) and indicator_kriging (indicator_kriging.h:323)
# since F-M5, and SGS/SIS since F-M6 — so the C++ thread-local stats are
# always populated (matches the hpgl_wrap.py:649-651 producer list). The
# Python sentinel is then populated via get_kriging_stats() only by
# wrappers that call ``_finalize_kriging_stats`` (ordinary_kriging,
# simple_kriging, lvm_kriging, median_ik, indicator_kriging,
# simple_cokriging_markI/II, sgs.py, sis.py). Since M-9 the median_ik and
# indicator_kriging wrappers also call ``_finalize_kriging_stats()``, so a
# singular/degenerate kriging system surfaces a RuntimeError via
# ``_check_kriging_failure_stats`` instead of being silently
# marginal-substituted (they previously were deliberately RESET-ONLY).
# Reset/populate run inside ``_hpgl_call_lock`` (atomic per-call
# sentinel, F-M23/F-N12) via the ``_reset_kriging_stats`` /
# ``_finalize_kriging_stats`` helpers. Callers can inspect
# geo_bsd.geo._last_kriging_stats after a kriging call to detect partial
# solver failure (e.g. points_without_neighbours > 0). Set to None if the
# native library does not export hpgl_get_kriging_stats.
_last_kriging_stats: dict | None = None
# Deferred cache of old CFUNCTYPE handler+param references to prevent
# use-after-free in concurrent kriging calls. Kriging/simulation FFI calls
# hold _hpgl_call_lock (so a concurrent handler clear cannot run mid-call),
# and both the set and clear paths defer the displaced CFUNCTYPE+param
# here. The cache is a shared bounded FIFO across BOTH output and progress
# handlers, so the bound accounts for both (4 generations each).
_old_handler_refs: list[tuple[Any, Any]] = []
_OLD_HANDLER_REFS_CAP = 8


class ContProperty:
    """Continuous (floating-point) property with informed/uninformed mask.

    The core data container for continuous geostatistical properties.
    Data and mask are stored as Fortran-order (column-major) arrays
    for direct passing to the C++ backend.

    Parameters
    ----------
    data : numpy.ndarray
        Array of property values. Converted to float32, Fortran order.
    mask : numpy.ndarray
        Array of mask values. Converted to uint8, Fortran order.
        Non-zero = informed cell, 0 = uninformed (masked) cell.

    Attributes
    ----------
    data : numpy.ndarray
        Property values (float32, Fortran order).
    mask : numpy.ndarray
        Mask array (uint8, Fortran order).
    """

    @property
    def data(self):
        """Property values (float32, Fortran order)."""
        return self._data

    @data.setter
    def data(self, val):
        arr = numpy.require(val, "float32", "F")
        checkFWA(arr)
        # 2-M-15: a caller replacing data with a different shape would
        # desynchronize data/mask — the C++ backend indexes both arrays with
        # the same strides and would read/write out of bounds.
        if hasattr(self, "_mask") and arr.shape != self._mask.shape:
            raise ValueError(
                f"Data shape {arr.shape} does not match mask shape {self._mask.shape}"
            )
        self._data = arr

    @property
    def mask(self):
        """Mask array (uint8, Fortran order)."""
        return self._mask

    @mask.setter
    def mask(self, val):
        arr = numpy.require(val, "uint8", "F")
        checkFWA(arr)
        # 2-M-15: same data/mask shape invariant as the data setter.
        if hasattr(self, "_data") and arr.shape != self._data.shape:
            raise ValueError(
                f"Mask shape {arr.shape} does not match data shape {self._data.shape}"
            )
        self._mask = arr

    def __init__(self, data: numpy.ndarray, mask: numpy.ndarray):
        # Use asarray() so lists/tuples are supported; ndim check must
        # happen before require() which may change ndim.
        ndarray = numpy.asarray(data)
        if ndarray.ndim not in (1, 3):
            raise ValueError(
                f"ContProperty data must be 1D or 3D, got {ndarray.ndim}D"
            )
        # III-11: the isfinite check must run on the STORED float32 array,
        # not the pre-downcast view. A float64 value such as 1e300 passes
        # the float64 check but overflows to inf in the float32 downcast
        # (numpy.require emits only a RuntimeWarning), silently storing inf
        # — which then propagates NaN through calc_mean and other consumers.
        # errstate keeps that intermediate warning out of the user's way:
        # the ValueError below is the authoritative signal.
        with numpy.errstate(over="ignore", invalid="ignore"):
            stored = numpy.require(ndarray, "float32", "F")
        if not numpy.all(numpy.isfinite(stored)):
            raise ValueError("ContProperty data contains NaN or Inf values")
        # Bypass property setters: validation is already done above,
        # and self.validate() below covers F/W/A checks.
        object.__setattr__(self, '_data', stored)
        object.__setattr__(self, '_mask', numpy.require(mask, "uint8", "F"))
        self.validate()

    def validate(self):
        checkFWA(self.data)
        checkFWA(self.mask)
        if self.data.shape != self.mask.shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match mask shape {self.mask.shape}"
            )

    def fix_shape(self, grid):
        if self.data.ndim != 3:
            if self.data.size == grid.x * grid.y * grid.z:
                # Bypass the property setters: both arrays are reshaped to
                # the same 3D shape together, preserving the data/mask shape
                # invariant (the 2-M-15 setters would reject the transient
                # 3D-vs-1D state while the second array is still 1D).
                object.__setattr__(
                    self, '_data',
                    self.data.reshape((grid.x, grid.y, grid.z), order="F"),
                )
        if self.mask.ndim != 3:
            if self.mask.size == grid.x * grid.y * grid.z:
                object.__setattr__(
                    self, '_mask',
                    self.mask.reshape((grid.x, grid.y, grid.z), order="F"),
                )

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        elif idx == 1:
            return self.mask
        else:
            raise RuntimeError("Index out of range.")


def _validate_indicator_data(ndarray, func_name):
    """Validate indicator values BEFORE the uint8 conversion (F-23/II-37).

    Shared by the ``IndProperty`` constructor and the public ``data``
    setter so both enforce the same invariants: 1D/3D ndim, finite values,
    values in [0, 255], and integrality. numpy.require(..., "uint8") would
    otherwise silently wrap out-of-range values (300 -> 44) and truncate
    fractional values (1.5 -> 1) before any range check could fire,
    defeating the indicator-range invariant.

    Args:
        ndarray: Raw input (array-like) to validate.
        func_name: Caller name for error messages.

    Returns:
        numpy.ndarray: The validated array (asarray view of the input).

    Raises:
        ValueError: If ndim is not 1 or 3, or any value is non-finite,
            outside [0, 255], or non-integral.
    """
    ndarray = numpy.asarray(ndarray)
    if ndarray.ndim not in (1, 3):
        raise ValueError(
            f"{func_name}: IndProperty data must be 1D or 3D, got {ndarray.ndim}D"
        )
    # Validate NaN/Inf before uint8 conversion which silently maps NaN→0.
    if not numpy.all(numpy.isfinite(numpy.asarray(ndarray, dtype=float))):
        raise ValueError(f"{func_name}: IndProperty data contains NaN or Inf values")
    # Validate values are in [0, 255] and integral BEFORE the uint8
    # conversion: numpy.require(..., "uint8") silently wraps out-of-range
    # values (256.0 → 0) and truncates fractional values (1.5 → 1), which
    # would defeat the post-conversion range check below.
    if numpy.any(ndarray < 0) or numpy.any(ndarray > 255):
        raise ValueError(
            f"{func_name}: IndProperty data values must be in [0, 255], "
            "got values outside the range"
        )
    if not numpy.all(numpy.equal(ndarray, numpy.floor(ndarray))):
        raise ValueError(f"{func_name}: IndProperty data must contain integer values")
    return ndarray


class IndProperty:
    """Indicator (categorical) property with informed/uninformed mask.

    Similar to ``ContProperty`` but stores unsigned 8-bit indicator
    values and tracks the number of indicator categories.

    Parameters
    ----------
    data : numpy.ndarray
        Array of indicator values. Converted to uint8, Fortran order.
        Values should be in ``[0, indicator_count)``.
    mask : numpy.ndarray
        Array of mask values. Converted to uint8, Fortran order.
    indicator_count : int
        Number of indicator categories.

    Raises
    ------
    RuntimeError
        If any informed cell has an indicator value outside
        ``[0, indicator_count)``.

    Attributes
    ----------
    data : numpy.ndarray
        Indicator values (uint8, Fortran order).
    mask : numpy.ndarray
        Mask array (uint8, Fortran order).
    indicator_count : int
        Number of indicator categories.
    """

    @property
    def data(self):
        """Indicator values (uint8, Fortran order)."""
        return self._data

    @data.setter
    def data(self, val):
        # F-23 + II-37: reuse the constructor's full validation block BEFORE
        # the uint8 conversion — numpy.require(..., "uint8") would otherwise
        # silently truncate fractional values (1.5 -> 1) and wrap out-of-range
        # values (300 -> 44), defeating the indicator-range invariant below.
        ndarray = _validate_indicator_data(val, "IndProperty.data")
        arr = numpy.require(ndarray, "uint8", "F")
        checkFWA(arr)
        # 2-M-15: a caller replacing data with a different shape would
        # desynchronize data/mask — the C++ backend indexes both arrays with
        # the same strides and would read/write out of bounds.
        if hasattr(self, "_mask") and arr.shape != self._mask.shape:
            raise ValueError(
                f"Data shape {arr.shape} does not match mask shape {self._mask.shape}"
            )
        # F-23: re-validate the indicator-range invariant for the new data —
        # the constructor enforces it (e19aa84) but the setter previously
        # accepted out-of-range values that the C++ backend then silently
        # dropped via equality matching.
        if (
            hasattr(self, "indicator_count")
            and numpy.sum(
                numpy.bitwise_and((self._mask > 0), (arr >= self.indicator_count))
            )
            > 0
        ):
            raise RuntimeError(
                "Property contains some indicators outside of [0..%s] range."
                % (self.indicator_count - 1)
            )
        self._data = arr

    @property
    def mask(self):
        """Mask array (uint8, Fortran order)."""
        return self._mask

    @mask.setter
    def mask(self, val):
        arr = numpy.require(val, "uint8", "F")
        checkFWA(arr)
        # 2-M-15: same data/mask shape invariant as the data setter.
        if hasattr(self, "_data") and arr.shape != self._data.shape:
            raise ValueError(
                f"Mask shape {arr.shape} does not match data shape {self._data.shape}"
            )
        # II-36: re-validate the indicator-range invariant for newly-informed
        # cells. The constructor checks only cells that are informed AT
        # CONSTRUCTION, so a masked cell holding the 255 sentinel (the
        # library's own masked-cell convention — _load_prop_ind_slow and
        # append_mask) is legal while masked; unmasking it via this setter
        # would produce an informed out-of-range value that the constructor
        # rejects (e19aa84 added the constructor check only).
        if (
            hasattr(self, "indicator_count")
            and numpy.sum(
                numpy.bitwise_and((arr > 0), (self._data >= self.indicator_count))
            )
            > 0
        ):
            raise RuntimeError(
                "Property contains some indicators outside of [0..%s] range."
                % (self.indicator_count - 1)
            )
        self._mask = arr

    def __init__(self, data: numpy.ndarray, mask: numpy.ndarray, indicator_count: int):
        if not 1 <= indicator_count <= 255:
            raise ValueError(
                f"indicator_count must be 1-255, got {indicator_count}"
            )
        # Use asarray() so lists/tuples are supported; ndim check must
        # happen before require() which may change ndim.
        ndarray = _validate_indicator_data(data, "IndProperty")
        mask_array = numpy.asarray(mask)
        # Bypass property setters: validation is already done above,
        # and self.validate() below covers F/W/A checks.
        object.__setattr__(self, '_data', numpy.require(ndarray, "uint8", "F"))
        object.__setattr__(self, '_mask', numpy.require(mask_array, "uint8", "F"))
        self.indicator_count = indicator_count
        if ndarray.shape != mask_array.shape:
            raise ValueError(
                f"Data shape {ndarray.shape} does not match mask shape {mask_array.shape}"
            )
        if numpy.sum(numpy.bitwise_and((self.mask > 0), (self.data >= indicator_count))) > 0:
            raise RuntimeError(
                "Property contains some indicators outside of [0..%s] range."
                % (indicator_count - 1)
            )
        self.validate()

    def validate(self):
        checkFWA(self.data)
        checkFWA(self.mask)
        if self.data.shape != self.mask.shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match mask shape {self.mask.shape}"
            )

    def fix_shape(self, grid):
        if self.data.ndim != 3:
            if self.data.size == grid.x * grid.y * grid.z:
                # Bypass the property setters: both arrays are reshaped to
                # the same 3D shape together, preserving the data/mask shape
                # invariant (the 2-M-15 setters would reject the transient
                # 3D-vs-1D state while the second array is still 1D).
                object.__setattr__(
                    self, '_data',
                    self.data.reshape((grid.x, grid.y, grid.z), order="F"),
                )
        if self.mask.ndim != 3:
            if self.mask.size == grid.x * grid.y * grid.z:
                object.__setattr__(
                    self, '_mask',
                    self.mask.reshape((grid.x, grid.y, grid.z), order="F"),
                )

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        elif idx == 1:
            return self.mask
        elif idx == 2:
            return self.indicator_count
        else:
            raise RuntimeError("Index out of range.")


def _prop_to_tuple_(prop):
    if isinstance(prop, ContProperty):
        return (prop.data, prop.mask)
    elif isinstance(prop, IndProperty):
        return (prop.data, prop.mask, prop.indicator_count)
    else:
        raise RuntimeError(f"_prop_to_tuple_: unknown property type: {type(prop)}")


def append_mask(prop, mask):
    infs = prop[1]
    infs &= mask
    # Use dtype-appropriate sentinel: -99 wraps to 157 in uint8 (mod 256).
    # For IndProperty (uint8), use 255 as the masked-cell sentinel since
    # valid indicator values are in [0, indicator_count) with max 254.
    # For ContProperty (float32), -99 is safe.
    if prop[0].dtype == numpy.uint8:
        infs.choose(255, prop[0], out=prop[0])
    else:
        infs.choose(-99, prop[0], out=prop[0])


class covariance:
    spherical = 0
    exponential = 1
    gaussian = 2


class SugarboxGrid:
    """Regular 3D grid definition used by all kriging and simulation functions.

    Parameters
    ----------
    x : int
        Number of cells along the X (first) dimension.
    y : int
        Number of cells along the Y (second) dimension.
    z : int
        Number of cells along the Z (third) dimension.

    Raises
    ------
    CriticalValidationError
        If any dimension is non-positive or exceeds the maximum allowed
        grid size.

    Attributes
    ----------
    x, y, z : int
        Grid dimensions.
    """

    def __init__(self, x: int, y: int, z: int):
        # Validate grid dimensions
        GridValidator.validate_grid_dimensions(x, y, z)
        self.x = x
        self.y = y
        self.z = z


class CovarianceModel:
    """Variogram/covariance model parameters for kriging and simulation.

    Parameters
    ----------
    type : int, optional
        Covariance model type. Use ``geo_bsd.covariance`` constants:
        ``spherical`` (0), ``exponential`` (1), or ``gaussian`` (2).
        Default is spherical.
    ranges : tuple of float, optional
        Anisotropy ranges ``(rx, ry, rz)``. Default ``(1, 1, 1)``.
    angles : tuple of float, optional
        Anisotropy angles ``(azimuth, dip, rotation)`` in degrees.
        Default ``(0.0, 0.0, 0.0)``.
    sill : float, optional
        Covariance sill (variance). Default 1.0.
    nugget : float, optional
        Nugget effect (variance at zero distance). Default 0.0.

    Raises
    ------
    CriticalValidationError
        If sill is not positive, nugget is negative, or ranges
        contain non-positive values.

    Attributes
    ----------
    type : int
    ranges : tuple of float
    angles : tuple of float
    sill : float
    nugget : float
    """

    def __init__(
        self,
        type: int = 0,
        ranges: tuple = (1, 1, 1),
        angles: tuple = (0.0, 0.0, 0.0),
        sill: float = 1.0,
        nugget: float = 0.0,
    ):
        # Validate covariance type against known C++ enum values
        # (covariance_type_t: COV_SPHERICAL=0, COV_EXPONENTIAL=1,
        #  COV_GAUSSIAN=2). C++ init_fun() throws on unknown types;
        # validating in Python provides a clearer error earlier.
        if type not in (0, 1, 2):
            raise validation.CriticalValidationError(
                f"Covariance type must be 0 (spherical), 1 (exponential), "
                f"or 2 (gaussian), got {type}",
                "type",
            )

        # Convert list values to tuples for ctypes compatibility.
        # ctypes (c_double * 3) fields require tuples — lists cause
        # TypeError at the kriging call sites (ordinary_kriging,
        # simple_kriging, lvm_kriging).
        if isinstance(ranges, list):
            ranges = tuple(ranges)
        if isinstance(angles, list):
            angles = tuple(angles)

        self.type = type
        self.ranges = ranges
        self.angles = angles
        self.sill = sill
        self.nugget = nugget

        # Validate covariance parameters
        ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)


def _load_prop_cont_slow(filename, undefined_value, basedir=None):
    values = []
    mask = []
    skipped_count = 0
    element_count = 0
    # Security: uses safe_open_read() which validates the path and opens
    # atomically with O_NOFOLLOW to prevent TOCTOU symlink attacks.
    # The basedir is the trusted base (DEFAULT_BASE_DIR unless the caller
    # supplies an explicit one) — NOT the filename's own directory, which
    # would defeat symlink containment (F-28).
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_open_read(filename, basedir=basedir) as f:
        for line in f:
            # 2-M-16: bound the per-line processing BEFORE any strip/split
            # work. A crafted newline-free line would otherwise materialize
            # a multi-GB token list (and strip() copy) before the element
            # cap fires, OOM-ing the process.
            if len(line) > _MAX_SLOW_PARSER_LINE_BYTES:
                raise MemoryError(
                    f"_load_prop_cont_slow: line exceeds {_MAX_SLOW_PARSER_LINE_BYTES} bytes. "
                    f"Use fast C++ reader by specifying `size` parameter."
                )
            if line.strip().startswith("--"):
                continue
            # Detect INC format end-of-data marker '/' — stop parsing.
            # The C++ writer emits '/' after all data values.
            if line.strip().startswith("/"):
                break
            for part in _iter_slow_tokens(line, "_load_prop_cont_slow"):
                # F-54: a mid-line '--' token is a comment in the C++ fast
                # reader — it skips the rest of the line. Match that here so
                # both parsers consume identical token streams.
                if part.startswith("--"):
                    break
                # Count all token attempts (including non-numeric) for DoS protection.
                # Prevents unbounded loop from malicious files with billions of
                # unparseable tokens.
                if element_count >= _MAX_SLOW_PARSER_ELEMENTS:
                    raise MemoryError(
                        f"_load_prop_cont_slow: file exceeds {_MAX_SLOW_PARSER_ELEMENTS} elements. "
                        f"Use fast C++ reader by specifying `size` parameter."
                    )
                element_count += 1
                try:
                    val = float(part.strip())
                    values.append(val)
                    # IEEE 754 NaN≠NaN: equality check fails when both are NaN.
                    # Use math.isnan() to detect NaN sentinel values reliably.
                    # F-M18/M-16: values outside the ±1.0e21 window are missing
                    # sentinels in addition to exact undefined_value matches
                    # (strict inequality per the GSLIB convention — see
                    # gslib_ref.py reference-fact table, got-20260802092630).
                    # NaN-aware: comparisons with NaN are false, so a NaN value
                    # is only masked when undefined_value is itself NaN
                    # (consistent with the C++ reader, which leaves NaN cells
                    # informed otherwise).
                    # E2-06/R-06: the undefined_value match is exact after
                    # a float32 conversion (see _is_undefined_value),
                    # mirroring the C++ fast reader's float32 == so both
                    # parsers mask the same cells on the same file.
                    if (
                        _is_undefined_value(val, undefined_value)
                        or val < -GSLIB_SENTINEL_WINDOW
                        or val > GSLIB_SENTINEL_WINDOW
                    ):
                        mask.append(0)
                    else:
                        mask.append(1)
                except (ValueError, TypeError):
                    skipped_count += 1
    # F-54 documented divergence: the C++ fast reader THROWS on unparseable
    # junk tokens ("Error parsing 'X' string.") while this slow fallback
    # SKIPS them with a warning. This is intentional — the slow parser is
    # the lenient fallback (test_edge_cases.py documents the skip behavior)
    # and legitimate HPGL-written files contain no junk tokens. Both paths
    # agree on the token/terminator semantics that DO occur in legitimate
    # files: line-start "/" ends the data, mid-line "/" is skipped, and a
    # "--" token anywhere skips the rest of the line.
    if skipped_count > 0:
        logger.warning(
            "_load_prop_cont_slow: skipped %d non-numeric tokens in %s", skipped_count, filename
        )

    data64 = numpy.array(values)
    mask_arr = numpy.array(mask, dtype="uint8")
    # E-M1: GSLIB sentinels outside the float32 range (|v| > ~3.4e38, e.g.
    # -1e39) are masked by the window check above but overflow to ±inf in
    # the float32 downcast, crashing the ContProperty ctor (whose isfinite
    # check runs on the STORED float32 array, III-11). NaN cells masked by
    # a NaN undefined_value hit the same crash. Masked cells are
    # uninformed — the C++ writer emits undefined_value for them
    # (property_writer.cpp write_property_cont) — so their stored value is
    # not part of the round-trip contract; clamp only the cells that
    # cannot survive the downcast. Informed non-finite cells still fail
    # loudly in the ctor.
    unsupported = (mask_arr == 0) & (
        ~numpy.isfinite(data64)
        | (numpy.abs(data64) > numpy.finfo(numpy.float32).max)
    )
    if numpy.any(unsupported):
        data64[unsupported] = 0.0
    return ContProperty(data64.astype("float32"), mask_arr)


def _load_prop_ind_slow(filename, undefined_value, ind_values, basedir=None):
    # Validate that ind_values contains no duplicates. Duplicate indicator
    # values cause dict_map overwrites, silently corrupting the category
    # mapping when later entries overwrite earlier ones.
    seen = set()
    for v in ind_values:
        if v in seen:
            raise ValueError(
                f"Duplicate indicator value {v} in ind_values. "
                f"Each indicator value must be unique."
            )
        seen.add(v)

    dict_map = {}
    for i in range(len(ind_values)):
        dict_map[ind_values[i]] = i

    values = []
    mask = []
    skipped_parse = 0
    unknown_values = set()
    element_count = 0

    # Security: uses safe_open_read() which validates the path and opens
    # atomically with O_NOFOLLOW to prevent TOCTOU symlink attacks.
    # The basedir is the trusted base (DEFAULT_BASE_DIR unless the caller
    # supplies an explicit one) — NOT the filename's own directory, which
    # would defeat symlink containment (F-28).
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_open_read(filename, basedir=basedir) as f:
        for line in f:
            # 2-M-16: bound the per-line processing BEFORE any strip/split
            # work (see _load_prop_cont_slow — same crafted-line DoS).
            if len(line) > _MAX_SLOW_PARSER_LINE_BYTES:
                raise MemoryError(
                    f"_load_prop_ind_slow: line exceeds {_MAX_SLOW_PARSER_LINE_BYTES} bytes. "
                    f"Use fast C++ reader by specifying `size` parameter."
                )
            if line.strip().startswith("--"):
                continue
            # Detect INC format end-of-data marker '/' — stop parsing.
            # The C++ writer emits '/' after all data values.
            if line.strip().startswith("/"):
                break
            for part in _iter_slow_tokens(line, "_load_prop_ind_slow"):
                # F-54: a mid-line '--' token is a comment in the C++ fast
                # reader — it skips the rest of the line. Match that here so
                # both parsers consume identical token streams.
                if part.startswith("--"):
                    break
                # Count all token attempts (including non-numeric) for DoS protection.
                # Prevents unbounded loop from malicious files with billions of
                # unparseable tokens.
                if element_count >= _MAX_SLOW_PARSER_ELEMENTS:
                    raise MemoryError(
                        f"_load_prop_ind_slow: file exceeds {_MAX_SLOW_PARSER_ELEMENTS} elements. "
                        f"Use fast C++ reader by specifying `size` parameter."
                    )
                element_count += 1
                try:
                    val = int(part.strip())
                except (ValueError, TypeError):
                    skipped_parse += 1
                    continue
                if val == undefined_value:
                    values.append(255)
                    mask.append(0)
                elif val in dict_map:
                    values.append(dict_map[val])
                    mask.append(1)
                else:
                    unknown_values.add(val)

    if skipped_parse > 0:
        logger.warning(
            "_load_prop_ind_slow: skipped %d unparseable tokens in %s", skipped_parse, filename
        )
    if unknown_values:
        sorted_unknown = sorted(unknown_values, key=int)
        raise ValueError(
            f"_load_prop_ind_slow: unknown indicator value(s) {sorted_unknown} "
            f"found in {filename}. Expected indicator values: {list(ind_values)}"
        )

    return IndProperty(
        numpy.array(values, dtype="uint8", order="F"),
        numpy.array(mask, dtype="uint8", order="F"),
        len(ind_values),
    )


def _create_cont_prop(size):
    return ContProperty(numpy.zeros(size, dtype="float32"), numpy.zeros(size, dtype="uint8"))


def _create_ind_prop(size, indicator_count):
    return IndProperty(
        numpy.zeros(size, dtype="uint8"), numpy.zeros(size, dtype="uint8"), indicator_count
    )


def _empty_clone(prop):
    data2 = prop.data.copy("F")
    data2.fill(0)
    mask2 = prop.mask.copy("F")
    mask2.fill(0)
    if isinstance(prop, IndProperty):
        return IndProperty(data2, mask2, prop.indicator_count)
    elif isinstance(prop, ContProperty):
        return ContProperty(data2, mask2)
    else:
        raise RuntimeError(f"_empty_clone: unknown property type: {type(prop)}")


def _clone_prop(prop):
    data2 = prop.data.copy("F")
    mask2 = prop.mask.copy("F")
    if isinstance(prop, IndProperty):
        return IndProperty(data2, mask2, prop.indicator_count)
    elif isinstance(prop, ContProperty):
        return ContProperty(data2, mask2)
    else:
        raise RuntimeError(f"_clone_prop: unknown property type: {type(prop)}")


def _require_cont_data(data):
    if data is None:
        return None
    return numpy.require(data, dtype="float32", requirements="F")


def _require_ind_data(data):
    if data is None:
        return None
    return numpy.require(data, dtype="uint8", requirements="F")


def accepts_tuple(arg_name, arg_pos):
    def tuple_to_prop(t):
        if isinstance(t, tuple):
            if len(t) == 3:
                return IndProperty(*t)
            elif len(t) == 2:
                return ContProperty(*t)
            else:
                raise RuntimeError(f"{arg_name}: tuple must have 2 or 3 elements, got {len(t)}")
        else:
            if not isinstance(t, (ContProperty, IndProperty)):
                raise RuntimeError(
                    f"{arg_name}: expected ContProperty, IndProperty, or tuple, got {type(t)}"
                )
            return t

    def decorator(f):
        @functools.wraps(f)
        def new_f(*args, **kargs):
            if arg_name in kargs:
                kargs[arg_name] = tuple_to_prop(kargs[arg_name])
            elif len(args) > arg_pos:
                args = args[:arg_pos] + (tuple_to_prop(args[arg_pos]),) + args[arg_pos + 1 :]
            else:
                raise RuntimeError(
                    f"{arg_name}: missing required positional argument at position {arg_pos}"
                )
            return f(*args, **kargs)

        return new_f

    return decorator


@accepts_tuple("prop", 0)
def write_property(
    prop, filename, prop_name, undefined_value, indicator_values=None, basedir=None
):
    """Write a property to an INC-format file via the C++ backend.

    Supports both ``ContProperty`` and ``IndProperty``. The file is
    written in INC format with the specified undefined value marker.

    Parameters
    ----------
    prop : ContProperty or IndProperty
        Property to write.
    filename : str
        Output file path (validated for security).
    prop_name : str
        Property name written to the file header.
    undefined_value : float or int
        Value marking undefined/uninformed cells in the output.
    indicator_values : list of int, optional
        Mapping of indicator categories to output values. Only used
        for ``IndProperty``.
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

    Raises
    ------
    RuntimeError
        If the C++ write operation fails.

    See Also
    --------
    write_gslib_property : Write property in GSLIB format.
    """
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=False
    )

    # F-N16: validate prop_name BEFORE the FFI call. Names containing '\n'/
    # control chars inject phantom header lines; a '--' prefix is a comment
    # marker the readers skip and leading/trailing whitespace shifts the
    # data off-by-one. Shared contract with the C++ writers
    # (property_writer.cpp validate_property_name).
    validate_property_name(prop_name, "write_property")

    if indicator_values is None:
        indicator_values = []
    ParameterValidator.validate_list_param(
        indicator_values, "indicator_values", "write_property"
    )

    if isinstance(prop, ContProperty):
        marr = _create_hpgl_cont_masked_array(prop, None)
        # II-35: hold _hpgl_call_lock around the FFI call AND the error-message
        # read. The C++ writers set a process-global locale during the file
        # write (locale_keeper); without serialization a concurrent I/O call on
        # another thread can interleave its parse window with this call's,
        # producing silently wrong output and corrupting the process locale.
        with _hpgl_call_lock:
            rc = call_write_inc_file_float(
                marr, safe_path.encode("utf-8"), undefined_value, prop_name.encode("utf-8")
            )
            if rc != 0:
                raise RuntimeError(
                    "write_property failed: "
                    + call_get_last_exception_message().decode("utf-8", errors="replace")
                )
    else:
        # F-44: validate indicator values BEFORE the uint8 conversion so an
        # out-of-range value raises a clear ValueError instead of a confusing
        # numpy OverflowError or silent ctypes wrap (300 -> 44). The C++
        # byte writer also rejects values outside [0, 255], but the Python
        # guard surfaces the error before any file is created.
        _validate_indicator_values(indicator_values, "write_property")
        if not _is_valid_byte_value(undefined_value):
            raise ValueError(
                f"write_property: undefined_value must be an integer in [0, 255] "
                f"for indicator (byte) properties, got {undefined_value!r}"
            )
        # II-38: a sentinel that collides with a mapped category makes
        # informed cells indistinguishable from masked ones on read-back.
        _validate_undefined_value_collision(
            undefined_value, indicator_values, "write_property"
        )
        # Security: Keep reference to indicator_values array
        ind_arr = numpy.array(indicator_values, dtype="uint8")
        marr = _create_hpgl_ind_masked_array(prop, None)
        # II-35: serialize the FFI call + error-message read (locale race).
        with _hpgl_call_lock:
            rc = call_write_inc_file_byte(
                marr,
                safe_path.encode("utf-8"),
                undefined_value,
                prop_name.encode("utf-8"),
                ind_arr,
                len(indicator_values),
            )
            if rc != 0:
                raise RuntimeError(
                    "write_property failed: "
                    + call_get_last_exception_message().decode("utf-8", errors="replace")
                )


@accepts_tuple("prop", 0)
def write_gslib_property(
    prop, filename, prop_name, undefined_value, indicator_values=None, basedir=None
):
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=False
    )

    # F-N16: validate prop_name BEFORE the FFI call (shared contract with
    # the C++ writers; see write_property).
    validate_property_name(prop_name, "write_gslib_property")

    if indicator_values is None:
        indicator_values = []
    ParameterValidator.validate_list_param(
        indicator_values, "indicator_values", "write_gslib_property"
    )

    if isinstance(prop, ContProperty):
        # II-35: serialize the FFI call + error-message read (locale race).
        with _hpgl_call_lock:
            rc = call_write_gslib_cont_property(
                _create_hpgl_cont_masked_array(prop, None),
                safe_path.encode("utf-8"),
                prop_name.encode("utf-8"),
                undefined_value,
            )
            if rc != 0:
                raise RuntimeError(
                    "write_gslib_property failed: "
                    + call_get_last_exception_message().decode("utf-8", errors="replace")
                )
    else:
        # F-44 + I2-55: validate indicator values and undefined_value before
        # the FFI call — _c_array(c_ubyte, ...) silently wraps 300 -> 44 and
        # the C++ byte writer rejects undefined values outside [0, 255].
        _validate_indicator_values(indicator_values, "write_gslib_property")
        if not _is_valid_byte_value(undefined_value):
            raise ValueError(
                f"write_gslib_property: undefined_value must be an integer in "
                f"[0, 255] for indicator (byte) properties, got {undefined_value!r}"
            )
        # II-38: sentinel/category collision → silent round-trip data loss.
        _validate_undefined_value_collision(
            undefined_value, indicator_values, "write_gslib_property"
        )
        # II-35: serialize the FFI call + error-message read (locale race).
        with _hpgl_call_lock:
            rc = call_write_gslib_byte_property(
                _create_hpgl_ind_masked_array(prop, None),
                safe_path.encode("utf-8"),
                prop_name.encode("utf-8"),
                undefined_value,
                _c_array(c_ubyte, len(indicator_values), indicator_values),
                len(indicator_values),
            )
            if rc != 0:
                raise RuntimeError(
                    "write_gslib_property failed: "
                    + call_get_last_exception_message().decode("utf-8", errors="replace")
                )


def _validate_indicator_values(indicator_values, func_name):
    """Reject indicator values that the byte write path would corrupt (F-44).

    Ports the seen-set duplicate check from the slow parser
    (``_load_prop_ind_slow``) and adds a [0, 255] integrality check. The
    ctypes ``_c_array(c_ubyte, ...)`` path wraps 300 -> 44 silently and
    numpy 2.x raises a confusing ``OverflowError``; both are replaced by a
    clear ``ValueError`` before any file is created.

    Raises:
        ValueError: If any value is not an integer in [0, 255] or a
            duplicate is present.
    """
    seen = set()
    for v in indicator_values:
        if not _is_valid_byte_value(v):
            raise ValueError(
                f"{func_name}: indicator_values must be integers in [0, 255], "
                f"got {v!r}"
            )
        iv = int(v)
        if iv in seen:
            raise ValueError(
                f"{func_name}: duplicate indicator value {iv} in indicator_values. "
                f"Each indicator value must be unique."
            )
        seen.add(iv)


def _is_valid_byte_value(value):
    """True if ``value`` is an integer (or integral float) in [0, 255]."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, numpy.integer)):
        return 0 <= int(value) <= 255
    if isinstance(value, (float, numpy.floating)):
        return numpy.isfinite(value) and float(value).is_integer() and 0 <= value <= 255
    return False


def _validate_undefined_value_collision(undefined_value, indicator_values, func_name):
    """Reject an undefined_value that collides with a mapped indicator value (II-38).

    The byte writers emit ``remap_table[val]`` for informed cells and
    ``undefined_value`` for uninformed ones; if an informed cell's mapped
    output value equals the undefined sentinel, the file is read back with
    that cell marked MASKED — silent round-trip data loss. The prior fix
    (3ad77ee) added ``_validate_indicator_values`` / ``_is_valid_byte_value``
    but missed this collision.

    Args:
        undefined_value: The undefined sentinel for the byte write.
        indicator_values: List of mapped indicator output values.
        func_name: Caller name for error messages.

    Raises:
        ValueError: If ``int(undefined_value)`` appears in ``indicator_values``.
    """
    if int(undefined_value) in {int(v) for v in indicator_values}:
        raise ValueError(
            f"{func_name}: undefined_value {undefined_value!r} is also an "
            f"indicator value; informed cells written with this value would be "
            f"read back as masked (silent round-trip data loss). Choose an "
            f"undefined_value that is not in indicator_values."
        )


def _validate_and_reshape_fallback(result, size, func_name):
    """Reshape slow-parser fallback result to match expected grid dimensions.

    When the fast C++ reader fails, the slow parser returns flat 1D arrays.
    This function validates that the element count matches the expected grid
    size and reshapes the result to 3D Fortran-order arrays.

    Args:
        result: A ``ContProperty`` or ``IndProperty`` from the slow parser.
        size: Tuple ``(nx, ny, nz)`` or a scalar element count.
        func_name: Name of the calling function for error messages.

    Raises:
        RuntimeError: If the slow parser read a different number of elements
            than expected from the size parameter.
    """
    if isinstance(size, (tuple, list)) and len(size) == 3:
        total = size[0] * size[1] * size[2]
    else:
        total = size
    if result.data.size != total:
        raise RuntimeError(
            f"{func_name}: Slow parser read {result.data.size} elements, "
            f"but expected {total} elements. "
            f"The file may be corrupted or the size parameter is incorrect."
        )
    if isinstance(size, (tuple, list)) and len(size) == 3:
        # Bypass the property setters: both arrays are reshaped to the same
        # 3D shape together, preserving the data/mask shape invariant (the
        # 2-M-15 setters would reject the transient 3D-vs-1D state while the
        # second array is still 1D).
        object.__setattr__(
            result, '_data',
            result.data.reshape((size[0], size[1], size[2]), order="F"),
        )
        object.__setattr__(
            result, '_mask',
            result.mask.reshape((size[0], size[1], size[2]), order="F"),
        )


def load_cont_property(filename, undefined_value, size=None, basedir=None):
    """Load a continuous property from an INC-format file.

    If ``size`` is provided, uses the fast C++ reader with
    pre-allocated buffers. If ``size`` is None, falls back to a
    Python-based parser suitable for smaller files.

    Parameters
    ----------
    filename : str
        Path to the INC file (must exist).
    undefined_value : float
        Value in the file that marks undefined/uninformed cells.
    size : tuple of int or None, optional
        Grid dimensions ``(nx, ny, nz)``. If None, uses slow parser.
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

    Returns
    -------
    ContProperty
        Loaded property with data and mask arrays. When ``size`` is a
        3-tuple, the arrays are reshaped to 3D Fortran order (matching
        the fallback path's ``_validate_and_reshape_fallback`` so both
        fast and slow paths return the same shape — F-55).

    Raises
    ------
    RuntimeError
        If the file read fails.

    Notes
    -----
    For files larger than ~100 MB, always specify ``size`` to use
    the fast C++ reader and avoid unbounded memory usage.

    See Also
    --------
    load_ind_property : Load indicator (categorical) property.
    """
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Validate filename for security
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=True
    )

    if size is None:
        logger.warning(
            "load_cont_property: Size is not specified. Using slow Python-based parser "
            "that loads entire file into memory unbounded. For large files (>100MB) "
            "specify size to use the fast C++ reader which uses pre-allocated buffers."
        )
        return _load_prop_cont_slow(safe_path, undefined_value, basedir=basedir)
    else:
        try:
            result = read_inc_file_float(safe_path, undefined_value, size, basedir=basedir)
        except RuntimeError as e:
            logger.warning(
                "load_cont_property: Fast C++ reader failed, falling back to slow "
                "Python parser. C++ error: %s",
                e,
            )
            result = _load_prop_cont_slow(safe_path, undefined_value, basedir=basedir)
            _validate_and_reshape_fallback(result, size, "load_cont_property")
            return result
        # F-55: normalize the fast-path result to the same shape the
        # fallback returns — a 3-tuple size yields 3D Fortran arrays so
        # the public contract is stable regardless of which path ran.
        if isinstance(size, (tuple, list)) and len(size) == 3:
            _validate_and_reshape_fallback(result, size, "load_cont_property")
        return result


def read_inc_file_float(filename, undefined_value, size, basedir=None):
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=True
    )

    # Validate size parameters
    GridValidator.validate_grid_size_param(size)

    total_elements = (
        size[0] * size[1] * size[2] if isinstance(size, (tuple, list)) and len(size) == 3 else size
    )
    if total_elements > 2147483647:
        raise ValueError(
            f"Grid too large: {total_elements} elements exceeds c_int max (2147483647)"
        )
    data = numpy.zeros(total_elements, dtype="float32", order="F")
    mask = numpy.zeros(total_elements, dtype="uint8", order="F")

    # II-35: serialize the FFI call + error-message read. The C++ fast
    # reader sets a process-global locale for the sscanf parse window
    # (locale_keeper); concurrent I/O from another thread could interleave
    # with this call's parse window and silently corrupt the numeric parse
    # AND the process locale on non-glibc platforms.
    with _hpgl_call_lock:
        rc = call_read_inc_file_float(
            safe_path.encode("utf-8"), undefined_value, total_elements, data, mask
        )
        if rc != 0:
            raise RuntimeError(
                "read_inc_file_float failed: "
                + call_get_last_exception_message().decode("utf-8", errors="replace")
            )

    return ContProperty(data, mask)


def read_inc_file_byte(filename, undefined_value, size, indicator_values, basedir=None):
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=True
    )

    # F-44: validate indicator values before the uint8 conversion — numpy
    # 2.x raises a confusing OverflowError for out-of-range values. The
    # C++ reader validates undefined_value against [0, 255] itself, and a
    # text sentinel such as -99 is legitimate input for the slow-parser
    # fallback, so undefined_value is intentionally NOT range-checked here.
    _validate_indicator_values(indicator_values, "read_inc_file_byte")

    # Validate size parameters
    GridValidator.validate_grid_size_param(size)

    total_elements = (
        size[0] * size[1] * size[2] if isinstance(size, (tuple, list)) and len(size) == 3 else size
    )
    if total_elements > 2147483647:
        raise ValueError(
            f"Grid too large: {total_elements} elements exceeds c_int max (2147483647)"
        )
    data = numpy.zeros(total_elements, dtype="uint8", order="F")
    mask = numpy.zeros(total_elements, dtype="uint8", order="F")
    # II-35: serialize the FFI call + error-message read (locale race —
    # see read_inc_file_float).
    with _hpgl_call_lock:
        rc = call_read_inc_file_byte(
            safe_path.encode("utf-8"),
            undefined_value,
            total_elements,
            data,
            mask,
            numpy.array(indicator_values, dtype="uint8"),
            len(indicator_values),
        )
        if rc != 0:
            raise RuntimeError(
                "read_inc_file_byte failed: "
                + call_get_last_exception_message().decode("utf-8", errors="replace")
            )
    return IndProperty(data, mask, len(indicator_values))


def load_ind_property(filename, undefined_value, indicator_values, size=None, basedir=None):
    """Load an indicator (categorical) property from an INC-format file.

    If ``size`` is provided, uses the fast C++ reader with
    pre-allocated buffers. If ``size`` is None, falls back to a
    Python-based parser suitable for smaller files.

    Parameters
    ----------
    filename : str
        Path to the INC file (must exist).
    undefined_value : int
        Value in the file that marks undefined/uninformed cells.
    indicator_values : list of int
        List of expected indicator values in the file. Each value is
        mapped to an internal category index starting from 0.
    size : tuple of int or None, optional
        Grid dimensions ``(nx, ny, nz)``. If None, uses slow parser.
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

    Returns
    -------
    IndProperty
        Loaded indicator property with data, mask, and indicator_count
        set to ``len(indicator_values)``. When ``size`` is a 3-tuple, the
        arrays are reshaped to 3D Fortran order (matching the fallback
        path so both fast and slow paths return the same shape — F-55).

    Raises
    ------
    RuntimeError
        If the file read fails.
    ValueError
        If the file contains indicator values not in ``indicator_values``.

    Notes
    -----
    For files larger than ~100 MB, always specify ``size`` to use
    the fast C++ reader and avoid unbounded memory usage.

    See Also
    --------
    load_cont_property : Load continuous property.
    """
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    # Validate filename for security
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=basedir, must_exist=True
    )

    if size is None:
        logger.warning(
            "load_ind_property: Size is not specified. Using slow Python-based parser "
            "that loads entire file into memory unbounded. For large files (>100MB) "
            "specify size to use the fast C++ reader which uses pre-allocated buffers."
        )
        return _load_prop_ind_slow(safe_path, undefined_value, indicator_values, basedir=basedir)
    else:
        try:
            result = read_inc_file_byte(
                safe_path, undefined_value, size, indicator_values, basedir=basedir
            )
        except RuntimeError as e:
            logger.warning(
                "load_ind_property: Fast C++ reader failed, falling back to slow "
                "Python parser. C++ error: %s",
                e,
            )
            result = _load_prop_ind_slow(
                safe_path, undefined_value, indicator_values, basedir=basedir
            )
            _validate_and_reshape_fallback(result, size, "load_ind_property")
            return result
        # F-55: normalize the fast-path result to the same shape the
        # fallback returns — a 3-tuple size yields 3D Fortran arrays so
        # the public contract is stable regardless of which path ran.
        if isinstance(size, (tuple, list)) and len(size) == 3:
            _validate_and_reshape_fallback(result, size, "load_ind_property")
        return result


def set_thread_num(num):
    """Set the number of OpenMP threads for parallel computation.

    Parameters
    ----------
    num : int
        Number of threads (must be at least 1).

    Raises
    ------
    TypeError
        If ``num`` is not an integer.
    ValueError
        If ``num`` is less than 1.

    Notes
    -----
    A warning is logged if ``num`` exceeds 4x the CPU count, as
    this may cause thread oversubscription.

    See Also
    --------
    get_thread_num : Get the current thread count.
    """
    ParameterValidator.validate_property_type(num, int, "set_thread_num", param_name="num", type_name="an integer")
    if num < 1:
        raise ValueError(f"set_thread_num: num must be at least 1, got {num}")
    # Sanity-check: warn if num exceeds available CPU count
    cpu_count = os.cpu_count()
    if cpu_count is not None and num > cpu_count * 4:
        logger.warning(
            "set_thread_num: num=%d exceeds 4x CPU count (%d). "
            "This may cause thread oversubscription and performance degradation.",
            num,
            cpu_count,
        )
    _hpgl_call_lock.acquire()
    try:
        rc = call_set_thread_num(num)
        if rc != 0:
            raise RuntimeError(f"set_thread_num: C++ call failed with return code {rc}")
    finally:
        _hpgl_call_lock.release()


def get_thread_num():
    """Get the current OpenMP thread count.

    Returns
    -------
    int
        The number of OpenMP threads (via ``omp_get_max_threads()``).

    Notes
    -----
    ``omp_get_max_threads()`` is thread-safe and does not require
    serialization via ``_hpgl_call_lock``.
    """
    return call_get_thread_num()


@accepts_tuple("prop", 0)
def calc_mean(prop):
    """Calculate the arithmetic mean of informed (unmasked) values.

    Parameters
    ----------
    prop : ContProperty
        Continuous property with data and mask arrays.

    Returns
    -------
    float
        Mean of values in informed cells.

    Raises
    ------
    ValueError
        If no informed values exist (all cells masked).
    """
    ParameterValidator.validate_property_type(prop, ContProperty, "calc_mean")

    masked = numpy.ma.masked_where(prop.mask == 0, prop.data)
    if masked.count() == 0:
        raise ValueError("calc_mean: no informed values (all masked)")
    # III-11: a non-finite informed value (NaN/Inf) would silently produce
    # NaN from the masked mean — reject it instead. Reachable via the
    # ContProperty.data setter (which is deliberately the NaN injection
    # path for the kriging wrappers' own validation tests).
    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError("calc_mean: prop.data contains NaN or Inf values")
    return float(masked.mean())


# _validate_kriging_params has been moved to validation.py as
# validate_kriging_params — it is now a centralized entry point
# used by all kriging and simulation functions.  The old name is
# kept as an alias for backward compatibility.
_validate_kriging_params = validate_kriging_params


def _reset_kriging_stats() -> None:
    """Reset the documented inspection sentinel BEFORE an FFI call (F-12/F-M23/F-N12).

    MUST be called inside ``with _hpgl_call_lock:``. Guarantees the sentinel
    stays None when the C++ call raises (stale stats never survive an exception).
    """
    global _last_kriging_stats
    _last_kriging_stats = None


def _finalize_kriging_stats(expected_calculated: int, func_name: str) -> None:
    """Populate the sentinel from C++ and surface solver failure (F-33/F-M4/F-M21).

    MUST be called inside ``with _hpgl_call_lock:`` immediately after the FFI
    call. The C++ stats are thread_local (api.cpp:187) but the Python sentinel
    is a shared module-global; doing reset+populate+check under the lock makes
    each call's sentinel write atomic with respect to other kriging threads, so
    the failure check always consumes THIS call's stats (F-M23/F-N12).

    ``get_kriging_stats`` is referenced as a module-global (NOT captured in a
    default argument) so the F-33 monkeypatch tests (test_s6_python_b_fixes.py:
    138-227) can patch it.

    ``expected_calculated`` = grid.x*grid.y*grid.z - informed_count (number of
    uninformed cells the solver should have calculated).
    """
    global _last_kriging_stats
    try:
        _last_kriging_stats = get_kriging_stats()
    except (NotImplementedError, AttributeError):
        # Library build without hpgl_get_kriging_stats (hpgl_wrap.py:636-641);
        # sentinel already None from the reset — honest "no stats available".
        return
    _check_kriging_failure_stats(_last_kriging_stats, expected_calculated, func_name)


def _check_kriging_failure_stats(stats, expected_calculated, func_name):
    """Surface partial/total kriging solver failure from C++ stats (F-33).

    SK/LVM kriging mean-fills on failure, so a call that failed everywhere
    returns a finite mean-filled property that is indistinguishable from a
    successful call. This consumes the C++ ``kriging_stats_t`` counters
    (via ``_last_kriging_stats``) so the Python wrapper surfaces the failure:

    * ``points_singularity > 0`` → RuntimeError: a singular kriging system
      is a genuine numerical solver failure — the covariance model /
      neighbourhood configuration produced a degenerate system and the
      mean-fill masks it. This is the F-33 defect ("no raise on
      points_singularity > 0").
    * ``points_calculated < expected`` with zero singularity → a warning:
      cells with no neighbours are mean-filled. This is the documented
      ``mean_on_failure`` contract (e.g. pure-nugget covariance or sparse
      data), so it is surfaced loudly but does not abort the call.

    Args:
        stats: The stats dict from ``get_kriging_stats()`` or None.
        expected_calculated: Number of uninformed cells that kriging
            should have calculated (``grid_size - informed_count``).
        func_name: Name of the calling wrapper for messages.
    """
    if stats is None:
        return
    singular = int(stats.get("points_singularity", 0))
    calculated = int(stats.get("points_calculated", 0))
    if singular > 0:
        raise RuntimeError(
            f"{func_name}: kriging system was singular at {singular} point(s); "
            f"those cells were left at the failure fallback (mean-filled for "
            f"SK/LVM/cokriging; undefined for OK). A singular kriging system "
            f"is a numerical solver failure — check the covariance model and "
            f"neighbourhood configuration. stats={stats}"
        )
    if expected_calculated > 0 and calculated < expected_calculated:
        missing = expected_calculated - calculated
        logger.warning(
            "%s: %d of %d cells could not be kriged (no neighbours in the "
            "search radius) and were left at the failure fallback (mean-filled "
            "for SK/LVM/cokriging; undefined for OK; marginal draw for SGS; "
            "marginal-probability substitution for SIS); stats=%s",
            func_name, missing, expected_calculated, stats,
        )


def _validate_prop_grid_shape(prop, grid, func_name):
    """Validate per-dimension 3D property shape against the grid (H-2).

    ``simple_kriging``/``lvm_kriging`` pass raw arrays to the C++ backend
    (unlike ``ordinary_kriging``, which routes through
    ``create_cont_masked_array``'s 3D shape check), so an equal-volume
    shape mismatch such as a (2,2,2) prop with a (1,2,4) grid is silently
    misread/miswritten without an explicit check. 1D (flat) properties are
    covered by the existing size check and carry no per-dimension meaning.
    """
    if prop.data.ndim == 3 and (
        prop.data.shape[0] != grid.x
        or prop.data.shape[1] != grid.y
        or prop.data.shape[2] != grid.z
    ):
        raise ValueError(
            f"{func_name}: 3D data shape {prop.data.shape} does not match "
            f"grid dimensions ({grid.x}, {grid.y}, {grid.z})"
        )



@accepts_tuple("prop", 0)
def ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model):
    """Perform Ordinary Kriging (OK) interpolation on a 3D grid.

    Ordinary Kriging estimates values at all grid cells using a
    neighborhood of nearby informed cells, weighted by the covariance
    model. The kriging weights sum to 1 (unbiasedness constraint).

    Parameters
    ----------
    prop : ContProperty
        Input property with informed (masked) data.
    grid : SugarboxGrid
        Grid definition specifying the output resolution.
    radiuses : tuple of int
        Search radii ``(rx, ry, rz)`` in grid cells.
    max_neighbours : int
        Maximum number of neighboring points to use per cell.
    cov_model : CovarianceModel
        Covariance model defining spatial correlation.

    Returns
    -------
    ContProperty
        Output property with kriged values at all grid cells.

    Raises
    ------
    CriticalValidationError
        If grid dimensions, radiuses, max_neighbours, or covariance
        parameters are invalid.
    ValueError
        If prop.data is empty, its size does not match the grid volume,
        or it contains NaN or Inf values.
    RuntimeError
        If the C++ computation produces an error (including a singular
        kriging system or non-finite output).

    Notes
    -----
    The underlying C++ function (``hpgl_ordinary_kriging``) returns
    void — there is no per-cell error signal.  When kriging fails for
    individual grid cells (e.g. no neighbours, singular system), those
    cells are left as 0.0 with mask=0 (uninformed) under the
    ``undefined_on_failure`` fallback and the call completes without
    raising; a ``RuntimeError`` is only raised if the post-call
    ``isfinite`` check detects a genuine solver-produced NaN.  Callers
    who need to detect the extent of partial failure can inspect
    ``geo_bsd.geo._last_kriging_stats`` after the call, which is populated
    from C++ ``kriging_stats_t`` via ``get_kriging_stats()``.
    """
    valid_radiuses = _validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model,
        min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
    )

    ParameterValidator.validate_property_type(prop, ContProperty, "ordinary_kriging")

    # Validate property data size against grid
    if prop.data.size == 0:
        raise ValueError("ordinary_kriging: prop.data is empty")
    expected_size = grid.x * grid.y * grid.z
    if prop.data.size != expected_size:
        raise ValueError(
            f"ordinary_kriging: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )

    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError("ordinary_kriging: prop.data contains NaN or Inf")

    out_prop = _empty_clone(prop)

    okp = _HPGL_OK_PARAMS(
        covariance_type=cov_model.type,
        ranges=cov_model.ranges,
        angles=cov_model.angles,
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=valid_radiuses,
        max_neighbours=max_neighbours,
    )

    inp = _create_hpgl_cont_masked_array(prop, grid)
    outp = _create_hpgl_cont_masked_array(out_prop, grid)
    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_ordinary_kriging(inp, okp, outp)
        _finalize_kriging_stats(expected, "ordinary_kriging")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "ordinary_kriging: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


@accepts_tuple("prop", 0)
def simple_kriging(prop, grid, radiuses, max_neighbours, cov_model, mean=None):
    """Perform Simple Kriging (SK) interpolation on a 3D grid.

    Simple Kriging assumes the global mean is known. If ``mean`` is
    None, it is computed automatically from the informed data.

    Parameters
    ----------
    prop : ContProperty
        Input property with informed (masked) data.
    grid : SugarboxGrid
        Grid definition specifying the output resolution.
    radiuses : tuple of int
        Search radii ``(rx, ry, rz)`` in grid cells.
    max_neighbours : int
        Maximum number of neighboring points to use per cell.
    cov_model : CovarianceModel
        Covariance model defining spatial correlation.
    mean : float or None, optional
        Known global mean. If None, computed automatically.

    Returns
    -------
    ContProperty
        Output property with kriged values at all grid cells.

    Raises
    ------
    CriticalValidationError
        If grid dimensions, radiuses, max_neighbours, or covariance
        parameters are invalid.
    ValueError
        If prop.data is empty, its size or per-dimension shape does not
        match the grid, the explicit ``mean`` is non-finite, or prop.data
        contains NaN or Inf values.
    RuntimeError
        If the C++ computation produces an error (including a singular
        kriging system or non-finite output).

    Notes
    -----
    The underlying C++ function (``hpgl_simple_kriging``) returns
    void — there is no per-cell error signal.  When kriging fails for
    individual grid cells (e.g. no neighbours, singular system), those
    cells are silently filled with the global mean (``mean_on_failure``
    fallback) and their output values are indistinguishable from cells
    that were kriged successfully.  Callers who need to detect partial
    results can inspect ``geo_bsd.geo._last_kriging_stats`` after the
    call, which is populated from C++ ``kriging_stats_t`` via
    ``get_kriging_stats()``.  For explicit failure detection, the
    weight-based API (:func:`simple_kriging_weights`) also detects
    failures explicitly.
    """
    valid_radiuses = _validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model,
        min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
    )

    ParameterValidator.validate_property_type(prop, ContProperty, "simple_kriging")

    # Validate property data size against grid
    if prop.data.size == 0:
        raise ValueError("simple_kriging: prop.data is empty")
    expected_size = grid.x * grid.y * grid.z
    if prop.data.size != expected_size:
        raise ValueError(
            f"simple_kriging: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )
    # H-2: equal-volume per-dimension shape mismatch must raise, not silently
    # misread (e.g. a (2,2,2) prop with a (1,2,4) grid — both volume 8).
    _validate_prop_grid_shape(prop, grid, "simple_kriging")

    # Validate mean for NaN/Inf when explicitly provided
    if mean is not None and not numpy.isfinite(mean):
        raise ValueError(f"simple_kriging: mean must be finite, got {mean}")

    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError("simple_kriging: prop.data contains NaN or Inf")

    out_prop = _empty_clone(prop)

    skp = _HPGL_SK_PARAMS(
        covariance_type=cov_model.type,
        ranges=cov_model.ranges,
        angles=cov_model.angles,
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=valid_radiuses,
        max_neighbours=max_neighbours,
        automatic_mean=(mean is None),
        mean=(mean if mean is not None else 0),
    )

    sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_simple_kriging(
            prop.data, prop.mask, sh, skp, out_prop[0], out_prop[1], sh
        )
        _finalize_kriging_stats(expected, "simple_kriging")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "simple_kriging: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


@accepts_tuple("prop", 0)
def lvm_kriging(prop, grid, mean_data, radiuses, max_neighbours, cov_model):
    """Perform Kriging with Locally Varying Mean (LVM).

    LVM Kriging uses a locally varying mean field instead of a
    constant global mean. The local mean is provided as a 3D array.

    Parameters
    ----------
    prop : ContProperty
        Input property with informed (masked) data.
    grid : SugarboxGrid
        Grid definition specifying the output resolution.
    mean_data : numpy.ndarray
        3D array of local mean values, shape matching ``grid``.
        Values are used as the locally varying mean at each cell.
    radiuses : tuple of int
        Search radii ``(rx, ry, rz)`` in grid cells.
    max_neighbours : int
        Maximum number of neighboring points to use per cell.
    cov_model : CovarianceModel
        Covariance model defining spatial correlation.

    Returns
    -------
    ContProperty
        Output property with kriged values at all grid cells.

    Raises
    ------
    CriticalValidationError
        If grid dimensions, radiuses, max_neighbours, or covariance
        parameters are invalid.
    ValueError
        If ``mean_data`` is not a NumPy array, if ``mean_data`` or
        prop.data is empty, if either size (or prop.data's per-dimension
        shape) does not match the grid, or if either contains NaN or Inf.
    RuntimeError
        If the C++ computation produces an error (including a singular
        kriging system or non-finite output).

    Notes
    -----
    The underlying C++ function (``hpgl_lvm_kriging``) returns
    void — there is no per-cell error signal.  When kriging fails for
    individual grid cells (e.g. no neighbours, singular system), those
    cells are silently filled with the local mean value and their output
    values are indistinguishable from cells that were kriged successfully.
    Callers who need to detect partial results can inspect
    ``geo_bsd.geo._last_kriging_stats`` after the call, which is populated
    from C++ ``kriging_stats_t`` via ``get_kriging_stats()``.
    """
    valid_radiuses = _validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model,
        min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
    )

    ParameterValidator.validate_property_type(prop, ContProperty, "lvm_kriging")

    # Validate mean_data
    if not isinstance(mean_data, numpy.ndarray):
        raise ValueError("lvm_kriging: mean_data must be a numpy array")
    mean_data = numpy.require(mean_data, dtype="float32", requirements="F")
    expected_size = grid.x * grid.y * grid.z
    if mean_data.size != expected_size:
        raise ValueError(
            f"lvm_kriging: mean_data size {mean_data.size} does not match "
            f"grid dimensions {grid.x}x{grid.y}x{grid.z} = {expected_size}"
        )
    # R-13 (H-2 residual): equal-volume per-dimension shape mismatch on
    # mean_data must raise, not silently misread. The C++ mean provider
    # consumes the buffer by flat node index (mean_provider.h:38-43), so a
    # (2,2,2) mean_data with a (1,2,4) grid — both volume 8 — permutes the
    # mean field with no exception. 1D (flat) mean vectors are covered by the
    # size check above and carry no per-dimension meaning.
    # E2-01: the R-13 gate was 3D-only — an equal-volume 2D mean_data
    # (e.g. (2,2) on a (1,2,2) grid) sailed past the size check and the
    # ndim==3 guard, then the F-order flat layout of the 2D array was
    # consumed by flat node index with a silently permuted mean field.
    # Only 1D flat and exactly grid-shaped 3D arrays are unambiguous;
    # reject every other ndim for the same reason.
    if mean_data.ndim not in (1, 3):
        raise ValueError(
            f"lvm_kriging: mean_data must be 1D flat (size {expected_size}) or 3D "
            f"with shape ({grid.x}, {grid.y}, {grid.z}), got {mean_data.ndim}D "
            f"shape {mean_data.shape}"
        )
    if mean_data.ndim == 3 and (
        mean_data.shape[0] != grid.x
        or mean_data.shape[1] != grid.y
        or mean_data.shape[2] != grid.z
    ):
        raise ValueError(
            f"lvm_kriging: 3D mean_data shape {mean_data.shape} does not match "
            f"grid dimensions ({grid.x}, {grid.y}, {grid.z})"
        )

    # Validate property data size against grid
    if prop.data.size == 0:
        raise ValueError("lvm_kriging: prop.data is empty")
    if prop.data.size != expected_size:
        raise ValueError(
            f"lvm_kriging: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )
    # H-2: equal-volume per-dimension shape mismatch must raise, not silently
    # misread (e.g. a (2,2,2) prop with a (1,2,4) grid — both volume 8).
    _validate_prop_grid_shape(prop, grid, "lvm_kriging")

    # Validate data for NaN/Inf before C++ call
    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError("lvm_kriging: prop.data contains NaN or Inf")
    if not numpy.all(numpy.isfinite(mean_data)):
        raise ValueError("lvm_kriging: mean_data contains NaN or Inf")

    out_prop = _empty_clone(prop)

    okp = _HPGL_OK_PARAMS(
        covariance_type=cov_model.type,
        ranges=cov_model.ranges,
        angles=cov_model.angles,
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=valid_radiuses,
        max_neighbours=max_neighbours,
    )

    sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_lvm_kriging(
            prop.data,
            prop.mask,
            sh,
            mean_data,
            sh,
            okp,
            out_prop.data,
            out_prop.mask,
            sh,
        )
        _finalize_kriging_stats(expected, "lvm_kriging")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "lvm_kriging: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


@accepts_tuple("prop", 0)
def median_ik(prop, grid, marginal_probs, radiuses, max_neighbours, cov_model):
    """Perform Median Indicator Kriging (Median IK) on a 3D grid.

    Median IK is an optimized form of indicator kriging for two-category
    (binary) indicators. It computes a single kriging system rather than
    one per category, using the median threshold approach.

    Parameters
    ----------
    prop : IndProperty
        Input indicator property with data, mask, and indicator_count.
        Must have exactly 2 indicator categories.
    grid : SugarboxGrid
        Grid definition specifying the output resolution.
    marginal_probs : tuple of float
        Marginal probabilities ``(p0, p1)`` for the two indicator
        categories. ``p1`` should equal ``1 - p0``.
    radiuses : tuple of int
        Search radii ``(rx, ry, rz)`` in grid cells.
    max_neighbours : int
        Maximum number of neighboring points to use per cell.
    cov_model : CovarianceModel
        Covariance model defining spatial correlation.

    Returns
    -------
    ContProperty
        Output property with median IK values at all grid cells.

    Raises
    ------
    CriticalValidationError
        If grid dimensions, radiuses, max_neighbours, or covariance
        parameters are invalid.
    ValueError
        If ``marginal_probs`` does not have exactly 2 elements or any
        probability is out of range, if prop.indicator_count is not 2,
        or if prop.data contains NaN or Inf values.
    RuntimeError
        If the C++ computation produces an error (including a singular
        kriging system or non-finite output).
    """
    valid_radiuses = _validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model,
        min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
    )

    ParameterValidator.validate_property_type(prop, IndProperty, "median_ik")

    # Validate marginal_probs
    if len(marginal_probs) != 2:
        raise ValueError("median_ik: marginal_probs must have exactly 2 elements")
    for i, p in enumerate(marginal_probs):
        ParameterValidator.validate_probability(p, f"marginal_probs[{i}]")
    ParameterValidator.validate_probability_sum(marginal_probs)

    # Validate indicator_count must be exactly 2
    # The C API (hpgl_median_ik) hardcodes indicator_count=2 with an
    # interleaved data layout; other counts would read wrong cells.
    if prop.indicator_count != 2:
        raise ValueError(f"median_ik: indicator_count must be 2, got {prop.indicator_count}")

    # Validate prop.data for NaN/Inf before C++ call (defensive consistency).
    # IndProperty uses uint8 data which cannot hold NaN/Inf natively,
    # but the guard matches the pattern used by all kriging functions
    # and future-proofs against potential dtype changes.
    if not numpy.all(numpy.isfinite(
        numpy.asarray(prop.data, dtype=numpy.float32)
    )):
        raise ValueError(
            "median_ik: prop.data contains NaN or Inf values"
        )

    out_prop = _empty_clone(prop)

    miksp = _HPGL_MEDIAN_IK_PARAMS(
        covariance_type=cov_model.type,
        ranges=cov_model.ranges,
        angles=cov_model.angles,
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=valid_radiuses,
        max_neighbours=max_neighbours,
        marginal_probs=marginal_probs,
    )

    inp = _create_hpgl_ind_masked_array(prop, grid)
    outp = _create_hpgl_ind_masked_array(out_prop, grid)
    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_median_ik(inp, miksp, outp)
        # M-9: consume C++ stats so a singular system raises RuntimeError
        # instead of being silently marginal-substituted (matches every
        # sibling kriging wrapper).
        _finalize_kriging_stats(expected, "median_ik")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "median_ik: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


@accepts_tuple("prop", 0)
def indicator_kriging(prop, grid, data, marginal_probs):
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    ParameterValidator.validate_property_type(prop, IndProperty, "indicator_kriging")

    # Validate indicator count
    ParameterValidator.validate_indicator_count(len(data))

    # Validate marginal_probs
    if len(marginal_probs) != len(data):
        raise ValueError(
            f"indicator_kriging: marginal_probs length ({len(marginal_probs)}) must match data length ({len(data)})"
        )
    for i, p in enumerate(marginal_probs):
        ParameterValidator.validate_probability(p, f"marginal_probs[{i}]")
    ParameterValidator.validate_probability_sum(marginal_probs)

    # Validate per-indicator parameters. In the 2-category case only
    # data[0] is used (the median_ik redirect below discards data[1]),
    # so skip validating the unused entry — rejecting an invalid radius
    # in the ignored data[1] would contradict the documented
    # "data[1] is ignored" contract (see redirect warning below).
    validate_entries = data[:1] if len(data) == 2 else data
    for i, ikd in enumerate(validate_entries):
        ParameterValidator.validate_radius(
            ikd["radiuses"], f"data[{i}].radiuses",
            min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
        )
        ParameterValidator.validate_max_neighbors(ikd["max_neighbours"])
        ParameterValidator.validate_covariance_parameters(
            ikd["cov_model"].sill,
            ikd["cov_model"].nugget,
            ikd["cov_model"].ranges,
            ikd["cov_model"].angles,
        )

    # Validate prop.data for NaN/Inf before C++ call (defensive consistency).
    # IndProperty uses uint8 data which cannot hold NaN/Inf natively,
    # but the guard matches the pattern used by all kriging functions
    # and future-proofs against potential dtype changes.
    if not numpy.all(numpy.isfinite(
        numpy.asarray(prop.data, dtype=numpy.float32)
    )):
        raise ValueError(
            "indicator_kriging: prop.data contains NaN or Inf values"
        )

    if len(data) == 2:
        # Two-category indicator kriging is redirected to median_ik, which
        # uses a single set of covariance/radius/neighbour parameters.
        # Only data[0] configuration is used; data[1] is discarded.
        # marginal_probs[0] is passed as-is; median_ik derives p1 = 1 - p0.
        # III-12: honor the "data[1] is ignored" contract even in the
        # warning — data[1] may be a minimal dict (no radiuses/cov_model),
        # which the validation above deliberately skips (validate_entries =
        # data[:1]). .get() keeps the minimal-dict path from KeyError-ing.
        data1 = data[1]
        logger.warning(
            "indicator_kriging: 2-category case redirects to median_ik. "
            "data[1] configuration (radiuses=%s, cov_model=%s) is ignored; "
            "only data[0] params are used.",
            data1.get("radiuses", "<not provided>"),
            data1.get("cov_model", "<not provided>"),
        )
        return median_ik(
            prop,
            grid,
            (marginal_probs[0], 1 - marginal_probs[0]),
            data[0]["radiuses"],
            data[0]["max_neighbours"],
            data[0]["cov_model"],
        )
    out_prop = _empty_clone(prop)
    inp = _create_hpgl_ind_masked_array(prop, grid)
    outp = _create_hpgl_ind_masked_array(out_prop, grid)
    params = __create_hpgl_ik_params(data, len(data), False, marginal_probs)
    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_indicator_kriging(inp, outp, params, len(data))
        # M-9: consume C++ stats so a singular system raises RuntimeError
        # instead of being silently marginal-substituted (matches every
        # sibling kriging wrapper).
        _finalize_kriging_stats(expected, "indicator_kriging")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "indicator_kriging: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


@accepts_tuple("prop", 0)
def simple_cokriging_markI(
    prop,
    grid,
    radiuses,
    max_neighbours,
    cov_model,
    secondary_data,
    primary_mean,
    secondary_mean,
    secondary_variance,
    correlation_coef,
):
    valid_radiuses = _validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model,
        min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
    )

    ParameterValidator.validate_property_type(prop, ContProperty, "simple_cokriging_markI")

    # Validate primary property data (matching other kriging functions)
    if prop.data.size == 0:
        raise ValueError("simple_cokriging_markI: prop.data is empty")
    expected_size = grid.x * grid.y * grid.z
    if prop.data.size != expected_size:
        raise ValueError(
            f"simple_cokriging_markI: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )

    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError("simple_cokriging_markI: prop.data contains NaN or Inf")

    # Validate cokriging-specific parameters
    ParameterValidator.validate_correlation_coef(correlation_coef)
    ParameterValidator.validate_variance(secondary_variance, "secondary_variance")

    # Validate mean values for NaN/Inf
    if not numpy.isfinite(primary_mean):
        raise ValueError(
            f"simple_cokriging_markI: primary_mean must be finite, got {primary_mean}"
        )
    if not numpy.isfinite(secondary_mean):
        raise ValueError(
            f"simple_cokriging_markI: secondary_mean must be finite, got {secondary_mean}"
        )

    # Validate secondary_data
    ParameterValidator.validate_property_type(
        secondary_data, ContProperty, "simple_cokriging_markI",
        param_name="secondary_data", type_name="a ContProperty",
    )
    sec_size = secondary_data.data.size
    expected_size = grid.x * grid.y * grid.z
    if sec_size == 0:
        raise ValueError("simple_cokriging_markI: secondary_data.data is empty")
    if sec_size != expected_size:
        raise ValueError(
            f"simple_cokriging_markI: secondary_data size {sec_size} "
            f"does not match grid size {expected_size} "
            f"({grid.x}x{grid.y}x{grid.z})"
        )

    if not numpy.all(numpy.isfinite(secondary_data.data)):
        raise ValueError(
            "simple_cokriging_markI: secondary_data.data contains NaN or Inf"
        )

    out_prop = _empty_clone(prop)

    inp = _create_hpgl_cont_masked_array(prop, grid)
    sec = _create_hpgl_cont_masked_array(secondary_data, grid)
    outp = _create_hpgl_cont_masked_array(out_prop, grid)
    params = __checked_create(
        __hpgl_cockriging_m1_params_t,
        covariance_type=cov_model.type,
        ranges=_c_array(c_double, 3, cov_model.ranges),
        angles=_c_array(c_double, 3, cov_model.angles),
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=_c_array(c_int, 3, valid_radiuses),
        max_neighbours=max_neighbours,
        primary_mean=primary_mean,
        secondary_mean=secondary_mean,
        secondary_variance=secondary_variance,
        correlation_coef=correlation_coef,
    )
    expected = grid.x * grid.y * grid.z - int(numpy.sum(prop.mask > 0))
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_simple_cokriging_mark1(inp, sec, params, outp)
        _finalize_kriging_stats(expected, "simple_cokriging_markI")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "simple_cokriging_markI: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


def simple_cokriging_markII(
    grid, primary_data, secondary_data, correlation_coef, radiuses, max_neighbours
):
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses and max_neighbours
    ParameterValidator.validate_radius(
        radiuses, "radiuses", min_radius=ValidationConstants.MIN_KRIGING_RADIUS
    )
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate correlation coefficient
    ParameterValidator.validate_correlation_coef(correlation_coef)

    # Validate primary_data and secondary_data are dicts with required keys
    for label, d in [("primary", primary_data), ("secondary", secondary_data)]:
        if not isinstance(d, dict):
            raise validation.CriticalValidationError(
                f"simple_cokriging_markII: {label}_data must be a dict, got {type(d).__name__}",
                f"{label}_data",
            )
        for key in ("data", "cov_model", "mean"):
            if key not in d:
                raise validation.CriticalValidationError(
                    f"simple_cokriging_markII: {label}_data missing required key '{key}'",
                    f"{label}_data",
                )
        if not isinstance(d["data"], ContProperty):
            raise validation.CriticalValidationError(
                f"simple_cokriging_markII: {label}_data['data'] must be a ContProperty, "
                f"got {type(d['data']).__name__}",
                f"{label}_data.data",
            )

    # Validate both covariance models
    for _label, d in [("primary", primary_data), ("secondary", secondary_data)]:
        cm = d["cov_model"]
        ParameterValidator.validate_covariance_parameters(cm.sill, cm.nugget, cm.ranges, cm.angles)

    # Validate mean values for NaN/Inf
    for label, d in [("primary", primary_data), ("secondary", secondary_data)]:
        mean_val = d["mean"]
        if not numpy.isfinite(mean_val):
            raise ValueError(
                f"simple_cokriging_markII: {label}_data['mean'] must be finite, got {mean_val}"
            )

    # Validate data arrays for NaN/Inf before C++ call
    for label, d in [("primary", primary_data), ("secondary", secondary_data)]:
        if not numpy.all(numpy.isfinite(d["data"].data)):
            raise ValueError(
                f"simple_cokriging_markII: {label}_data['data'] contains NaN or Inf"
            )

    out_prop = _empty_clone(primary_data["data"])

    pcp = primary_data["cov_model"]
    scp = secondary_data["cov_model"]

    inp = _create_hpgl_cont_masked_array(primary_data["data"], grid)
    sec = _create_hpgl_cont_masked_array(secondary_data["data"], grid)
    outp = _create_hpgl_cont_masked_array(out_prop, grid)
    params = __checked_create(
        __hpgl_cockriging_m2_params_t,
        primary_cov_params=__checked_create(
            __hpgl_cov_params_t,
            covariance_type=pcp.type,
            ranges=_c_array(c_double, 3, pcp.ranges),
            angles=_c_array(c_double, 3, pcp.angles),
            sill=pcp.sill,
            nugget=pcp.nugget,
        ),
        secondary_cov_params=__checked_create(
            __hpgl_cov_params_t,
            covariance_type=scp.type,
            ranges=_c_array(c_double, 3, scp.ranges),
            angles=_c_array(c_double, 3, scp.angles),
            sill=scp.sill,
            nugget=scp.nugget,
        ),
        radiuses=_c_array(c_int, 3, radiuses),
        max_neighbours=max_neighbours,
        primary_mean=primary_data["mean"],
        secondary_mean=secondary_data["mean"],
        correlation_coef=correlation_coef,
    )
    expected = grid.x * grid.y * grid.z - int(
        numpy.sum(primary_data["data"].mask > 0)
    )
    with _hpgl_call_lock:
        _reset_kriging_stats()
        call_simple_cokriging_mark2(inp, sec, params, outp)
        _finalize_kriging_stats(expected, "simple_cokriging_markII")

    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "simple_cokriging_markII: output data contains NaN or Inf after C++ computation"
        )

    return out_prop


def simple_kriging_weights(
    center_point,
    n_x,
    n_y,
    n_z,
    ranges=(100000, 100000, 100000),
    sill=1,
    cov_type=covariance.exponential,
    nugget=None,
    angles=None,
):
    if angles is None:
        angles = (0, 0, 0)
    if nugget is None:
        nugget = 0

    # Validate pointset arrays have matching lengths and non-zero count
    if len(n_x) != len(n_y) or len(n_x) != len(n_z):
        raise RuntimeError(f"Invalid pointset. {len(n_x)},{len(n_y)},{len(n_z)}.")
    if len(n_x) == 0:
        raise RuntimeError("simple_kriging_weights: at least one data point is required")
    if len(n_x) > validation.ValidationConstants.MAX_NEIGHBORS:
        validation.validation_logger.warning(
            f"simple_kriging_weights: point count {len(n_x)} exceeds recommended maximum "
            f"{validation.ValidationConstants.MAX_NEIGHBORS}. Performance may be degraded."
        )

    # Validate covariance parameters
    ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)

    # Validate pointset arrays for NaN/inf
    for name, arr in [("n_x", n_x), ("n_y", n_y), ("n_z", n_z)]:
        arr_np = numpy.asarray(arr, dtype="float32")
        if numpy.any(numpy.isnan(arr_np)) or numpy.any(numpy.isinf(arr_np)):
            raise ValueError(f"simple_kriging_weights: {name} contains NaN or infinite values")

    # Validate center_point for NaN/inf
    cp = numpy.asarray(center_point, dtype="float32")
    if numpy.any(numpy.isnan(cp)) or numpy.any(numpy.isinf(cp)):
        raise ValueError("simple_kriging_weights: center_point contains NaN or infinite values")

    covp = __checked_create(
        __hpgl_cov_params_t,
        covariance_type=cov_type,
        ranges=_c_array(c_double, 3, ranges),
        angles=_c_array(c_double, 3, angles),
        sill=sill,
        nugget=nugget,
    )

    weights = numpy.array([0] * len(n_x), dtype="float32")

    with _hpgl_call_lock:
        _reset_kriging_stats()
        rc = call_simple_kriging_weights(
            _c_array(c_float, 3, center_point),
            numpy.array(n_x, dtype="float32"),
            numpy.array(n_y, dtype="float32"),
            numpy.array(n_z, dtype="float32"),
            len(n_x),
            covp,
            weights,
        )
    if rc != 0:
        raise RuntimeError(
            "simple_kriging_weights failed: "
            + call_get_last_exception_message().decode("utf-8", errors="replace")
        )

    # M-19: dpotrs_ reports no error for near-singular systems — a success
    # return code can still yield NaN weights, which would propagate
    # silently. Validate finite output, consistent with the other kriging
    # wrapper families.
    if not numpy.all(numpy.isfinite(weights)):
        raise RuntimeError(
            "simple_kriging_weights: output weights contain NaN or Inf after C++ computation"
        )

    return weights


def get_gslib_property(prop_dict, prop_name, undefined_value):
    """Extract a property and its mask from a GSLIB property dictionary.

    Parameters
    ----------
    prop_dict : dict
        Dictionary mapping property names to numpy arrays.
    prop_name : str
        Name of the property to extract.
    undefined_value : float
        Value marking undefined/uninformed cells.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Property array and uint8 mask array where 1 = informed, 0 = masked.

    Raises
    ------
    TypeError
        If prop_dict is not a dict.
    KeyError
        If prop_name is not in prop_dict.
    """
    ParameterValidator.validate_property_type(
        prop_dict, dict, "get_gslib_property",
        param_name="prop_dict", type_name="a dict",
    )
    prop = prop_dict[prop_name]
    # L-56: a non-ndarray value (e.g. a list) previously surfaced a raw
    # AttributeError from prop.shape. Raise a clear TypeError instead.
    if not isinstance(prop, numpy.ndarray):
        raise TypeError(
            f"get_gslib_property: property '{prop_name}' must be a numpy array, "
            f"got {type(prop).__name__}"
        )
    informed_array = numpy.zeros(prop.shape, dtype=numpy.uint8)
    # F-M18: GSLIB missing-value trimming — values outside the ±1.0e21
    # window (strict inequality per the GSLIB convention; see gslib_ref.py
    # reference-fact table, got-20260802092630) are missing sentinels in
    # addition to exact undefined_value matches, matching the C++ fast
    # reader (read_inc_file.cpp:287-305). NaN sentinels are handled
    # separately since NaN != NaN.
    out_of_window = is_gslib_missing_sentinel(prop)
    if numpy.isnan(undefined_value):
        uninformed = numpy.isnan(prop) | out_of_window
    else:
        # E2-06/R-06: exact float32 re-mask, mirroring the C++ fast reader
        # (read_inc_file.cpp:379 float32 ==). The %.9E writer
        # (property_writer.cpp:121) round-trips float32 exactly, so a
        # sentinel cell written by HPGL always re-masks and near-sentinel
        # real data (e.g. -99.00005 vs -99.0) stays informed — the earlier
        # 1e-6 relative band over-masked the latter class (calibrated to
        # the pre-fix %E-6 writer) and diverged from the C++ side. prop is
        # already float32; the explicit conversion of undefined_value keeps
        # the comparison in float32 arithmetic.
        uninformed = (prop == numpy.float32(undefined_value)) | out_of_window
    informed_array = numpy.where(uninformed, 0, 1).astype(numpy.uint8)
    return (prop_dict[prop_name], informed_array)


def set_output_handler(handler, param):
    """Set or clear the HPGL C++ output handler callback.

    Parameters
    ----------
    handler : callable or None
        Output handler function accepting (message: str, param) and
        returning int. If None, clears the output handler.
    param : any
        User parameter forwarded to the handler callback.

    Raises
    ------
    TypeError
        If handler is not callable and not None.

    Notes
    -----
    The C++ handler pointer is stored by atomic store, but the handler+param
    pair is written in two separate stores — there is a brief window where a
    concurrent C++ reader could see a mismatched handler/param pair.  This
    function acquires ``_hpgl_call_lock`` to serialise handler mutation with
    other HPGL calls, and keeps the old CFUNCTYPE object alive until the new
    handler has been installed to prevent use-after-free of the C function
    pointer.

    Reentrancy (F-M22): handlers are invoked from the C++ engine while a
    kriging/simulation FFI call holds ``_hpgl_call_lock``. The C++ engine
    only invokes the handler on the OpenMP master thread (output.cpp F-M22),
    so a handler that calls back into a kriging/simulation API on the same
    thread re-enters the lock legally (RLock). Handlers MUST NOT, however,
    spawn work on other threads that then call kriging/simulation APIs —
    that would block on ``_hpgl_call_lock`` held by the calling thread while
    the C++ engine waits for the handler, deadlocking the call.
    """
    global _h, _output_handler_param
    if handler is not None and not callable(handler):
        raise TypeError(
            f"set_output_handler: handler must be callable or None, got {type(handler).__name__}"
        )
    _hpgl_call_lock.acquire()
    try:
        if handler is None:
            old_h = _h
            old_param = _output_handler_param
            call_set_output_handler(None, None)
            _h = None
            _output_handler_param = None
            # Defer the cleared CFUNCTYPE+param so a concurrent kriging
            # call cannot invoke a freed trampoline (mirror the set path).
            if old_h is not None:
                _old_handler_refs.append((old_h, old_param))
                if len(_old_handler_refs) > _OLD_HANDLER_REFS_CAP:
                    _old_handler_refs.pop(0)
        else:
            # Keep old handler objects alive during the transition so the
            # CFUNCTYPE thunk is not freed while C++ may still reference it.
            old_h = _h
            old_param = _output_handler_param
            new_h = hpgl_output_handler(handler)
            _h = new_h
            _output_handler_param = param
            call_set_output_handler(new_h, param)
            # Hold old handler references in deferred cache to prevent
            # CFUNCTYPE trampoline use-after-free in concurrent kriging
            # calls. Kriging/simulation FFI calls hold _hpgl_call_lock, so
            # a concurrent clear cannot run mid-call; deferring deletion
            # additionally keeps the trampoline alive across the
            # replacement (and the C++ read-then-invoke window).
            _old_handler_refs.append((old_h, old_param))
            if len(_old_handler_refs) > _OLD_HANDLER_REFS_CAP:
                _old_handler_refs.pop(0)
    finally:
        _hpgl_call_lock.release()


def set_progress_handler(handler, param):
    """Set or clear the HPGL C++ progress handler callback.

    Parameters
    ----------
    handler : callable or None
        Progress handler function accepting (message: str, percent: int, param)
        and returning int. If None, clears the progress handler.
    param : any
        User parameter forwarded to the handler callback.

    Raises
    ------
    TypeError
        If handler is not callable and not None.

    Notes
    -----
    The C++ handler pointer is stored by atomic store, but the handler+param
    pair is written in two separate stores — there is a brief window where a
    concurrent C++ reader could see a mismatched handler/param pair.  This
    function acquires ``_hpgl_call_lock`` to serialise handler mutation with
    other HPGL calls, and keeps the old CFUNCTYPE object alive until the new
    handler has been installed to prevent use-after-free of the C function
    pointer.

    Reentrancy (F-M22): same contract as :func:`set_output_handler` — handlers
    run on the OpenMP master thread during a lock-holding FFI call; same-thread
    re-entry of kriging APIs is legal (RLock), cross-thread re-entry deadlocks.
    """
    global _progress_handler, _progress_handler_param
    if handler is not None and not callable(handler):
        raise TypeError(
            f"set_progress_handler: handler must be callable or None, got {type(handler).__name__}"
        )
    _hpgl_call_lock.acquire()
    try:
        if handler is None:
            old_h = _progress_handler
            old_param = _progress_handler_param
            call_set_progress_handler(None, None)
            _progress_handler = None
            _progress_handler_param = None
            # Defer the cleared CFUNCTYPE+param so a concurrent simulation
            # call cannot invoke a freed trampoline (mirror the set path).
            if old_h is not None:
                _old_handler_refs.append((old_h, old_param))
                if len(_old_handler_refs) > _OLD_HANDLER_REFS_CAP:
                    _old_handler_refs.pop(0)
        else:
            # Keep old handler objects alive during the transition so the
            # CFUNCTYPE thunk is not freed while C++ may still reference it.
            old_h = _progress_handler
            old_param = _progress_handler_param
            new_h = hpgl_progress_handler(handler)
            _progress_handler = new_h
            _progress_handler_param = param
            call_set_progress_handler(new_h, param)
            # Hold old handler references in deferred cache to prevent
            # CFUNCTYPE trampoline use-after-free in concurrent simulation
            # calls (same reasoning as set_output_handler).
            _old_handler_refs.append((old_h, old_param))
            if len(_old_handler_refs) > _OLD_HANDLER_REFS_CAP:
                _old_handler_refs.pop(0)
    finally:
        _hpgl_call_lock.release()


__all__ = [
    "ContProperty",
    "IndProperty",
    "SugarboxGrid",
    "CovarianceModel",
    "covariance",
    "checkFWA",
    "append_mask",
    "accepts_tuple",
    "ordinary_kriging",
    "simple_kriging",
    "lvm_kriging",
    "indicator_kriging",
    "median_ik",
    "simple_cokriging_markI",
    "simple_cokriging_markII",
    "simple_kriging_weights",
    "write_property",
    "write_gslib_property",
    "load_cont_property",
    "load_ind_property",
    "read_inc_file_float",
    "read_inc_file_byte",
    "calc_mean",
    "set_thread_num",
    "get_thread_num",
    "get_gslib_property",
    "set_output_handler",
    "set_progress_handler",
]
