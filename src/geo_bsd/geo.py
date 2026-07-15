# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import contextlib
import ctypes as C
import functools
import logging
import os
import threading

import numpy

# Import validation framework
from . import validation
from .hpgl_wrap import (
    _HPGL_CONT_MASKED_ARRAY,
    _HPGL_FLOAT_ARRAY,
    _HPGL_IK_PARAMS,
    _HPGL_IND_MASKED_ARRAY,
    _HPGL_MEDIAN_IK_PARAMS,
    _HPGL_OK_PARAMS,
    _HPGL_SHAPE,
    _HPGL_SK_PARAMS,
    _HPGL_UBYTE_ARRAY,
    __hpgl_cockriging_m1_params_t,
    __hpgl_cockriging_m2_params_t,
    __hpgl_cov_params_t,
    _hpgl_so,
    hpgl_output_handler,
    hpgl_progress_handler,
)
from .validation import (
    GridValidator,
    ParameterValidator,
    PathValidator,
)

# Module-level state to prevent stale C++ error propagation between tests.
# The HPGL C++ DLL stores the last exception message globally and does not
# clear it after it's read. Without tracking, an error from one test (e.g.
# a failed read_inc_file_byte) would cause all subsequent _check_hpgl_error
# calls to falsely report failure.
#
# Using thread-local storage so that concurrent HPGL FFI calls in different
# threads each track their own pre/post error snapshot independently.
# The lock synchronizes access to thread-local snapshot storage.
_error_local = threading.local()
_error_snapshot_lock = threading.Lock()

# Serializes C++ HPGL calls with error checking to prevent cross-thread
# error races. The C++ error state is global (not thread_local), so
# concurrent calls from different threads can corrupt error reporting.
# This lock ensures only one thread is in the snapshot→C++ call→check
# window at a time. This is a Python-side mitigation; the proper fix
# requires making the C++ error state thread_local.
_hpgl_call_lock = threading.Lock()


@contextlib.contextmanager
def _hpgl_error_guard(context=""):
    """Serialize a C++ HPGL call with error checking to prevent cross-thread races.

    Usage::

        with _hpgl_error_guard("ordinary_kriging"):
            _hpgl_so.hpgl_ordinary_kriging(...)

    Acquires ``_hpgl_call_lock`` before the snapshot, holds it across the
    C++ call window, and releases it in ``finally``.  Error checking
    happens inside the ``try`` block so that a ``RuntimeError`` from a
    new C++ error does not prevent lock release.
    """
    _hpgl_call_lock.acquire()
    _snapshot_hpgl_error()
    try:
        yield
        _check_hpgl_error(context)
    finally:
        _hpgl_call_lock.release()


logger = logging.getLogger(__name__)

# Maximum number of elements for slow Python-based file parsers.
# Files exceeding this limit should use the fast C++ reader instead
# (specify `size` parameter in load_cont_property / load_ind_property).
_MAX_SLOW_PARSER_ELEMENTS = 10_000_000


def _snapshot_hpgl_error():
    """
    Take a snapshot of the current HPGL C++ error state.

    Call this BEFORE invoking any C++ function that might succeed.
    _check_hpgl_error will compare the post-call error against this snapshot
    to detect only NEW errors, avoiding stale error propagation.

    Stores the snapshot in thread-local storage under _error_snapshot_lock.
    The lock protects the snapshot write against concurrent snapshot reads
    in _check_hpgl_error.
    """
    with _error_snapshot_lock:
        _error_local._hpgl_error_snapshot = _hpgl_so.hpgl_get_last_exception_message()


def _check_hpgl_error(context=""):
    """
    Check for NEW HPGL C++ errors after a computation call.

    Compares current error state against a pre-call snapshot (set via
    _snapshot_hpgl_error). Raises RuntimeError ONLY if the error message
    has changed since the snapshot, indicating a new error from the
    current operation rather than a stale error from a previous call.

    Uses thread-local storage so concurrent HPGL operations in different
    threads each track their own pre/post error state independently.

    The entire read-compare-raise sequence is atomic under
    _error_snapshot_lock, preventing races between concurrent calls
    to this function that access the same thread-local snapshot.
    However, between _snapshot_hpgl_error releasing the lock and this
    function acquiring it, another thread's C++ call may update the
    global HPGL error state. In practice, snapshot identity comparison
    (byte-for-byte match) combined with the C++ library's inclusion
    of __FILE__:__LINE__ in error messages makes false-positive
    RuntimeErrors infeasible.

    Args:
        context: Description of the operation being checked (e.g. "ordinary_kriging")

    Raises:
        RuntimeError: If the C++ computation produced a new error
    """
    with _error_snapshot_lock:
        err = _hpgl_so.hpgl_get_last_exception_message()
        if err is not None and len(err) > 0:
            snapshot = getattr(_error_local, "_hpgl_error_snapshot", None)
            if err == snapshot:
                # Error unchanged from pre-call snapshot — C++ call did
                # not produce a new error. Always suppress stale errors.
                return
            # Genuine new error (different from pre-call snapshot)
            err_str = err.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{context} failed: {err_str}" if context else f"HPGL error: {err_str}"
            )


# Module-level handler references initialized to None
_h = None
_progress_handler = None
# Store param references to prevent garbage collection while C++ holds void*.
# Without these, the caller dropping param would create a dangling pointer.
_output_handler_param = None
_progress_handler_param = None


def _c_array(ar_type, size, values):
    if len(values) != size:
        raise RuntimeError(f"{len(values)} values specified for array of {size} elements")
    result = (ar_type * size)(*values)
    # Preserve references to input values to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = tuple(values)
    return result


def _create_hpgl_shape(shape, strides=None):
    # Normalize shape to 3D tuple
    if len(shape) == 1:
        shape = (shape[0], 1, 1)
    elif len(shape) == 2:
        shape = (shape[0], shape[1], 1)

    if strides is None:
        # C-order strides (row-major) to match C++ indexing: z * x * y + y * x + x
        # The strides array is (stride_x, stride_y, stride_z)
        return _HPGL_SHAPE(
            m_data=_c_array(C.c_int, 3, shape),
            m_strides=_c_array(C.c_int, 3, (1, shape[0], shape[0] * shape[1])),
        )
    else:
        return _HPGL_SHAPE(
            m_data=_c_array(C.c_int, 3, shape), m_strides=_c_array(C.c_int, 3, strides)
        )


def __get_strides(prop):
    ndim = prop.ndim
    if ndim == 1:
        return (1, prop.shape[0], prop.shape[0])
    elif ndim == 2:
        return (1, prop.shape[0], prop.shape[0] * prop.shape[1])
    else:  # ndim == 3
        return (
            prop.strides[0] // prop.itemsize,
            prop.strides[1] // prop.itemsize,
            prop.strides[2] // prop.itemsize,
        )


def __checked_create(T, **kargs):
    fields = []
    for f, _ in T._fields_:
        fields.append(f)
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
    if fields:
        raise validation.CriticalValidationError(
            f"No values for parameters: {fields}", "ctypes_struct"
        )
    return T(**kargs)


def checkFWA(a):
    """
    Checks for fortran-order, writable and aligned flags.
    """
    if not (a.flags["F"] and a.flags["W"] and a.flags["A"]):
        raise RuntimeError(
            f"Array checkFWA failed: F={a.flags['F']}, W={a.flags['W']}, A={a.flags['A']}"
        )


def _create_hpgl_cont_masked_array(prop, grid):
    if grid is None:
        sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
    else:
        # Use actual NumPy strides if array is 3D, otherwise compute strides from grid
        if prop.data.ndim == 3:
            sh = _create_hpgl_shape((grid.x, grid.y, grid.z), __get_strides(prop.data))
        else:
            # For 1D arrays, compute expected strides based on grid dimensions
            sh = _create_hpgl_shape((grid.x, grid.y, grid.z))
        if grid.x * grid.y * grid.z != prop.data.size:
            raise RuntimeError(
                f"Invalid data size. Size of data = {prop.data.size}. "
                f"Size of grid = {grid.x * grid.y * grid.z}"
            )

    # Security: Keep references to arrays to prevent use-after-free
    # The C code will hold pointers to this memory, so we must ensure
    # the Python objects aren't garbage collected while in use
    result = _HPGL_CONT_MASKED_ARRAY(
        data=prop.data.ctypes.data_as(C.POINTER(C.c_float)),
        mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
        shape=sh,
    )

    # Store array references on the result object to prevent garbage collection
    # This ensures the arrays live as long as the C structure does
    result._array_refs = (prop.data, prop.mask)

    return result


def _create_hpgl_ind_masked_array(prop, grid):
    if grid is None:
        sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
        assert prop.data.strides == prop.mask.strides
    else:
        sh = _create_hpgl_shape((grid.x, grid.y, grid.z))
        if grid.x * grid.y * grid.z != prop.data.size:
            raise RuntimeError(
                f"Invalid data size. Size of data = {prop.data.size}. "
                f"Size of grid = {grid.x * grid.y * grid.z}"
            )

    # Security: Keep references to arrays to prevent use-after-free
    result = _HPGL_IND_MASKED_ARRAY(
        data=prop.data.ctypes.data_as(C.POINTER(C.c_ubyte)),
        mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
        shape=sh,
        indicator_count=prop.indicator_count,
    )

    # Store array references to prevent garbage collection while C code uses them
    result._array_refs = (prop.data, prop.mask)

    return result


def _create_hpgl_ubyte_array(array, grid):
    checkFWA(array)
    if grid is None:
        sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
    else:
        sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

    # Security: Keep array reference to prevent use-after-free
    result = _HPGL_UBYTE_ARRAY(data=array.ctypes.data_as(C.POINTER(C.c_ubyte)), shape=sh)
    result._array_ref = array
    return result


def _create_hpgl_float_array(array, grid):
    checkFWA(array)
    if grid is None:
        sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
    else:
        sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

    # Security: Keep array reference to prevent use-after-free
    result = _HPGL_FLOAT_ARRAY(data=array.ctypes.data_as(C.POINTER(C.c_float)), shape=sh)
    result._array_ref = array
    return result


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

    def __init__(self, data: numpy.ndarray, mask: numpy.ndarray):
        self.data = numpy.require(data, "float32", "F")
        self.mask = numpy.require(mask, "uint8", "F")

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
                self.data = self.data.reshape((grid.x, grid.y, grid.z), order="F")
        if self.mask.ndim != 3:
            if self.mask.size == grid.x * grid.y * grid.z:
                self.mask = self.mask.reshape((grid.x, grid.y, grid.z), order="F")

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        elif idx == 1:
            return self.mask
        else:
            raise RuntimeError("Index out of range.")


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

    def __init__(self, data: numpy.ndarray, mask: numpy.ndarray, indicator_count: int):
        self.data = numpy.require(data, "uint8", "F")
        self.mask = numpy.require(mask, "uint8", "F")
        self.indicator_count = indicator_count
        if numpy.sum(numpy.bitwise_and((mask > 0), (data >= indicator_count))) > 0:
            raise RuntimeError(
                "Property contains some indicators outside of [0..%s] range."
                % (indicator_count - 1)
            )
        if data.shape != mask.shape:
            raise ValueError(f"Data shape {data.shape} does not match mask shape {mask.shape}")

    def validate(self):
        checkFWA(self.data)
        checkFWA(self.mask)
        if self.data.shape != self.mask.shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match mask shape {self.mask.shape}"
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
        Anisotropy ranges ``(rx, ry, rz)``. Default ``(0, 0, 0)``.
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
        ranges: tuple = (0, 0, 0),
        angles: tuple = (0.0, 0.0, 0.0),
        sill: float = 1.0,
        nugget: float = 0.0,
    ):
        self.type = type
        self.ranges = ranges
        self.angles = angles
        self.sill = sill
        self.nugget = nugget

        # Validate covariance parameters
        ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)


def _load_prop_cont_slow(filename, undefined_value):
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=False
    )

    values = []
    mask = []
    skipped_count = 0
    element_count = 0
    # Use validated path and explicit encoding
    with open(safe_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("--"):
                continue
            # Detect INC format end-of-data marker '/' — stop parsing.
            # The C++ writer emits '/' after all data values.
            if line.strip().startswith("/"):
                break
            for part in line.split():
                if element_count >= _MAX_SLOW_PARSER_ELEMENTS:
                    raise MemoryError(
                        f"_load_prop_cont_slow: file exceeds {_MAX_SLOW_PARSER_ELEMENTS} elements. "
                        f"Use fast C++ reader by specifying `size` parameter."
                    )
                try:
                    val = float(part.strip())
                    values.append(val)
                    element_count += 1
                    if val == undefined_value:
                        mask.append(0)
                    else:
                        mask.append(1)
                except (ValueError, TypeError):
                    skipped_count += 1
    if skipped_count > 0:
        logger.warning(
            "_load_prop_cont_slow: skipped %d non-numeric tokens in %s", skipped_count, safe_path
        )

    return ContProperty(numpy.array(values, dtype="float32"), numpy.array(mask, dtype="uint8"))


def _load_prop_ind_slow(filename, undefined_value, ind_values):
    dict_map = {}
    for i in range(len(ind_values)):
        dict_map[ind_values[i]] = i

    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=False
    )

    values = []
    mask = []
    skipped_parse = 0
    unknown_values = set()
    element_count = 0

    # Use validated path and explicit encoding
    with open(safe_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("--"):
                continue
            # Detect INC format end-of-data marker '/' — stop parsing.
            # The C++ writer emits '/' after all data values.
            if line.strip().startswith("/"):
                break
            for part in line.split():
                if element_count >= _MAX_SLOW_PARSER_ELEMENTS:
                    raise MemoryError(
                        f"_load_prop_ind_slow: file exceeds {_MAX_SLOW_PARSER_ELEMENTS} elements. "
                        f"Use fast C++ reader by specifying `size` parameter."
                    )
                try:
                    val = int(part.strip())
                except (ValueError, TypeError):
                    skipped_parse += 1
                    continue
                if val == undefined_value:
                    values.append(255)
                    mask.append(0)
                    element_count += 1
                elif val in dict_map:
                    values.append(dict_map[val])
                    mask.append(1)
                    element_count += 1
                else:
                    unknown_values.add(val)
                    element_count += 1

    if skipped_parse > 0:
        logger.warning(
            "_load_prop_ind_slow: skipped %d unparseable tokens in %s", skipped_parse, safe_path
        )
    if unknown_values:
        sorted_unknown = sorted(unknown_values, key=int)
        raise ValueError(
            f"_load_prop_ind_slow: unknown indicator value(s) {sorted_unknown} "
            f"found in {safe_path}. Expected indicator values: {list(ind_values)}"
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
def write_property(prop, filename, prop_name, undefined_value, indicator_values=None):
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

    Raises
    ------
    RuntimeError
        If the C++ write operation fails.

    See Also
    --------
    write_gslib_property : Write property in GSLIB format.
    """
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=False
    )

    if indicator_values is None:
        indicator_values = []
    if not isinstance(indicator_values, list):
        raise TypeError(
            f"write_property: indicator_values must be a list, got {type(indicator_values).__name__}"
        )

    if prop.data.ndim == 3:
        sh = _create_hpgl_shape(prop.data.shape)
    else:
        sh = _create_hpgl_shape((prop.data.size, 1, 1))
    if isinstance(prop, ContProperty):
        marr = _HPGL_CONT_MASKED_ARRAY(
            data=prop.data.ctypes.data_as(C.POINTER(C.c_float)),
            mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
            shape=sh,
        )
        # Security: Keep array references to prevent use-after-free
        marr._array_refs = (prop.data, prop.mask)
        with _hpgl_error_guard("write_property"):
            rc = _hpgl_so.hpgl_write_inc_file_float(
                safe_path.encode("utf-8"), C.byref(marr), undefined_value, prop_name.encode("utf-8")
            )
            if rc != 0:
                raise RuntimeError(
                    "write_property failed: "
                    + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
                )
    else:
        # Security: Keep reference to indicator_values array
        ind_arr = numpy.array(indicator_values, dtype="uint8")
        marr = _HPGL_IND_MASKED_ARRAY(
            data=prop.data.ctypes.data_as(C.POINTER(C.c_ubyte)),
            mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
            shape=sh,
            indicator_count=prop.indicator_count,
        )
        # Security: Keep array references to prevent use-after-free
        marr._array_refs = (prop.data, prop.mask, ind_arr)
        with _hpgl_error_guard("write_property"):
            rc = _hpgl_so.hpgl_write_inc_file_byte(
                safe_path.encode("utf-8"),
                C.byref(marr),
                undefined_value,
                prop_name.encode("utf-8"),
                ind_arr.ctypes.data_as(C.POINTER(C.c_ubyte)),
                len(indicator_values),
            )
            if rc != 0:
                raise RuntimeError(
                    "write_property failed: "
                    + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
                )


@accepts_tuple("prop", 0)
def write_gslib_property(prop, filename, prop_name, undefined_value, indicator_values=None):
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=False
    )

    if indicator_values is None:
        indicator_values = []
    if not isinstance(indicator_values, list):
        raise TypeError(
            f"write_gslib_property: indicator_values must be a list, "
            f"got {type(indicator_values).__name__}"
        )

    if isinstance(prop, ContProperty):
        with _hpgl_error_guard("write_gslib_property"):
            rc = _hpgl_so.hpgl_write_gslib_cont_property(
                _create_hpgl_cont_masked_array(prop, None),
                safe_path.encode("utf-8"),
                prop_name.encode("utf-8"),
                undefined_value,
            )
            if rc != 0:
                raise RuntimeError(
                    "write_gslib_property failed: "
                    + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
                )
    else:
        with _hpgl_error_guard("write_gslib_property"):
            rc = _hpgl_so.hpgl_write_gslib_byte_property(
                _create_hpgl_ind_masked_array(prop, None),
                safe_path.encode("utf-8"),
                prop_name.encode("utf-8"),
                undefined_value,
                _c_array(C.c_ubyte, len(indicator_values), indicator_values),
                len(indicator_values),
            )
            if rc != 0:
                raise RuntimeError(
                    "write_gslib_property failed: "
                    + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
                )


def load_cont_property(filename, undefined_value, size=None):
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

    Returns
    -------
    ContProperty
        Loaded property with data and mask arrays.

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
    # Validate filename for security
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=True
    )

    if size is None:
        logger.warning(
            "load_cont_property: Size is not specified. Using slow Python-based parser "
            "that loads entire file into memory unbounded. For large files (>100MB) "
            "specify size to use the fast C++ reader which uses pre-allocated buffers."
        )
        return _load_prop_cont_slow(safe_path, undefined_value)
    else:
        try:
            return read_inc_file_float(safe_path, undefined_value, size)
        except RuntimeError as e:
            logger.warning(
                "load_cont_property: Fast C++ reader failed, falling back to slow "
                "Python parser. C++ error: %s",
                e,
            )
            return _load_prop_cont_slow(safe_path, undefined_value)


def read_inc_file_float(filename, undefined_value, size):
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=True
    )

    # Validate size parameters
    if isinstance(size, tuple) and len(size) == 3:
        GridValidator.validate_grid_dimensions(size[0], size[1], size[2])

    total_elements = (
        size[0] * size[1] * size[2] if isinstance(size, tuple) and len(size) == 3 else size
    )
    if total_elements > 2147483647:
        raise ValueError(
            f"Grid too large: {total_elements} elements exceeds c_int max (2147483647)"
        )
    data = numpy.zeros(total_elements, dtype="float32", order="F")
    mask = numpy.zeros(total_elements, dtype="uint8", order="F")

    with _hpgl_error_guard("read_inc_file_float"):
        rc = _hpgl_so.hpgl_read_inc_file_float(
            safe_path.encode("utf-8"), undefined_value, total_elements, data, mask
        )
    if rc != 0:
        raise RuntimeError(
            "read_inc_file_float failed: "
            + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
        )

    return ContProperty(data, mask)


def read_inc_file_byte(filename, undefined_value, size, indicator_values):
    # Security: Validate filename to prevent directory traversal attacks
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=True
    )

    # Validate size parameters
    if isinstance(size, tuple) and len(size) == 3:
        GridValidator.validate_grid_dimensions(size[0], size[1], size[2])

    total_elements = (
        size[0] * size[1] * size[2] if isinstance(size, tuple) and len(size) == 3 else size
    )
    if total_elements > 2147483647:
        raise ValueError(
            f"Grid too large: {total_elements} elements exceeds c_int max (2147483647)"
        )
    data = numpy.zeros(total_elements, dtype="uint8", order="F")
    mask = numpy.zeros(total_elements, dtype="uint8", order="F")
    with _hpgl_error_guard("read_inc_file_byte"):
        rc = _hpgl_so.hpgl_read_inc_file_byte(
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
            + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
        )
    return IndProperty(data, mask, len(indicator_values))


def load_ind_property(filename, undefined_value, indicator_values, size=None):
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

    Returns
    -------
    IndProperty
        Loaded indicator property with data, mask, and indicator_count
        set to ``len(indicator_values)``.

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
    # Validate filename for security
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=True
    )

    if size is None:
        logger.warning(
            "load_ind_property: Size is not specified. Using slow Python-based parser "
            "that loads entire file into memory unbounded. For large files (>100MB) "
            "specify size to use the fast C++ reader which uses pre-allocated buffers."
        )
        return _load_prop_ind_slow(safe_path, undefined_value, indicator_values)
    else:
        try:
            return read_inc_file_byte(safe_path, undefined_value, size, indicator_values)
        except RuntimeError as e:
            logger.warning(
                "load_ind_property: Fast C++ reader failed, falling back to slow "
                "Python parser. C++ error: %s",
                e,
            )
            return _load_prop_ind_slow(safe_path, undefined_value, indicator_values)


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
    if not isinstance(num, int):
        raise TypeError(f"set_thread_num: num must be an integer, got {type(num).__name__}")
    if num < 1:
        raise ValueError(f"set_thread_num: num must be at least 1, got {num}")
    # Sanity-check: warn if num exceeds available CPU count
    import os

    cpu_count = os.cpu_count()
    if cpu_count is not None and num > cpu_count * 4:
        logger.warning(
            "set_thread_num: num=%d exceeds 4x CPU count (%d). "
            "This may cause thread oversubscription and performance degradation.",
            num,
            cpu_count,
        )
    _hpgl_so.hpgl_set_thread_num(num)


def get_thread_num():
    return _hpgl_so.hpgl_get_thread_num()


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
    masked = numpy.ma.masked_where(prop.mask == 0, prop.data)
    if masked.count() == 0:
        raise ValueError("calc_mean: no informed values (all masked)")
    return float(masked.mean())


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
    RuntimeError
        If the C++ computation produces an error.
    """
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses
    valid_radiuses = ParameterValidator.validate_radius(radiuses, "radiuses")

    # Validate max_neighbours
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

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

    with _hpgl_error_guard("ordinary_kriging"):
        _hpgl_so.hpgl_ordinary_kriging(
            _create_hpgl_cont_masked_array(prop, grid),
            C.byref(okp),
            _create_hpgl_cont_masked_array(out_prop, grid),
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
    RuntimeError
        If the C++ computation produces an error.
    """
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses
    valid_radiuses = ParameterValidator.validate_radius(radiuses, "radiuses")

    # Validate max_neighbours
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

    # Validate property data size against grid
    if prop.data.size == 0:
        raise ValueError("simple_kriging: prop.data is empty")
    expected_size = grid.x * grid.y * grid.z
    if prop.data.size != expected_size:
        raise ValueError(
            f"simple_kriging: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )

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

    with _hpgl_error_guard("simple_kriging"):
        _hpgl_so.hpgl_simple_kriging(
            prop.data, prop.mask, C.byref(sh), C.byref(skp), out_prop[0], out_prop[1], C.byref(sh)
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
        If ``mean_data`` is not a NumPy array or its size doesn't match
        the grid.
    RuntimeError
        If the C++ computation produces an error.
    """
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses
    valid_radiuses = ParameterValidator.validate_radius(radiuses, "radiuses")

    # Validate max_neighbours
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

    # Validate mean_data
    if not isinstance(mean_data, numpy.ndarray):
        raise ValueError("lvm_kriging: mean_data must be a numpy array")
    if mean_data.dtype != numpy.float32:
        mean_data = numpy.require(mean_data, dtype="float32")
    expected_size = grid.x * grid.y * grid.z
    if mean_data.size != expected_size:
        raise ValueError(
            f"lvm_kriging: mean_data size {mean_data.size} does not match "
            f"grid dimensions {grid.x}x{grid.y}x{grid.z} = {expected_size}"
        )

    # Validate property data size against grid
    if prop.data.size == 0:
        raise ValueError("lvm_kriging: prop.data is empty")
    if prop.data.size != expected_size:
        raise ValueError(
            f"lvm_kriging: prop.data size {prop.data.size} does not match "
            f"grid size {expected_size} ({grid.x}x{grid.y}x{grid.z})"
        )

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

    with _hpgl_error_guard("lvm_kriging"):
        _hpgl_so.hpgl_lvm_kriging(
            prop.data,
            prop.mask,
            C.byref(sh),
            mean_data,
            C.byref(sh),
            C.byref(okp),
            out_prop.data,
            out_prop.mask,
            C.byref(sh),
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
        If ``marginal_probs`` does not have exactly 2 elements or
        any probability is out of range.
    RuntimeError
        If the C++ computation produces an error.
    """
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses
    valid_radiuses = ParameterValidator.validate_radius(radiuses, "radiuses")

    # Validate max_neighbours
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

    # Validate marginal_probs
    if len(marginal_probs) != 2:
        raise ValueError("median_ik: marginal_probs must have exactly 2 elements")
    for i, p in enumerate(marginal_probs):
        ParameterValidator.validate_probability(p, f"marginal_probs[{i}]")

    # Validate indicator_count must be exactly 2
    # The C API (hpgl_median_ik) hardcodes indicator_count=2 with an
    # interleaved data layout; other counts would read wrong cells.
    if prop.indicator_count != 2:
        raise ValueError(f"median_ik: indicator_count must be 2, got {prop.indicator_count}")

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
    with _hpgl_error_guard("median_ik"):
        _hpgl_so.hpgl_median_ik(C.byref(inp), C.byref(miksp), C.byref(outp))
    return out_prop


def __create_hpgl_ik_params(data, indicator_count, is_lvm, marginal_probs):
    ikps = []
    assert len(data) == indicator_count
    for i in range(indicator_count):
        ikd = data[i]
        ikp = __checked_create(
            _HPGL_IK_PARAMS,
            covariance_type=ikd["cov_model"].type,
            ranges=_c_array(C.c_double, 3, ikd["cov_model"].ranges),
            angles=_c_array(C.c_double, 3, ikd["cov_model"].angles),
            sill=ikd["cov_model"].sill,
            nugget=ikd["cov_model"].nugget,
            radiuses=_c_array(C.c_int, 3, tuple(int(r) for r in ikd["radiuses"])),
            max_neighbours=ikd["max_neighbours"],
            marginal_prob=0 if is_lvm else marginal_probs[i],
        )
        ikps.append(ikp)
    return _c_array(_HPGL_IK_PARAMS, indicator_count, ikps)


@accepts_tuple("prop", 0)
def indicator_kriging(prop, grid, data, marginal_probs):
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

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

    # Validate per-indicator parameters
    for i, ikd in enumerate(data):
        ParameterValidator.validate_radius(ikd["radiuses"], f"data[{i}].radiuses")
        ParameterValidator.validate_max_neighbors(ikd["max_neighbours"])
        ParameterValidator.validate_covariance_parameters(
            ikd["cov_model"].sill,
            ikd["cov_model"].nugget,
            ikd["cov_model"].ranges,
            ikd["cov_model"].angles,
        )

    if len(data) == 2:
        return median_ik(
            prop,
            grid,
            (marginal_probs[0], 1 - marginal_probs[0]),
            data[0]["radiuses"],
            data[0]["max_neighbours"],
            data[0]["cov_model"],
        )
    out_prop = _empty_clone(prop)
    with _hpgl_error_guard("indicator_kriging"):
        _hpgl_so.hpgl_indicator_kriging(
            C.byref(_create_hpgl_ind_masked_array(prop, grid)),
            C.byref(_create_hpgl_ind_masked_array(out_prop, grid)),
            __create_hpgl_ik_params(data, len(data), False, marginal_probs),
            len(data),
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
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses and max_neighbours
    ParameterValidator.validate_radius(radiuses, "radiuses")
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

    # Validate cokriging-specific parameters
    ParameterValidator.validate_correlation_coef(correlation_coef)
    ParameterValidator.validate_variance(secondary_variance, "secondary_variance")

    # Validate secondary_data
    if not isinstance(secondary_data, ContProperty):
        raise TypeError(
            f"simple_cokriging_markI: secondary_data must be a ContProperty, "
            f"got {type(secondary_data).__name__}"
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

    out_prop = _empty_clone(prop)

    with _hpgl_error_guard("simple_cokriging_markI"):
        _hpgl_so.hpgl_simple_cokriging_mark1(
            C.byref(_create_hpgl_cont_masked_array(prop, grid)),
            C.byref(_create_hpgl_cont_masked_array(secondary_data, grid)),
            C.byref(
                __checked_create(
                    __hpgl_cockriging_m1_params_t,
                    covariance_type=cov_model.type,
                    ranges=_c_array(C.c_double, 3, cov_model.ranges),
                    angles=_c_array(C.c_double, 3, cov_model.angles),
                    sill=cov_model.sill,
                    nugget=cov_model.nugget,
                    radiuses=_c_array(C.c_int, 3, radiuses),
                    max_neighbours=max_neighbours,
                    primary_mean=primary_mean,
                    secondary_mean=secondary_mean,
                    secondary_variance=secondary_variance,
                    correlation_coef=correlation_coef,
                )
            ),
            C.byref(_create_hpgl_cont_masked_array(out_prop, grid)),
        )
    return out_prop


def simple_cokriging_markII(
    grid, primary_data, secondary_data, correlation_coef, radiuses, max_neighbours
):
    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses and max_neighbours
    ParameterValidator.validate_radius(radiuses, "radiuses")
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

    # Validate both covariance models
    for _label, d in [("primary", primary_data), ("secondary", secondary_data)]:
        cm = d["cov_model"]
        ParameterValidator.validate_covariance_parameters(cm.sill, cm.nugget, cm.ranges, cm.angles)

    out_prop = _empty_clone(primary_data["data"])

    pcp = primary_data["cov_model"]
    scp = secondary_data["cov_model"]

    with _hpgl_error_guard("simple_cokriging_markII"):
        _hpgl_so.hpgl_simple_cokriging_mark2(
            C.byref(_create_hpgl_cont_masked_array(primary_data["data"], grid)),
            C.byref(_create_hpgl_cont_masked_array(secondary_data["data"], grid)),
            C.byref(
                __checked_create(
                    __hpgl_cockriging_m2_params_t,
                    primary_cov_params=__checked_create(
                        __hpgl_cov_params_t,
                        covariance_type=pcp.type,
                        ranges=_c_array(C.c_double, 3, pcp.ranges),
                        angles=_c_array(C.c_double, 3, pcp.angles),
                        sill=pcp.sill,
                        nugget=pcp.nugget,
                    ),
                    secondary_cov_params=__checked_create(
                        __hpgl_cov_params_t,
                        covariance_type=scp.type,
                        ranges=_c_array(C.c_double, 3, scp.ranges),
                        angles=_c_array(C.c_double, 3, scp.angles),
                        sill=scp.sill,
                        nugget=scp.nugget,
                    ),
                    radiuses=_c_array(C.c_int, 3, radiuses),
                    max_neighbours=max_neighbours,
                    primary_mean=primary_data["mean"],
                    secondary_mean=secondary_data["mean"],
                    correlation_coef=correlation_coef,
                )
            ),
            C.byref(_create_hpgl_cont_masked_array(out_prop, grid)),
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

    covp = C.byref(
        __checked_create(
            __hpgl_cov_params_t,
            covariance_type=cov_type,
            ranges=_c_array(C.c_double, 3, ranges),
            angles=_c_array(C.c_double, 3, angles),
            sill=sill,
            nugget=nugget,
        )
    )

    weights = numpy.array([0] * len(n_x), dtype="float32")

    with _hpgl_error_guard("simple_kriging_weights"):
        rc = _hpgl_so.hpgl_simple_kriging_weights(
            _c_array(C.c_float, 3, center_point),
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
                + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace")
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
    if not isinstance(prop_dict, dict):
        raise TypeError(
            f"get_gslib_property: prop_dict must be a dict, got {type(prop_dict).__name__}"
        )
    prop = prop_dict[prop_name]
    informed_array = numpy.zeros(prop.shape, dtype=numpy.uint8)
    # Use exact equality for undefined_value comparison to match
    # the C++ fast reader and _load_prop_cont_slow parser behavior.
    # NaN sentinels are handled separately since NaN != NaN.
    if numpy.isnan(undefined_value):
        uninformed = numpy.isnan(prop)
    else:
        uninformed = prop == undefined_value
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
    """
    global _h, _output_handler_param
    if handler is not None and not callable(handler):
        raise TypeError(
            f"set_output_handler: handler must be callable or None, got {type(handler).__name__}"
        )
    if handler is None:
        # Cast None to the expected type for ctypes compatibility
        _hpgl_so.hpgl_set_output_handler(C.cast(None, hpgl_output_handler), None)  # type: ignore[arg-type]
        _h = None
        _output_handler_param = None
    else:
        _h = hpgl_output_handler(handler)
        _output_handler_param = param
        _hpgl_so.hpgl_set_output_handler(_h, param)


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
    """
    global _progress_handler, _progress_handler_param
    if handler is not None and not callable(handler):
        raise TypeError(
            f"set_progress_handler: handler must be callable or None, got {type(handler).__name__}"
        )
    if handler is None:
        # Cast None to the expected type for ctypes compatibility
        _hpgl_so.hpgl_set_progress_handler(C.cast(None, hpgl_progress_handler), None)  # type: ignore[arg-type]
        _progress_handler = None
        _progress_handler_param = None
    else:
        _progress_handler = hpgl_progress_handler(handler)
        _progress_handler_param = param
        _hpgl_so.hpgl_set_progress_handler(_progress_handler, param)


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
