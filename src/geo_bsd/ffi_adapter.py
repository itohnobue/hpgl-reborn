# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
r"""Centralized FFI adapter for all HPGL C API calls.

This module owns:

1. **Library loading** — imports the loaded ``_hpgl_so`` from :mod:`hpgl_wrap`.
2. **argtypes/restypes** — declared in :mod:`hpgl_wrap`, consumed here.
3. **GC reference management** — numpy array pinning via ``_array_refs`` on
   every ctypes struct that holds a data pointer.
4. **Error-state checking** — snapshot-and-compare pattern (``_hpgl_error_guard``)
   wrapping every C API call.
5. **Clean Python-level wrappers** — one function per C API call, accepting
   ctypes structs (not raw pointers), applying ``byref`` internally, and
   returning status codes where applicable.

All callers in :mod:`geo`, :mod:`sgs`, and :mod:`sis` route their C
calls through this module — no raw ``ctypes`` usage outside this file.
"""

from __future__ import annotations

import contextlib
import ctypes as C
import threading

import numpy

# Re-export all low-level ctypes definitions from hpgl_wrap.
# Callers import struct types and the library handle from here.
from .hpgl_wrap import (  # noqa: F401 — re-exports for callers
    # Struct types
    _HPGL_CONT_MASKED_ARRAY,
    _HPGL_FLOAT_ARRAY,
    _HPGL_IK_PARAMS,
    _HPGL_IND_MASKED_ARRAY,
    _HPGL_KRIGING_KIND,
    _HPGL_MEDIAN_IK_PARAMS,
    _HPGL_OK_PARAMS,
    _HPGL_SGS_PARAMS,
    _HPGL_SHAPE,
    _HPGL_SK_PARAMS,
    _HPGL_UBYTE_ARRAY,
    __hpgl_cockriging_m1_params_t,
    __hpgl_cockriging_m2_params_t,
    __hpgl_cov_params_t,
    # Library handle
    _hpgl_so,
    # Stats helper
    get_kriging_stats,
    hpgl_non_parametric_cdf_t,
    # Callback types
    hpgl_output_handler,
    hpgl_progress_handler,
)

# Convenience re-exports with shorter names
HPGL_CONT_MASKED_ARRAY = _HPGL_CONT_MASKED_ARRAY
HPGL_FLOAT_ARRAY = _HPGL_FLOAT_ARRAY
HPGL_IK_PARAMS = _HPGL_IK_PARAMS
HPGL_IND_MASKED_ARRAY = _HPGL_IND_MASKED_ARRAY
HPGL_MEDIAN_IK_PARAMS = _HPGL_MEDIAN_IK_PARAMS
HPGL_OK_PARAMS = _HPGL_OK_PARAMS
HPGL_SGS_PARAMS = _HPGL_SGS_PARAMS
HPGL_SHAPE = _HPGL_SHAPE
HPGL_SK_PARAMS = _HPGL_SK_PARAMS
HPGL_UBYTE_ARRAY = _HPGL_UBYTE_ARRAY
HPGL_COKRIGING_M1_PARAMS = __hpgl_cockriging_m1_params_t
HPGL_COKRIGING_M2_PARAMS = __hpgl_cockriging_m2_params_t
HPGL_COV_PARAMS = __hpgl_cov_params_t
HPGL_NONPARAM_CDF = hpgl_non_parametric_cdf_t

# Expose common ctypes types for use with _c_array() by callers
c_double = C.c_double
c_float = C.c_float
c_int = C.c_int
c_ubyte = C.c_ubyte

# ============================================================================
# Error-checking infrastructure
# ============================================================================

# Thread-local state to prevent stale C++ error propagation between tests.
# The HPGL C++ DLL stores the last exception message globally and does not
# clear it after it's read. Without tracking, an error from one call would
# cause all subsequent error checks to falsely report failure.
#
# Using thread-local storage so that concurrent HPGL FFI calls in different
# threads each track their own pre/post error snapshot independently.
_error_local = threading.local()
_error_snapshot_lock = threading.Lock()

# Serializes C++ HPGL calls for operations that require cross-thread
# exclusion. The C++ error state is thread_local, so error checking
# does not need serialization. This lock is used for handler mutation
# (set_output_handler, set_progress_handler) and thread count changes
# (set_thread_num) where concurrent access could produce inconsistent
# CFUNCTYPE/global state.
_hpgl_call_lock = threading.Lock()


def _snapshot_hpgl_error():
    """Take a snapshot of the current HPGL C++ error state.

    Call this BEFORE invoking any C++ function that might succeed.
    ``_check_hpgl_error`` will compare the post-call error against this
    snapshot to detect only NEW errors, avoiding stale error propagation.

    Stores the snapshot in thread-local storage under
    ``_error_snapshot_lock``.
    """
    with _error_snapshot_lock:
        _error_local._hpgl_error_snapshot = _hpgl_so.hpgl_get_last_exception_message()


def _check_hpgl_error(context: str = "") -> None:
    """Check for NEW HPGL C++ errors after a computation call.

    Compares current error state against a pre-call snapshot (set via
    ``_snapshot_hpgl_error``). Raises ``RuntimeError`` ONLY if the error
    message has changed since the snapshot, indicating a new error from
    the current operation rather than a stale error from a previous call.

    Uses thread-local storage so concurrent HPGL operations in different
    threads each track their own pre/post error state independently.

    Args:
        context: Description of the operation being checked
            (e.g. "ordinary_kriging").

    Raises:
        RuntimeError: If the C++ computation produced a new error.
    """
    with _error_snapshot_lock:
        err = _hpgl_so.hpgl_get_last_exception_message()
        if err is not None and len(err) > 0:
            snapshot = getattr(_error_local, "_hpgl_error_snapshot", None)
            if err == snapshot:
                # Error unchanged from pre-call snapshot — C++ call did
                # not produce a new error. Always suppress stale errors.
                return
            # Genuine new error (different from pre-call snapshot).
            # Update the snapshot BEFORE raising so the thread-local
            # state reflects the consumed error, preventing double-raises
            # on re-entry within the same guard window.
            _error_local._hpgl_error_snapshot = err
            err_str = err.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{context} failed: {err_str}"
                if context
                else f"HPGL error: {err_str}"
            )


@contextlib.contextmanager
def error_guard(context: str = ""):
    """Guard a C++ HPGL call with error checking to detect new C++ errors.

    Usage::

        with error_guard("ordinary_kriging"):
            _hpgl_so.hpgl_ordinary_kriging(...)

    Snapshots the C++ error state, runs the C++ call (yield), and checks
    for new errors. The C++ error state is ``thread_local``, so no lock
    is needed for error isolation between threads.
    """
    _snapshot_hpgl_error()
    yield
    _check_hpgl_error(context)


# ============================================================================
# Numpy array validation
# ============================================================================


def checkFWA(a: numpy.ndarray) -> None:
    """Check Fortran-order, Writable, and Aligned flags on a numpy array.

    Raises:
        RuntimeError: If any of the F/W/A flags are not set.
    """
    if not (a.flags["F"] and a.flags["W"] and a.flags["A"]):
        raise RuntimeError(
            f"Array checkFWA failed: F={a.flags['F']}, W={a.flags['W']}, A={a.flags['A']}"
        )


# ============================================================================
# ctypes struct construction helpers
# ============================================================================


def _c_array(ar_type, size, values):
    """Create a fixed-size ctypes array with GC pinning.

    Stores ``tuple(values)`` as ``_array_refs`` to prevent garbage
    collection of input values while C code holds pointers to the
    underlying data.
    """
    if len(values) != size:
        raise RuntimeError(
            f"{len(values)} values specified for array of {size} elements"
        )
    result = (ar_type * size)(*values)
    # Preserve references to input values to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = tuple(values)
    return result


def _create_hpgl_shape(shape, strides=None):
    """Create an ``_HPGL_SHAPE`` ctypes struct from shape and strides.

    Normalizes 1D/2D shapes to 3D and computes C-order (row-major)
    strides matching the C++ indexing convention ``z * x * y + y * x + x``.
    """
    # Normalize shape to 3D tuple
    if len(shape) == 1:
        shape = (shape[0], 1, 1)
    elif len(shape) == 2:
        shape = (shape[0], shape[1], 1)

    if strides is None:
        # C-order strides (row-major) to match C++ indexing
        return _HPGL_SHAPE(
            m_data=_c_array(C.c_int, 3, shape),
            m_strides=_c_array(C.c_int, 3, (1, shape[0], shape[0] * shape[1])),
        )
    else:
        return _HPGL_SHAPE(
            m_data=_c_array(C.c_int, 3, shape),
            m_strides=_c_array(C.c_int, 3, strides),
        )


def __get_strides(prop):
    """Compute element strides from a numpy array's strides in bytes."""
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


def checked_create(T, **kargs):
    """Create a ctypes struct with field-completeness validation.

    Verifies that every field in the struct's ``_fields_`` is provided
    as a keyword argument. Raises ``CriticalValidationError`` if any
    field is missing.

    .. note::

       This helper does NOT pin numpy array references via
       ``_array_refs``. It is intended for parameter structs whose
       fields are scalar types only — no ``ctypes.data_as()``
       pointers. See ``create_cont_masked_array`` and friends for
       array-pinning constructors.
    """
    from .validation import CriticalValidationError

    fields = []
    for f, _ in T._fields_:
        fields.append(f)
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
    if fields:
        raise CriticalValidationError(
            f"No values for parameters: {fields}", "ctypes_struct"
        )
    return T(**kargs)


def create_cont_masked_array(prop, grid):
    """Create an ``_HPGL_CONT_MASKED_ARRAY`` from a ``ContProperty``.

    Extracts ``data_as()`` pointers from the property's numpy arrays
    and pins them via ``_array_refs`` to prevent garbage collection
    while C code holds the pointers.

    Args:
        prop: A ``ContProperty`` with ``.data`` (float32) and
            ``.mask`` (uint8) arrays.
        grid: A ``SugarboxGrid`` or ``None``. If ``None``, shape
            and strides are derived from ``prop.data`` directly.

    Returns:
        ``_HPGL_CONT_MASKED_ARRAY`` with pinned array references.
    """
    if grid is None:
        sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
    else:
        # Use actual NumPy strides if array is 3D, otherwise compute
        # strides from grid.
        if prop.data.ndim == 3:
            # Validate that the 3D data shape matches grid dimensions.
            if (
                prop.data.shape[0] != grid.x
                or prop.data.shape[1] != grid.y
                or prop.data.shape[2] != grid.z
            ):
                raise RuntimeError(
                    f"3D data shape {prop.data.shape} does not match "
                    f"grid dimensions ({grid.x}, {grid.y}, {grid.z})"
                )
            sh = _create_hpgl_shape(
                (grid.x, grid.y, grid.z), __get_strides(prop.data)
            )
        else:
            # For 1D arrays, compute expected strides from grid
            sh = _create_hpgl_shape((grid.x, grid.y, grid.z))
        if grid.x * grid.y * grid.z != prop.data.size:
            raise RuntimeError(
                f"Invalid data size. Size of data = {prop.data.size}. "
                f"Size of grid = {grid.x * grid.y * grid.z}"
            )

    # Security: Keep references to arrays to prevent use-after-free
    result = _HPGL_CONT_MASKED_ARRAY(
        data=prop.data.ctypes.data_as(C.POINTER(C.c_float)),
        mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
        shape=sh,
    )

    # Store array references on the result object to prevent garbage collection
    result._array_refs = (prop.data, prop.mask)

    return result


def create_ind_masked_array(prop, grid):
    """Create an ``_HPGL_IND_MASKED_ARRAY`` from an ``IndProperty``.

    Extracts ``data_as()`` pointers from the property's numpy arrays
    and pins them via ``_array_refs`` to prevent garbage collection.

    Args:
        prop: An ``IndProperty`` with ``.data`` (uint8), ``.mask``
            (uint8), and ``.indicator_count``.
        grid: A ``SugarboxGrid`` or ``None``.

    Returns:
        ``_HPGL_IND_MASKED_ARRAY`` with pinned array references.
    """
    if grid is None:
        sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
        assert prop.data.strides == prop.mask.strides
    else:
        # Use actual NumPy strides if array is 3D, otherwise compute
        # strides from grid.
        if prop.data.ndim == 3:
            if (
                prop.data.shape[0] != grid.x
                or prop.data.shape[1] != grid.y
                or prop.data.shape[2] != grid.z
            ):
                raise RuntimeError(
                    f"3D data shape {prop.data.shape} does not match "
                    f"grid dimensions ({grid.x}, {grid.y}, {grid.z})"
                )
            sh = _create_hpgl_shape(
                (grid.x, grid.y, grid.z), __get_strides(prop.data)
            )
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

    # Store array references to prevent garbage collection
    result._array_refs = (prop.data, prop.mask)

    return result


def create_ubyte_array(array, grid):
    """Create an ``_HPGL_UBYTE_ARRAY`` from a uint8 numpy array."""
    checkFWA(array)
    if grid is None:
        sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
    else:
        sh = _create_hpgl_shape((grid.x, grid.y, grid.z))
        if grid.x * grid.y * grid.z != array.size:
            raise RuntimeError(
                f"Invalid data size. Size of data = {array.size}. "
                f"Size of grid = {grid.x * grid.y * grid.z}"
            )

    result = _HPGL_UBYTE_ARRAY(
        data=array.ctypes.data_as(C.POINTER(C.c_ubyte)), shape=sh
    )
    result._array_ref = array
    return result


def create_float_array(array, grid):
    """Create an ``_HPGL_FLOAT_ARRAY`` from a float32 numpy array."""
    checkFWA(array)
    if grid is None:
        sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
    else:
        sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

    result = _HPGL_FLOAT_ARRAY(
        data=array.ctypes.data_as(C.POINTER(C.c_float)), shape=sh
    )
    result._array_ref = array
    return result


def create_nonparam_cdf(cdf_data):
    """Create an ``hpgl_non_parametric_cdf_t`` from ``CdfData``.

    Extracts ``data_as()`` pointers from the CDF value/probability
    arrays and pins them via ``_array_refs``.
    """
    from .cdf import CdfData

    cd2 = cdf_data
    if not isinstance(cdf_data, CdfData):
        raise TypeError(
            f"create_nonparam_cdf: expected CdfData, got {type(cdf_data).__name__}"
        )
    result = checked_create(
        hpgl_non_parametric_cdf_t,
        values=cd2.values.ctypes.data_as(C.POINTER(C.c_float)),
        probs=cd2.probs.ctypes.data_as(C.POINTER(C.c_float)),
        size=cd2.values.size,
    )
    # Preserve references to numpy arrays to prevent garbage collection
    result._array_refs = (cd2.values, cd2.probs)
    return result


def create_ik_params(data, indicator_count, is_lvm, marginal_probs):
    """Create a ctypes array of ``_HPGL_IK_PARAMS`` structs.

    Used by both ``indicator_kriging`` (geo.py) and ``sis_simulation``
    (sis.py) — this is the consolidated implementation replacing the
    former duplicate in both modules.
    """
    ikps = []
    assert len(data) == indicator_count
    for i in range(indicator_count):
        ikd = data[i]
        ikp = checked_create(
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


# ============================================================================
# C API wrappers — void-return kriging / simulation functions
# ============================================================================


def call_ordinary_kriging(inp_struct, params_struct, outp_struct):
    """Call ``hpgl_ordinary_kriging`` with error guard.

    Accepts ctypes structs (not pointers) — ``byref`` is applied
    internally.
    """
    with error_guard("ordinary_kriging"):
        _hpgl_so.hpgl_ordinary_kriging(
            C.byref(inp_struct), C.byref(params_struct), C.byref(outp_struct)
        )


def call_simple_kriging(data, mask, shape, params, out_data, out_mask, out_shape):
    """Call ``hpgl_simple_kriging`` with error guard.

    ``data``, ``mask``, ``out_data``, ``out_mask`` are numpy arrays
    (validated by ``ndpointer`` argtypes). ``shape``, ``params``,
    ``out_shape`` are ctypes structs (``byref`` applied internally).
    """
    with error_guard("simple_kriging"):
        _hpgl_so.hpgl_simple_kriging(
            data,
            mask,
            C.byref(shape),
            C.byref(params),
            out_data,
            out_mask,
            C.byref(out_shape),
        )


def call_lvm_kriging(
    data,
    mask,
    shape,
    mean_data,
    mean_shape,
    params,
    out_data,
    out_mask,
    out_shape,
):
    """Call ``hpgl_lvm_kriging`` with error guard.

    ``data``, ``mask``, ``mean_data``, ``out_data``, ``out_mask`` are
    numpy arrays. ``shape``, ``mean_shape``, ``params``, ``out_shape``
    are ctypes structs.
    """
    with error_guard("lvm_kriging"):
        _hpgl_so.hpgl_lvm_kriging(
            data,
            mask,
            C.byref(shape),
            mean_data,
            C.byref(mean_shape),
            C.byref(params),
            out_data,
            out_mask,
            C.byref(out_shape),
        )


def call_median_ik(inp_struct, params_struct, outp_struct):
    """Call ``hpgl_median_ik`` with error guard."""
    with error_guard("median_ik"):
        _hpgl_so.hpgl_median_ik(
            C.byref(inp_struct), C.byref(params_struct), C.byref(outp_struct)
        )


def call_indicator_kriging(inp_struct, outp_struct, params_array_struct, count):
    """Call ``hpgl_indicator_kriging`` with error guard.

    ``params_array_struct`` is a ctypes array of ``_HPGL_IK_PARAMS``
    (created by ``create_ik_params``). It is NOT ``byref``'d — the C
    function receives the array by value.
    """
    with error_guard("indicator_kriging"):
        _hpgl_so.hpgl_indicator_kriging(
            C.byref(inp_struct),
            C.byref(outp_struct),
            params_array_struct,
            count,
        )


def call_sgs_simulation(cont_marr, params, cdf_struct_or_none, scalar_mean_or_none, mask_struct_or_none):
    """Call ``hpgl_sgs_simulation`` with error guard.

    Args:
        cont_marr: ``_HPGL_CONT_MASKED_ARRAY`` struct.
        params: ``_HPGL_SGS_PARAMS`` struct.
        cdf_struct_or_none: ``hpgl_non_parametric_cdf_t`` struct or ``None``.
        scalar_mean_or_none: ``float`` or ``None`` (converted to ``c_double``).
        mask_struct_or_none: ``_HPGL_UBYTE_ARRAY`` struct or ``None``.
    """
    hpgl_cdf = C.byref(cdf_struct_or_none) if cdf_struct_or_none is not None else None
    _c_mean = C.c_double(scalar_mean_or_none) if scalar_mean_or_none is not None else None
    hpgl_mask = C.byref(mask_struct_or_none) if mask_struct_or_none is not None else None
    with error_guard("sgs_simulation"):
        _hpgl_so.hpgl_sgs_simulation(
            C.byref(cont_marr),
            C.byref(params),
            hpgl_cdf,
            C.byref(_c_mean) if _c_mean is not None else None,
            hpgl_mask,
        )


def call_sgs_lvm_simulation(cont_marr, params, cdf_struct_or_none, float_arr_struct, mask_struct_or_none):
    """Call ``hpgl_sgs_lvm_simulation`` with error guard."""
    hpgl_cdf = C.byref(cdf_struct_or_none) if cdf_struct_or_none is not None else None
    hpgl_mask = C.byref(mask_struct_or_none) if mask_struct_or_none is not None else None
    with error_guard("sgs_lvm_simulation"):
        _hpgl_so.hpgl_sgs_lvm_simulation(
            C.byref(cont_marr),
            C.byref(params),
            hpgl_cdf,
            C.byref(float_arr_struct),
            hpgl_mask,
        )


def call_sis_simulation(ind_marr, params, count, seed, mask_struct_or_none):
    """Call ``hpgl_sis_simulation`` with error guard."""
    hpgl_mask = C.byref(mask_struct_or_none) if mask_struct_or_none is not None else None
    with error_guard("sis_simulation"):
        _hpgl_so.hpgl_sis_simulation(
            ind_marr, params, count, seed, hpgl_mask
        )


def call_sis_simulation_lvm(
    ind_marr, params, float_arrs_struct, count, seed, mask_struct_or_none, use_correlogram
):
    """Call ``hpgl_sis_simulation_lvm`` with error guard."""
    hpgl_mask = C.byref(mask_struct_or_none) if mask_struct_or_none is not None else None
    with error_guard("sis_simulation_lvm"):
        _hpgl_so.hpgl_sis_simulation_lvm(
            ind_marr,
            params,
            float_arrs_struct,
            count,
            seed,
            hpgl_mask,
            use_correlogram,
        )


def call_simple_cokriging_mark1(
    primary_struct, secondary_struct, params_struct, output_struct
):
    """Call ``hpgl_simple_cokriging_mark1`` with error guard."""
    with error_guard("simple_cokriging_markI"):
        _hpgl_so.hpgl_simple_cokriging_mark1(
            C.byref(primary_struct),
            C.byref(secondary_struct),
            C.byref(params_struct),
            C.byref(output_struct),
        )


def call_simple_cokriging_mark2(
    primary_struct, secondary_struct, params_struct, output_struct
):
    """Call ``hpgl_simple_cokriging_mark2`` with error guard."""
    with error_guard("simple_cokriging_markII"):
        _hpgl_so.hpgl_simple_cokriging_mark2(
            C.byref(primary_struct),
            C.byref(secondary_struct),
            C.byref(params_struct),
            C.byref(output_struct),
        )


# ============================================================================
# C API wrappers — int-return I/O and utility functions
# ============================================================================


def call_simple_kriging_weights(
    center_point, n_x, n_y, n_z, count, cov_params_struct, weights
):
    """Call ``hpgl_simple_kriging_weights`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("simple_kriging_weights"):
        return _hpgl_so.hpgl_simple_kriging_weights(
            center_point,
            n_x,
            n_y,
            n_z,
            count,
            C.byref(cov_params_struct),
            weights,
        )


def call_read_inc_file_float(filename_bytes, undefined_value, total_elements, data, mask):
    """Call ``hpgl_read_inc_file_float`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("read_inc_file_float"):
        return _hpgl_so.hpgl_read_inc_file_float(
            filename_bytes, undefined_value, total_elements, data, mask
        )


def call_read_inc_file_byte(
    filename_bytes, undefined_value, total_elements, data, mask, indicator_values, count
):
    """Call ``hpgl_read_inc_file_byte`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("read_inc_file_byte"):
        return _hpgl_so.hpgl_read_inc_file_byte(
            filename_bytes, undefined_value, total_elements, data, mask, indicator_values, count
        )


def call_write_inc_file_float(marr_struct, filename_bytes, undefined_value, prop_name_bytes):
    """Call ``hpgl_write_inc_file_float`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("write_property"):
        return _hpgl_so.hpgl_write_inc_file_float(
            filename_bytes, C.byref(marr_struct), undefined_value, prop_name_bytes
        )


def call_write_inc_file_byte(
    marr_struct, filename_bytes, undefined_value, prop_name_bytes,
    indicator_values_array, indicator_count,
):
    """Call ``hpgl_write_inc_file_byte`` with error guard.

    Args:
        marr_struct: ``_HPGL_IND_MASKED_ARRAY`` struct.
        filename_bytes: Encoded filename.
        undefined_value: Undefined value sentinel (int).
        prop_name_bytes: Encoded property name.
        indicator_values_array: numpy uint8 array of indicator values
            (converted to ctypes pointer internally).
        indicator_count: Number of indicator values.

    Returns the C status code (0 = success).
    """
    ind_ptr = indicator_values_array.ctypes.data_as(C.POINTER(C.c_ubyte))
    with error_guard("write_property"):
        return _hpgl_so.hpgl_write_inc_file_byte(
            filename_bytes,
            C.byref(marr_struct),
            undefined_value,
            prop_name_bytes,
            ind_ptr,
            indicator_count,
        )


def call_write_gslib_cont_property(marr_struct, filename_bytes, prop_name_bytes, undefined_value):
    """Call ``hpgl_write_gslib_cont_property`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("write_gslib_property"):
        return _hpgl_so.hpgl_write_gslib_cont_property(
            marr_struct, filename_bytes, prop_name_bytes, undefined_value
        )


def call_write_gslib_byte_property(
    marr_struct, filename_bytes, prop_name_bytes, undefined_value,
    indicator_values_ptr, indicator_count,
):
    """Call ``hpgl_write_gslib_byte_property`` with error guard.

    Returns the C status code (0 = success).
    """
    with error_guard("write_gslib_property"):
        return _hpgl_so.hpgl_write_gslib_byte_property(
            marr_struct,
            filename_bytes,
            prop_name_bytes,
            undefined_value,
            indicator_values_ptr,
            indicator_count,
        )


def call_set_thread_num(num):
    """Call ``hpgl_set_thread_num`` with error guard.

    Returns the C status code.

    .. note::

       This wrapper only handles the C call and error checking.
       Callers that need cross-thread exclusion must acquire
       ``_hpgl_call_lock`` separately (see ``geo.set_thread_num``).
    """
    with error_guard("set_thread_num"):
        return _hpgl_so.hpgl_set_thread_num(num)


def call_get_thread_num():
    """Call ``hpgl_get_thread_num`` (no error guard needed).

    Returns the OpenMP max thread count as an ``int``.
    """
    return _hpgl_so.hpgl_get_thread_num()


def call_set_output_handler(handler_or_none, param):
    """Call ``hpgl_set_output_handler``.

    Does NOT use ``error_guard`` — this is a simple pointer-set
    operation that cannot fail. Callers must manage CFUNCTYPE lifetime
    (see ``geo.set_output_handler``).

    Args:
        handler_or_none: A callable (converted to CFUNCTYPE by the caller),
            or ``None`` to clear the handler. When ``None``, the internal
            ``C.cast(None, ...)`` is handled automatically.
        param: The user parameter or ``None``.
    """
    if handler_or_none is None:
        _hpgl_so.hpgl_set_output_handler(C.cast(None, hpgl_output_handler), None)  # type: ignore[arg-type]
    else:
        _hpgl_so.hpgl_set_output_handler(handler_or_none, param)


def call_set_progress_handler(handler_or_none, param):
    """Call ``hpgl_set_progress_handler``.

    Does NOT use ``error_guard`` — same reasoning as
    ``call_set_output_handler``.
    """
    if handler_or_none is None:
        _hpgl_so.hpgl_set_progress_handler(C.cast(None, hpgl_progress_handler), None)  # type: ignore[arg-type]
    else:
        _hpgl_so.hpgl_set_progress_handler(handler_or_none, param)


def call_get_last_exception_message():
    """Call ``hpgl_get_last_exception_message``.

    Returns the current C++ error message as ``bytes`` or ``None``.
    Used by the snapshot/check infrastructure — normally callers use
    ``error_guard`` instead of calling this directly.
    """
    return _hpgl_so.hpgl_get_last_exception_message()
