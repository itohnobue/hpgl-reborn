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
#
# PR-02: RLock (reentrant) — C++ invokes user output/progress handlers on
# the calling thread DURING kriging/simulation FFI calls that hold this
# lock. A handler that calls set_output_handler/set_progress_handler (or
# re-enters a kriging call) would self-deadlock on a plain Lock; RLock
# allows the same thread to re-acquire while preserving cross-thread
# exclusion.
_hpgl_call_lock = threading.RLock()

# Note: cross-call error detection (distinguishing a re-appearing error
# from a persistent stale one across multiple guard invocations) uses a
# call sequence counter: _snapshot_hpgl_error increments the counter and
# _check_hpgl_error compares both the error message AND the sequence
# number. This prevents identical error messages from consecutive C calls
# from being suppressed, while still suppressing stale errors within the
# same call window.
#
# KNOWN LIMITATION — NO C++ CLEAR-ERROR API: The HPGL C++ shared library
# stores the last exception message globally (thread_local) and does NOT
# expose a C ABI ``hpgl_clear_last_exception_message()`` function. When the
# pre-call snapshot accidentally matches a genuinely new error (same
# message, C++ state not cleared), the error is suppressed. The sequence
# counter mitigates this for consecutive calls but cannot help when a
# fresh identical error arises after intervening calls that did NOT
# change the C++ error state. A C++ clear-error API would be the correct
# fix; without it, this is a documented limitation with an approximate
# workaround.
#
# F-07: _check_hpgl_error CONSUMES the C++ error after each guard (raise or
# suppress) by resetting it to empty via the exported C++ setter
# (hpgl::set_last_exception_message, Itanium-mangled on macOS/Linux). This
# makes the "stale" state self-cleaning: a second distinct call that FAILS
# with the identical message raises (the pre-call snapshot is clean), while
# a successful call never sees a stale error. Platforms where the setter is
# not dynamically exported (e.g. Windows MSVC) fall back to the
# snapshot/seq suppression (pre-fix behavior).


def _snapshot_hpgl_error():
    """Take a snapshot of the current HPGL C++ error state.

    Call this BEFORE invoking any C++ function that might succeed.
    ``_check_hpgl_error`` will compare the post-call error against this
    snapshot to detect only NEW errors, avoiding stale error propagation.

    Stores the snapshot in thread-local storage under
    ``_error_snapshot_lock``. Increments a per-thread call sequence
    counter so that ``_check_hpgl_error`` can distinguish between a
    stale error (same call window) and a genuinely new error from a
    different C call that happens to produce the same message.
    """
    with _error_snapshot_lock:
        # Bump the call sequence counter to identify the current C call window
        seq = getattr(_error_local, "_hpgl_call_seq", 0) + 1
        _error_local._hpgl_call_seq = seq
        _error_local._hpgl_error_snapshot = _hpgl_so.hpgl_get_last_exception_message()
        _error_local._hpgl_error_snapshot_seq = seq


def _check_hpgl_error(context: str = "") -> None:
    """Check for NEW HPGL C++ errors after a computation call.

    Compares current error state against a pre-call snapshot (set via
    ``_snapshot_hpgl_error``). Raises ``RuntimeError`` ONLY if the error
    message has changed since the snapshot, indicating a new error from
    the current operation rather than a stale error from a previous call.

    Uses a call sequence counter to distinguish between a stale error
    within the same call window (suppress) and a genuinely new error
    from a different C call that happens to produce an identical
    message (raise). The counter prevents identical-message collision
    across consecutive ``error_guard`` invocations.

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
            snapshot_seq = getattr(_error_local, "_hpgl_error_snapshot_seq", 0)
            current_seq = getattr(_error_local, "_hpgl_call_seq", 0)
            if err == snapshot and snapshot_seq == current_seq:
                # Error unchanged from pre-call snapshot AND this is the
                # same call window — C++ call did not produce a new error.
                # Suppress stale error, then consume it so the next call
                # starts from a clean C++ error state (F-07).
                _clear_hpgl_error()
                return
            # Genuine new error (different from pre-call snapshot, or
            # identical message from a different C++ call).
            # Update the snapshot BEFORE raising so the thread-local
            # state reflects the consumed error, preventing double-raises
            # on re-entry within the same guard window.
            _error_local._hpgl_error_snapshot = err
            _error_local._hpgl_error_snapshot_seq = current_seq
            # Consume the C++ error before raising so a later call that
            # succeeds does not see a stale error, while a later call that
            # FAILS with the identical message raises (F-07).
            _clear_hpgl_error()
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


def _clear_hpgl_error() -> None:
    """Reset the C++ thread-local error after the guard consumes it (F-07).

    The C ABI exposes only ``hpgl_get_last_exception_message`` (read); the
    C++ ``hpgl::set_last_exception_message(const char*)`` symbol IS exported
    with Itanium C++ mangling on macOS/Linux and is callable through ctypes.
    Consuming the error lets a later call start from a clean state: a second
    distinct call that fails with the IDENTICAL message raises (previously
    silently suppressed), while a successful call does not see a stale error.

    Falls back to a no-op when the symbol is unavailable (e.g. Windows MSVC
    builds) — the snapshot/seq logic then still suppresses the stale error.
    """
    try:
        setter = _hpgl_so["_ZN4hpgl26set_last_exception_messageEPKc"]
        setter.argtypes = [C.c_char_p]
        setter.restype = None
        setter(b"")
    except (AttributeError, KeyError, TypeError):
        pass


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
# FFI output-buffer contract (got-20260802015741 — recurring class: the
# contiguity → size → writeability → full-init facets surface one per run).
#
# ONE shared helper enforces the complete contract on every FFI output
# buffer, so a future call site cannot skip a facet the way sibling call
# sites did across runs (F-06/F-08 contiguity → 2-M-15 size → III-39
# writeability → III-40 full-init). Route ALL output-buffer validations
# through require_output_buffer.
# ============================================================================

# Sentinel value the C++ kernels use to mark "cell not written" where a
# full-init is contractually required (e.g. top-tail cells of stack_layers).
_UNWRITTEN_CELL_SENTINEL = -500.0


def require_output_buffer(
    arr: numpy.ndarray,
    expected_size: int,
    context: str,
    dtype: numpy.dtype | None = None,
) -> None:
    """Enforce the complete FFI output-buffer contract on ``arr``.

    The contract is: C-contiguous or F-contiguous layout (contiguity), exact
    element count (size), writable memory (writeability), and — when
    ``require_full_init`` is True — all cells finite (full initialization).
    Contiguity comes first: a strided view would be read/written by the C
    kernel as if it were flat, producing garbage or OOB access. Size second:
    the C kernel computes ``min(num_lags, result_length)``-style truncation
    or writes past a short buffer. Writeability third: a read-only buffer is
    silently mutated (or SIGBUS). Every check raises ``ValueError`` with the
    caller's ``context`` in the message.

    Args:
        arr: The output numpy array the C kernel will write into.
        expected_size: The exact number of elements the kernel will write.
        context: Operation name for error messages.
        dtype: Optional required dtype. When None, the dtype is not checked
            (callers that enforce a dtype via ``ndpointer`` argtypes or
            explicit checks keep their existing behavior).

    Raises:
        ValueError: If any facet of the contract is violated.
    """
    if not isinstance(arr, numpy.ndarray):
        raise ValueError(
            f"{context}: output buffer must be a numpy.ndarray, "
            f"got {type(arr).__name__}"
        )
    if not (arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]):
        raise ValueError(
            f"{context}: output buffer must be contiguous (got a strided view "
            "— the C kernel reads/writes the buffer as flat memory)"
        )
    if arr.size != expected_size:
        raise ValueError(
            f"{context}: output buffer size {arr.size} does not match expected "
            f"{expected_size} (the C kernel computes with min(size, expected) "
            "or writes past a short buffer)"
        )
    if not arr.flags["W"]:
        raise ValueError(
            f"{context}: output buffer must be writable (read-only buffer would "
            "be silently mutated or SIGBUS)"
        )
    if dtype is not None and arr.dtype != dtype:
        raise ValueError(
            f"{context}: output buffer dtype {arr.dtype} does not match expected "
            f"{dtype}"
        )


def require_output_buffer_full_init(arr: numpy.ndarray, context: str) -> None:
    """Verify the C kernel fully initialized an output buffer.

    A C kernel that loops ``for i < nz`` over an output buffer that has more
    layers than ``nz`` leaves the top-tail cells holding their pre-call
    garbage (or a sentinel). The caller must have pre-initialized the buffer
    to ``_UNWRITTEN_CELL_SENTINEL``; this check asserts no cell still holds
    that sentinel after the call, proving the kernel wrote every cell it was
    contracted to (full-init facet of the FFI output-buffer contract).

    Args:
        arr: The output buffer after the C call.
        context: Operation name for error messages.

    Raises:
        ValueError: If any cell still holds the unwritten-cell sentinel.
    """
    if numpy.any(arr == _UNWRITTEN_CELL_SENTINEL):
        raise ValueError(
            f"{context}: output buffer not fully initialized — cells still hold "
            f"the pre-call sentinel ({_UNWRITTEN_CELL_SENTINEL}). The C kernel "
            "did not write every contracted cell."
        )


def prefill_output_buffer(arr: numpy.ndarray) -> None:
    """Pre-fill an output buffer with the unwritten-cell sentinel.

    Call BEFORE the C call when the kernel is contracted to write every cell
    (e.g. CStackLayers with ``nz`` layers into a deeper buffer): the sentinel
    makes unwritten cells detectable after the call via
    ``require_output_buffer_full_init``.

    Args:
        arr: The output buffer to pre-fill.
    """
    arr[:] = _UNWRITTEN_CELL_SENTINEL


# ============================================================================
# Mask semantics contract (got-20260803180153 — recurring class: mask
# semantics must be ONE definition everywhere). The library-wide contract is
# "non-zero = informed". C++ kernels gate on ``mask[node] == 1``
# (sequential_simulation.h:124, sequential_indicator_simulation.cpp:114), so
# any non-zero value must be normalized to 1 at the FFI boundary BEFORE the
# C++ kernel sees it — otherwise a mask value of 2 is counted as informed by
# Python (mask != 0) but silently skipped by C++ (mask == 1). Centralized
# here so every boundary uses the same normalization.
# ============================================================================


def normalize_mask_binary(mask: numpy.ndarray, context: str) -> numpy.ndarray | None:
    """Normalize a mask array to the binary (0/1) contract used by the C++.

    ``None`` input stays ``None`` (no mask). A non-binary mask (any value
    outside {0, 1}, e.g. 2) is converted to binary with ``mask != 0`` — the
    library-wide "non-zero = informed" definition — so the Python expected-
    cell count (mask != 0) and the C++ simulation gate (mask == 1) agree on
    the SAME set of cells after normalization.

    Returns a float64/uint8-compatible binary mask. The returned array is a
    new uint8 array (values 0/1); the caller's array is not mutated.

    Args:
        mask: The user-supplied mask array or ``None``.
        context: Operation name for error messages.

    Raises:
        TypeError: If ``mask`` is not a numpy array.
    """
    if mask is None:
        return None
    if not isinstance(mask, numpy.ndarray):
        raise TypeError(
            f"{context}: mask must be a numpy.ndarray, got {type(mask).__name__}"
        )
    # E-M7: ALWAYS return a fresh uint8 copy, per the documented contract
    # ("The returned array is a new uint8 array (values 0/1); the caller's
    # array is not mutated"). The previous early return handed the caller's
    # array back unchanged when it was already binary — a non-uint8 binary
    # array (bool, float64 0/1) then flowed downstream with its original
    # dtype/layout, violating the uint8 contract the C++ boundary assumes.
    # astype's default order="K" preserves the caller's memory layout, so
    # F-order masks (e.g. after geo.py _require_ind_data's numpy.require F
    # coercion) stay F-order and still pass checkFWA in create_ubyte_array.
    return (mask != 0).astype(numpy.uint8)


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
    if ar_type is C.c_ubyte:
        _validate_ubyte_values(values)
    result = (ar_type * size)(*values)
    # Preserve references to input values to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = tuple(values)
    return result


def _validate_ubyte_values(values):
    """Reject values that ctypes would silently corrupt in a ``c_ubyte`` array.

    ctypes array construction ``(c_ubyte * n)(*values)`` does NOT
    range-check Python ints: 300 wraps to 44 and 1.5 truncates to 1 with
    no error (F-44). On the GSLIB/INC write paths those corrupted values
    go straight to the on-disk file — silent data corruption. Validate
    integrality and the [0, 255] byte range before constructing.

    Raises:
        ValueError: If any value is not an integer in [0, 255].
    """
    for v in values:
        if isinstance(v, (int, numpy.integer)):
            ok = 0 <= int(v) <= 255
        elif isinstance(v, (float, numpy.floating)):
            ok = numpy.isfinite(v) and float(v).is_integer() and 0 <= v <= 255
        else:
            ok = False
        if not ok:
            raise ValueError(
                f"_c_array(c_ubyte): values must be integers in [0, 255], got {v!r}"
            )


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
    field is missing. P-04: unknown keyword arguments (typo'd field
    names) are also rejected with ``CriticalValidationError`` — a
    replacement typo previously degraded to the misleading
    missing-field error, and an extra typo was silently passed to the
    ctypes constructor.

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
    unknown = []
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
        else:
            unknown.append(k)
    if unknown:
        raise CriticalValidationError(
            f"Unexpected keyword arguments: {unknown}", "ctypes_struct"
        )
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
    # N2-26: enforce the FWA contract on the property arrays at the FFI
    # boundary (mirror create_ubyte_array:706 / create_float_array:738).
    # m_strides is inert in C++ (the kernels read both buffers flat by
    # node index), so a C-order or strided input would be silently
    # permuted with no exception. ContProperty's setters/ctor already
    # coerce Fortran order; this guard catches direct FFI misuse.
    checkFWA(prop.data)
    checkFWA(prop.mask)
    if grid is None:
        # F-24: validate that data and mask have the same element count.
        # C++ indexes both arrays with the same stride values, so a
        # data.size != mask.size mismatch reads/writes past the end of the
        # shorter array (heap OOB read). The stride comparison below only
        # catches layout mismatches, not length mismatches.
        if prop.data.size != prop.mask.size:
            raise ValueError(
                f"create_cont_masked_array: data size {prop.data.size} "
                f"does not match mask size {prop.mask.size}"
            )
        # Validate that data and mask have compatible element strides.
        # C++ indexes both arrays with the same stride values, so element
        # strides (byte strides / itemsize) must match.  Byte strides
        # naturally differ across float32 (4B) and uint8 (1B) even when
        # the arrays have the same shape and layout.
        data_es = tuple(s // prop.data.itemsize for s in prop.data.strides)
        mask_es = tuple(s // prop.mask.itemsize for s in prop.mask.strides)
        if data_es != mask_es:
            raise ValueError(
                f"create_cont_masked_array: data element strides {data_es} "
                f"do not match mask element strides {mask_es}"
            )
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
    # N2-26: enforce the FWA contract on the property arrays at the FFI
    # boundary (mirror create_ubyte_array:706 / create_float_array:738).
    # m_strides is inert in C++ (the kernels read both buffers flat by
    # node index), so a C-order or strided input would be silently
    # permuted with no exception. IndProperty's setters/ctor already
    # coerce Fortran order; this guard catches direct FFI misuse.
    checkFWA(prop.data)
    checkFWA(prop.mask)
    if grid is None:
        sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
        # F-25: validate that data and mask have the same element count.
        # C++ indexes both arrays with the same stride values, so a
        # data.size != mask.size mismatch reads/writes past the end of the
        # shorter array (heap OOB read for the byte writer).
        if prop.data.size != prop.mask.size:
            raise ValueError(
                f"create_ind_masked_array: data size {prop.data.size} "
                f"does not match mask size {prop.mask.size}"
            )
        # Validate that data and mask have compatible element strides.
        # C++ indexes both arrays with the same stride values, so element
        # strides (byte strides / itemsize) must match.
        data_es = tuple(s // prop.data.itemsize for s in prop.data.strides)
        mask_es = tuple(s // prop.mask.itemsize for s in prop.mask.strides)
        if data_es != mask_es:
            raise ValueError(
                f"create_ind_masked_array: data element strides {data_es} "
                f"do not match mask element strides {mask_es}"
            )
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
        # III-14: preserve the caller's actual mask shape when the array is
        # 3D so the C++ per-dimension shape guard
        # (validate_simulation_mask_shape_or_throw, api.cpp:197-217) can
        # fire. Pre-fix the shape was stamped to (grid.x, grid.y, grid.z),
        # so an equal-volume (2,8,1) mask on a (4,4,1) grid passed the
        # volume check AND the C++ guard (the stamped shape always matched
        # the grid), silently permuting the simulated cells. For 1D (flat)
        # masks the grid dims are used (the flat buffer carries no
        # per-dimension meaning, matching create_cont_masked_array).
        if array.ndim == 3:
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
        # F-26: the grid path had no size check — a float array smaller
        # than the grid volume was passed to C++ with a grid-sized shape
        # struct, and the kernel read past the end of the buffer (heap OOB
        # read). Mirror create_ubyte_array's grid-size validation.
        if grid.x * grid.y * grid.z != array.size:
            raise RuntimeError(
                f"Invalid data size. Size of data = {array.size}. "
                f"Size of grid = {grid.x * grid.y * grid.z}"
            )

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
