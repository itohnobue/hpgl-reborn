# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import contextlib
import ctypes as C
import threading

import numpy

# NumPy 2.0+ compatibility
from numpy import ctypeslib as NC

from .ffi_adapter import (
    require_output_buffer,
)
from .hpgl_wrap import _safe_load_library

cvar = _safe_load_library("_cvariogram", __file__)

# Thread-safe error tracking (mirrors geo.py _error_local pattern).
# The cvariogram C++ library stores the last error globally and does not
# clear it after reading. Without tracking, a stale error from a previous
# C call would cause all subsequent _check_cvar_error calls to falsely
# report failure.
_cvar_error_local = threading.local()
_cvar_error_snapshot_lock = threading.Lock()

# Serializes C++ cvariogram calls with error checking to prevent cross-thread
# error races. The C++ error state is global (not thread_local), so
# concurrent calls from different threads can corrupt error reporting.
# This lock ensures only one thread is in the snapshot→C++ call→check
# window at a time, mirroring geo.py's _hpgl_call_lock.
_cvar_call_lock = threading.Lock()

cvar.cvar_get_last_error.restype = C.c_char_p
cvar.cvar_get_last_error.argtypes = []

# PR-01: clear the process-global C++ error after the guard consumes it.
# cvar_clear_last_error is exported by fresh builds only; the prototype is
# guarded so a stale library (predating the symbol) still imports — the
# freshness check (hpgl_wrap._EXPECTED_LIBRARY_SYMBOLS) warns on staleness.
if hasattr(cvar, "cvar_clear_last_error"):
    cvar.cvar_clear_last_error.restype = None
    cvar.cvar_clear_last_error.argtypes = []

MAX_NUM_LAGS = 10000
MAX_POINT_SET_SIZE = 1_000_000

# F-38: magnitude caps on search-geometry parameters. Without caps, absurd
# values (e.g. lag_separation=1e6 with num_lags=10000) drive the C++
# search-window loop in variograms.cpp calc_search_template_window to an
# effectively unbounded number of iterations (a hang). These Python-side
# caps reject the degenerate inputs at the FFI boundary; cpp-C owns the
# C++ work-based bound.
MAX_LAG_SEPARATION = 100000.0
MAX_FIRST_LAG_DISTANCE = 100000.0
MAX_LAG_WIDTH = 100000.0
MAX_TOL_DISTANCE = 1000.0
MAX_ELLIPSOID_RANGE = 100000.0
# Total lag extent bound: lag_separation * num_lags + first_lag_distance.
# Keeps the C++ search window (which spans ~2x this extent per axis) from
# iterating an unbounded number of cells.
MAX_TOTAL_LAG_EXTENT = 1000000.0


@contextlib.contextmanager
def _cvar_error_guard(context=""):
    """Serialize a C++ cvariogram call with error checking to prevent cross-thread races.

    Usage::

        with _cvar_error_guard("CalcVariograms"):
            cvar.calc_variograms(...)

    Acquires ``_cvar_call_lock`` before the snapshot, holds it across the
    C++ call window, and releases it in ``finally``.  Error checking
    happens inside the ``try`` block so that a ``RuntimeError`` from a
    new C++ error does not prevent lock release.
    """
    _cvar_call_lock.acquire()
    try:
        _snapshot_cvar_error()
        yield
        _check_cvar_error(context)
    finally:
        _cvar_call_lock.release()


def _snapshot_cvar_error():
    """
    Take a snapshot of the current cvariogram C++ error state.

    Call this BEFORE invoking any C++ cvariogram function.
    _check_cvar_error will compare the post-call error against this snapshot
    to detect only NEW errors, avoiding stale error propagation.

    Stores the snapshot in thread-local storage under _cvar_error_snapshot_lock.
    The lock protects the snapshot write against concurrent snapshot reads
    in _check_cvar_error.

    F-37: mirrors the ffi_adapter raised-flag reset. When the previous call
    window RAISED, the C++ global error is treated as consumed (it was
    cleared by _check_cvar_error's clear-on-consume; the flag also covers
    stale libraries that lack cvar_clear_last_error) so a fresh identical
    error in a new call window raises instead of being suppressed as stale.

    III-32: on a STALE library (no cvar_clear_last_error) the clear-on-
    consume in _check_cvar_error is a NO-OP — the C++ error from the
    raising call is still set. A forced None snapshot would then make that
    persistent error look NEW on every subsequent call, permanently
    poisoning all later calls with spurious RuntimeErrors. When clear is
    unavailable, snapshot the CURRENT (persistent) error so a subsequent
    successful call suppresses it as stale instead of locking out.
    """
    with _cvar_error_snapshot_lock:
        if getattr(_cvar_error_local, "_cvar_last_check_raised", False):
            _cvar_error_local._cvar_last_check_raised = False
            if hasattr(cvar, "cvar_clear_last_error"):
                # Clear-on-consume succeeded (fresh library): the C++ error
                # was emptied, so a fresh identical error in a new call
                # window raises (F-37).
                _cvar_error_local._cvar_error_snapshot = None
            else:
                # III-32: stale library without cvar_clear_last_error — the
                # C++ error from the raising call is still set (clear was a
                # no-op). Snapshot the CURRENT error so a subsequent call
                # (successful or failing with the same persistent message)
                # suppresses it as stale rather than locking out forever.
                _cvar_error_local._cvar_error_snapshot = cvar.cvar_get_last_error()
        else:
            _cvar_error_local._cvar_error_snapshot = cvar.cvar_get_last_error()


def _clear_cvar_error():
    """Reset the C++ cvariogram error after the guard consumes it (PR-01).

    The C++ ``last_cvariogram_error`` is process-global and is never cleared
    on success — a failure followed by a successful call would otherwise
    false-raise on the stale error forever. ``cvar_clear_last_error`` is
    exported by fresh builds; when the deployed library predates it (stale
    binary, detected by hpgl_wrap's freshness check), this falls back to a
    no-op and the thread-local snapshot/flag logic still suppresses stale
    errors (pre-fix behavior).
    """
    try:
        cvar.cvar_clear_last_error()
    except AttributeError:
        pass


def _check_cvar_error(context=""):
    """
    Check for NEW cvariogram C++ errors after a computation call.

    Compares current error state against a pre-call snapshot (set via
    _snapshot_cvar_error). Raises RuntimeError ONLY if the error message
    has changed since the snapshot, indicating a new error from the
    current operation rather than a stale error from a previous call.

    Uses thread-local storage so concurrent cvariogram calls in different
    threads each track their own pre/post error state independently.

    The entire read-compare-raise sequence is atomic under
    _cvar_error_snapshot_lock, preventing races between concurrent calls
    to this function that access the same thread-local snapshot.

    PR-01: consumes the C++ error on every path (suppress and raise),
    mirroring ffi_adapter's F-07 clear-on-consume. The C++ error is
    process-global and never cleared on success, so without this a
    failure-then-success sequence raises a spurious RuntimeError on the
    next call (the post-raise snapshot reset alone cannot clear the C++
    state).

    Args:
        context: Description of the operation being checked (e.g. "CalcVariograms")

    Raises:
        RuntimeError: If the C++ computation produced a new error
    """
    with _cvar_error_snapshot_lock:
        err = cvar.cvar_get_last_error()
        if err is not None and len(err) > 0:
            snapshot = getattr(_cvar_error_local, "_cvar_error_snapshot", None)
            if err == snapshot:
                # Same error as pre-call snapshot — stale, suppress. Consume
                # the C++ error so the next call starts from a clean state.
                _cvar_error_local._cvar_last_check_raised = False
                _clear_cvar_error()
                return
            err_str = err.decode("utf-8", errors="replace")
            # Update the snapshot BEFORE raising so re-entry within the
            # same guard window does not double-raise (F-37).
            _cvar_error_local._cvar_error_snapshot = err
            _cvar_error_local._cvar_last_check_raised = True
            # Consume the C++ error before raising so a later call that
            # SUCCEEDS does not see a stale error, while a later call that
            # fails with the identical message raises (PR-01).
            _clear_cvar_error()
            raise RuntimeError(
                f"{context} failed: {err_str}" if context else f"cvariogram error: {err_str}"
            )


class vector_t(C.Structure):
    _fields_ = [("data", C.c_double * 3)]


class ellipsoid_t(C.Structure):
    _fields_ = [
        ("direction1", vector_t),
        ("direction2", vector_t),
        ("direction3", vector_t),
        ("R1", C.c_double),
        ("R2", C.c_double),
        ("R3", C.c_double),
    ]


class variogram_search_template_t(C.Structure):
    _fields_ = [
        ("lag_width", C.c_double),
        ("lag_separation", C.c_double),
        ("tol_distance", C.c_double),
        ("num_lags", C.c_int),
        ("first_lag_distance", C.c_double),
        ("ellipsoid", ellipsoid_t),
    ]


class hard_data_t(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("data_shape", (C.c_int * 3)),
        ("data_strides", (C.c_int * 3)),
        ("mask_shape", (C.c_int * 3)),
        ("mask_strides", (C.c_int * 3)),
    ]


class cont_point_set_t(C.Structure):
    _fields_ = [
        ("xs", C.POINTER(C.c_float)),
        ("ys", C.POINTER(C.c_float)),
        ("zs", C.POINTER(C.c_float)),
        ("values", C.POINTER(C.c_float)),
        ("size", C.c_int),
    ]


class float_data_t(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("data_shape", (C.c_int * 3)),
        ("data_strides", (C.c_int * 3)),
    ]


cvar.fill_ellipsoid_directions.restype = None
cvar.fill_ellipsoid_directions.argtypes = [
    C.POINTER(ellipsoid_t),
    C.c_double,
    C.c_double,
    C.c_double,
]

cvar.calc_variograms.restype = None
cvar.calc_variograms.argtypes = [
    C.POINTER(variogram_search_template_t),
    C.POINTER(hard_data_t),
    NC.ndpointer(dtype=numpy.float32, flags=["W"]),
    C.c_int,
    C.c_int,
]

cvar.calc_variograms_from_point_set.restype = None
cvar.calc_variograms_from_point_set.argtypes = [
    C.POINTER(variogram_search_template_t),
    C.POINTER(cont_point_set_t),
    NC.ndpointer(dtype=numpy.float32, flags=["W"]),
    C.c_int,
]

cvar.cvar_stack_layers.restype = None
cvar.cvar_stack_layers.argtypes = [
    C.POINTER(float_data_t),
    C.POINTER(C.c_int),
    C.c_int,
    C.c_int,
    C.c_float,
    C.c_int,
    C.POINTER(float_data_t),
]


def checked_create(T, **kargs):
    fields = []
    for f, _ in T._fields_:
        fields.append(f)
    # Allow callers to pass explicit references to prevent garbage collection
    # of numpy arrays whose ctypes pointers are passed as struct fields.
    # When ctypes pointers from .ctypes.data_as() are stored in the struct,
    # the original numpy arrays must be kept alive via _array_refs.
    # Match the pattern in geo.py:188 which stores (prop.data, prop.mask).
    refs = kargs.pop("_refs", ())
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
    if fields:
        raise RuntimeError(f"No values for parameters: {fields}")
    result = T(**kargs)
    # Preserve references to input values to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = tuple(refs) if refs else tuple(kargs.values())
    return result


def __strides(array):
    ndim = array.ndim
    if ndim == 1:
        return (1, array.shape[0], array.shape[0])
    elif ndim >= 3:
        return (
            array.strides[0] // array.itemsize,
            array.strides[1] // array.itemsize,
            array.strides[2] // array.itemsize,
        )
    else:
        raise ValueError(
            f"__strides: unsupported ndim={ndim}. "
            f"Expected ndim=1 or ndim>=3 (all current callers use ndim=3)."
        )


def _c_array(t, size, values):
    if len(values) != size:
        raise ValueError(f"_c_array: size {size} does not match len(values) {len(values)}")
    arr = (t * size)(*values)
    arr._array_refs = tuple(values)
    return arr


class Ellipsoid:
    def __init__(self, R1, R2, R3, azimuth, dip, rotation):
        # I2-01: mirror variogram.py TVEllipsoid validation — NaN/Inf and
        # negative ranges pass into C++ where is_in_tunnel is NaN-blind
        # (variograms.cpp:142-149) and silently produce an all-zero
        # variogram. Angles must be finite too (NaN angle → NaN direction
        # vector → silent all-zero).
        for name, val in (("R1", R1), ("R2", R2), ("R3", R3)):
            if not numpy.isfinite(val) or val < 0:
                raise ValueError(
                    f"Ellipsoid: {name} must be finite and non-negative, got {val!r}"
                )
        for name, val in (("azimuth", azimuth), ("dip", dip), ("rotation", rotation)):
            if not numpy.isfinite(val):
                raise ValueError(f"Ellipsoid: {name} must be finite, got {val!r}")
        # F-38: magnitude caps — huge radii inflate the C++ search window
        # and can hang the variogram scan.
        for name, val in (("R1", R1), ("R2", R2), ("R3", R3)):
            if val > MAX_ELLIPSOID_RANGE:
                raise ValueError(
                    f"Ellipsoid: {name} ({val!r}) exceeds maximum {MAX_ELLIPSOID_RANGE}"
                )
        vec = checked_create(vector_t, data=_c_array(C.c_double, 3, (0, 0, 0)))
        self.ell = checked_create(
            ellipsoid_t, direction1=vec, direction2=vec, direction3=vec, R1=R1, R2=R2, R3=R3
        )
        with _cvar_error_guard("fill_ellipsoid_directions"):
            cvar.fill_ellipsoid_directions(C.byref(self.ell), azimuth, dip, rotation)


class VariogramSearchTemplate:
    def __init__(
        self, lag_width, lag_separation, tol_distance, num_lags, first_lag_distance, ellipsoid
    ):
        if num_lags > MAX_NUM_LAGS:
            raise ValueError(
                f"VariogramSearchTemplate: num_lags {num_lags} exceeds maximum {MAX_NUM_LAGS}"
            )
        # I2-01: mirror variogram.py TVVariogramSearchTemplate validation —
        # NaN/Inf/zero/negative geometry must not reach C++ (NaN blindspots
        # produce silent all-zero variograms). num_lags <= 0 is validated at
        # CalcVariograms call time (existing contract).
        if not numpy.isfinite(lag_width) or lag_width <= 0:
            raise ValueError(
                f"VariogramSearchTemplate: lag_width must be finite and positive, got {lag_width!r}"
            )
        if not numpy.isfinite(lag_separation) or lag_separation <= 0:
            raise ValueError(
                f"VariogramSearchTemplate: lag_separation must be finite and positive, "
                f"got {lag_separation!r}"
            )
        if not numpy.isfinite(tol_distance) or tol_distance <= 0:
            raise ValueError(
                f"VariogramSearchTemplate: tol_distance must be finite and positive, "
                f"got {tol_distance!r}"
            )
        if not numpy.isfinite(first_lag_distance) or first_lag_distance < 0:
            raise ValueError(
                f"VariogramSearchTemplate: first_lag_distance must be finite and "
                f"non-negative, got {first_lag_distance!r}"
            )
        # F-38: magnitude caps — the C++ search window extent is
        # ~2*(lag_separation*num_lags + first_lag_distance) per axis, so
        # unbounded values hang the variogram scan.
        if lag_width > MAX_LAG_WIDTH:
            raise ValueError(
                f"VariogramSearchTemplate: lag_width ({lag_width!r}) exceeds maximum "
                f"{MAX_LAG_WIDTH}"
            )
        if lag_separation > MAX_LAG_SEPARATION:
            raise ValueError(
                f"VariogramSearchTemplate: lag_separation ({lag_separation!r}) exceeds "
                f"maximum {MAX_LAG_SEPARATION}"
            )
        if tol_distance > MAX_TOL_DISTANCE:
            raise ValueError(
                f"VariogramSearchTemplate: tol_distance ({tol_distance!r}) exceeds maximum "
                f"{MAX_TOL_DISTANCE}"
            )
        if first_lag_distance > MAX_FIRST_LAG_DISTANCE:
            raise ValueError(
                f"VariogramSearchTemplate: first_lag_distance ({first_lag_distance!r}) "
                f"exceeds maximum {MAX_FIRST_LAG_DISTANCE}"
            )
        if lag_separation * num_lags + first_lag_distance > MAX_TOTAL_LAG_EXTENT:
            raise ValueError(
                f"VariogramSearchTemplate: total lag extent "
                f"(lag_separation*num_lags + first_lag_distance = "
                f"{lag_separation * num_lags + first_lag_distance}) exceeds maximum "
                f"{MAX_TOTAL_LAG_EXTENT}"
            )
        self.templ = checked_create(
            variogram_search_template_t,
            lag_width=lag_width,
            lag_separation=lag_separation,
            tol_distance=tol_distance,
            num_lags=num_lags,
            first_lag_distance=first_lag_distance,
            ellipsoid=ellipsoid.ell,
        )

        self.num_lags = num_lags
        self.lag_separation = lag_separation
        self.first_lag_distance = first_lag_distance


def _validate_template_not_degenerate(templ, context):
    """F-40: reject templates the C++ tunnel filter silently degenerates.

    The C++ ``is_in_tunnel`` (variograms.cpp:127-145) returns false for a
    zero ellipsoid range and for zero direction vectors — so such templates
    produce a silent all-zero variogram (no C++ error is set). A legitimate
    all-zero variogram (e.g. constant data) is still returned; only the
    degenerate TEMPLATE is rejected here.

    Args:
        templ: the ``variogram_search_template_t`` ctypes struct.
        context: operation name for the error message.

    Raises:
        ValueError: If any ellipsoid range is zero or any direction vector
            is the zero vector.
    """
    ell = templ.ellipsoid
    if ell.R1 == 0 or ell.R2 == 0 or ell.R3 == 0:
        raise ValueError(
            f"{context}: degenerate search template — ellipsoid range is zero "
            f"(R1={ell.R1}, R2={ell.R2}, R3={ell.R3}); the C++ tunnel filter "
            f"accepts no pairs and returns an all-zero variogram"
        )
    for name in ("direction1", "direction2", "direction3"):
        vec = getattr(ell, name)
        if all(vec.data[j] == 0.0 for j in range(3)):
            raise ValueError(
                f"{context}: degenerate search template — ellipsoid {name} is the "
                f"zero vector; the C++ tunnel filter accepts no pairs and returns "
                f"an all-zero variogram"
            )


def CalcVariograms(templ, hard_data, percent=100, seed=None):
    if templ.num_lags <= 0:
        raise ValueError("CalcVariograms: num_lags must be positive")
    if percent < 1 or percent > 100:
        raise ValueError(f"CalcVariograms: percent must be in [1, 100], got {percent}")
    # 2-M-3: optional RNG seed for the grid-path percent sampling. When
    # provided, the C++ kernel's thread_local mt19937 is re-seeded with the
    # given value so identical inputs produce identical variograms
    # (reproducible published experiments). None keeps the current
    # non-deterministic behavior.
    if seed is not None:
        if not isinstance(seed, (int, numpy.integer)) or isinstance(seed, bool):
            raise TypeError(
                f"CalcVariograms: seed must be an int, got {type(seed).__name__}"
            )
        if seed < 0:
            raise ValueError(
                f"CalcVariograms: seed must be non-negative, got {seed}"
            )
        if not hasattr(cvar, "calc_variograms_seeded"):
            raise RuntimeError(
                "CalcVariograms: seed= requires a _cvariogram library that "
                "exports calc_variograms_seeded (C++ kernel seeded-RNG "
                "support). The installed library predates the seeded API."
            )
    _validate_template_not_degenerate(templ.templ, "CalcVariograms")
    if not isinstance(hard_data[0], numpy.ndarray) or hard_data[0].dtype != numpy.float32:
        raise TypeError(
            f"CalcVariograms: hard_data[0] must be a float32 ndarray, got {type(hard_data[0]).__name__}"
        )
    if not isinstance(hard_data[1], numpy.ndarray) or hard_data[1].dtype != numpy.uint8:
        raise TypeError(
            f"CalcVariograms: hard_data[1] must be a uint8 ndarray, got {type(hard_data[1]).__name__}"
        )
    if hard_data[0].ndim != 3:
        raise ValueError(
            f"CalcVariograms: hard_data[0] must be 3-dimensional, got {hard_data[0].ndim}d"
        )
    if hard_data[1].ndim != 3:
        raise ValueError(
            f"CalcVariograms: hard_data[1] must be 3-dimensional, got {hard_data[1].ndim}d"
        )
    if any(d <= 0 for d in hard_data[0].shape):
        raise ValueError(
            f"CalcVariograms: all grid dimensions must be positive, got {hard_data[0].shape}"
        )
    if hard_data[0].shape != hard_data[1].shape:
        raise ValueError(
            f"CalcVariograms: hard_data[0].shape {hard_data[0].shape} does not match "
            f"hard_data[1].shape {hard_data[1].shape}"
        )
    # III-16: the C++ grid scan silently SKIPS non-finite informed values
    # (variograms.cpp:690,704 `if (!std::isfinite(v1)) continue;`), producing
    # a silently under-counted variogram, while the pure-Python path raises
    # ValueError (variogram.py:830-835). Reject non-finite INFORMED cells at
    # the FFI boundary for parity. Masked-out cells may legitimately hold
    # sentinel garbage (e.g. -999 / NaN "undefined" values), and the C++
    # never reads them — only informed cells are validated.
    if not numpy.all(numpy.isfinite(hard_data[0][hard_data[1] != 0])):
        raise ValueError(
            "CalcVariograms: hard_data[0] contains NaN or Inf values in "
            "informed (mask != 0) cells"
        )
    variogram = numpy.array([0] * templ.num_lags, dtype="float32")
    # FFI output-buffer contract (got-20260802015741): contiguity + exact
    # size + writeability on the internal buffer the C kernel writes via a
    # raw float pointer (variograms.cpp:570-571). Enforced uniformly with the
    # caller-supplied buffer paths so no facet can be skipped by a future
    # call site.
    require_output_buffer(variogram, templ.num_lags, "CalcVariograms", numpy.dtype("float32"))

    hd = checked_create(
        hard_data_t,
        data=hard_data[0].ctypes.data_as(C.POINTER(C.c_float)),
        mask=hard_data[1].ctypes.data_as(C.POINTER(C.c_ubyte)),
        data_shape=_c_array(C.c_int, 3, hard_data[0].shape),
        mask_shape=_c_array(C.c_int, 3, hard_data[1].shape),
        data_strides=_c_array(C.c_int, 3, __strides(hard_data[0])),
        mask_strides=_c_array(C.c_int, 3, __strides(hard_data[1])),
        _refs=(hard_data[0], hard_data[1]),
    )

    with _cvar_error_guard("CalcVariograms"):
        if seed is not None:
            # 2-M-3: seeded entry point. Declared lazily so a stale library
            # that predates calc_variograms_seeded still imports (the
            # hasattr guard above rejects the call before we get here).
            _seeded = cvar.calc_variograms_seeded
            _seeded.restype = None
            _seeded.argtypes = [
                C.POINTER(variogram_search_template_t),
                C.POINTER(hard_data_t),
                NC.ndpointer(dtype=numpy.float32, flags=["W"]),
                C.c_int,
                C.c_int,
                C.c_uint64,
            ]
            _seeded(
                C.byref(templ.templ),
                C.byref(hd),
                variogram,
                variogram.size,
                percent,
                int(seed),
            )
        else:
            cvar.calc_variograms(
                C.byref(templ.templ), C.byref(hd), variogram, variogram.size, percent
            )

    # Post-call validation: check for NaN/Inf in output
    if numpy.any(numpy.isnan(variogram)) or numpy.any(numpy.isinf(variogram)):
        raise RuntimeError(
            f"CalcVariograms: C function returned NaN or Inf in variogram array "
            f"(num_lags={templ.num_lags})"
        )

    lags_borders = numpy.zeros(templ.num_lags)

    for k in range(templ.num_lags):
        lags_borders[k] = k * templ.lag_separation + templ.first_lag_distance

    return (lags_borders, variogram)


def CalcVariogramsFromPointSet(templ, point_set, variogram):
    if templ.num_lags <= 0:
        raise ValueError("CalcVariogramsFromPointSet: num_lags must be positive")
    _validate_template_not_degenerate(templ.templ, "CalcVariogramsFromPointSet")
    for key in ("X", "Y", "Z", "Property"):
        if key not in point_set:
            raise ValueError(f"CalcVariogramsFromPointSet: point_set missing required key '{key}'")
    # Pre-call size validation: prevent silent C++ failure
    pts = point_set["Property"]
    if len(pts) == 0:
        raise ValueError("CalcVariogramsFromPointSet: point_set must have at least one point")
    if len(pts) > MAX_POINT_SET_SIZE:
        raise ValueError(
            f"CalcVariogramsFromPointSet: point_set size {len(pts)} exceeds "
            f"MAX_POINT_SET_SIZE ({MAX_POINT_SET_SIZE})"
        )
    # Validate coordinate arrays have matching lengths
    x_len = len(point_set["X"])
    y_len = len(point_set["Y"])
    z_len = len(point_set["Z"])
    p_len = len(point_set["Property"])
    if not (x_len == y_len == z_len == p_len):
        raise ValueError(
            f"CalcVariogramsFromPointSet: coordinate array length mismatch: "
            f"len(X)={x_len}, len(Y)={y_len}, len(Z)={z_len}, len(Property)={p_len}"
        )
    # Validate dtypes before raw ctypes pointer cast prevents silent garbage results
    for key in ("X", "Y", "Z", "Property"):
        arr = point_set[key]
        if not isinstance(arr, numpy.ndarray) or arr.dtype != numpy.float32:
            raise TypeError(
                f"CalcVariogramsFromPointSet: point_set['{key}'] must be a float32 ndarray, "
                f"got {type(arr).__name__}"
            )
    # F-M13-py: the C++ point-set scan reads xs/ys/zs/values LINEARLY via
    # pointer arithmetic with no shape/strides metadata (variograms.cpp
    # F-M13-cpp contract), so a non-1D array or a non-contiguous view would
    # be misread as a contiguous flat buffer (garbage values or OOB reads).
    # Enforce ndim == 1 and copy non-contiguous views to contiguous buffers,
    # mirroring the CStackLayers sibling precedent (cvariogram.py:627-632,
    # 659-661). numpy.ascontiguousarray returns the same array when already
    # C-contiguous (no copy); the cont_point_set_t keeps a reference to the
    # NEW arrays via _refs below.
    for key in ("X", "Y", "Z", "Property"):
        arr = point_set[key]
        if arr.ndim != 1:
            raise ValueError(
                f"CalcVariogramsFromPointSet: point_set['{key}'] must be "
                f"1-dimensional, got {arr.ndim}d"
            )
    # III-16: the C++ point-set scan silently SKIPS non-finite property
    # values (variograms.cpp:899 `if (!std::isfinite(v1) || !std::isfinite(v2))
    # continue;`), producing a silently under-counted variogram, while the
    # pure-Python path raises ValueError (variogram.py:830-835). Point sets
    # have no mask — every point is used — so validate all property values
    # for parity at the FFI boundary.
    if not numpy.all(numpy.isfinite(point_set["Property"])):
        raise ValueError(
            "CalcVariogramsFromPointSet: point_set['Property'] contains "
            "NaN or Inf values"
        )
    point_set = {key: numpy.ascontiguousarray(arr) for key, arr in point_set.items()}
    if variogram is None:
        variogram = numpy.array([0] * templ.num_lags, dtype="float32")
    else:
        # II-40: caller-provided output buffer size must match num_lags. The
        # C++ kernel silently computes min(num_lags, result_length)
        # (variograms.cpp:821-823), so an undersized buffer truncates the
        # variogram (lags_borders has num_lags entries, variogram has fewer)
        # with no error. Reject before the call.
        # III-39: the C++ kernel writes into the caller's variogram buffer via
        # a raw float pointer. A read-only buffer is silently mutated (or
        # SIGBUS on a true read-only mmap).
        # Contract: contiguity + size + writeability (got-20260802015741).
        require_output_buffer(
            variogram, templ.num_lags, "CalcVariogramsFromPointSet", numpy.dtype("float32")
        )

    ps = checked_create(
        cont_point_set_t,
        xs=point_set["X"].ctypes.data_as(C.POINTER(C.c_float)),
        ys=point_set["Y"].ctypes.data_as(C.POINTER(C.c_float)),
        zs=point_set["Z"].ctypes.data_as(C.POINTER(C.c_float)),
        values=point_set["Property"].ctypes.data_as(C.POINTER(C.c_float)),
        size=point_set["Property"].size,
        _refs=(point_set["X"], point_set["Y"], point_set["Z"], point_set["Property"]),
    )

    with _cvar_error_guard("CalcVariogramsFromPointSet"):
        cvar.calc_variograms_from_point_set(
            C.byref(templ.templ), C.byref(ps), variogram, variogram.size
        )

    # Post-call validation: check for NaN/Inf in output
    if numpy.any(numpy.isnan(variogram)) or numpy.any(numpy.isinf(variogram)):
        raise RuntimeError(
            f"CalcVariogramsFromPointSet: C function returned NaN or Inf in variogram array "
            f"(num_lags={templ.num_lags})"
        )

    lags_borders = numpy.zeros(templ.num_lags)

    for k in range(templ.num_lags):
        lags_borders[k] = k * templ.lag_separation + templ.first_lag_distance

    return (lags_borders, variogram)


def _create_float_data(array):
    return checked_create(
        float_data_t,
        data=array.ctypes.data_as(C.POINTER(C.c_float)),
        data_shape=_c_array(C.c_int, 3, array.shape),
        data_strides=_c_array(C.c_int, 3, __strides(array)),
        _refs=(array,),
    )


def CStackLayers(layers, markers, nz, scalez, blank_value, result):
    if len(layers) == 0:
        raise ValueError("CStackLayers: layers list is empty")
    if nz <= 0:
        raise ValueError(f"CStackLayers: nz must be positive, got {nz}")
    # Validate scalez and blank_value are finite and positive
    if not numpy.isfinite(scalez):
        raise ValueError(f"CStackLayers: scalez must be finite, got {scalez}")
    if scalez <= 0:
        raise ValueError(f"CStackLayers: scalez must be positive, got {scalez}")
    if not numpy.isfinite(blank_value):
        raise ValueError(f"CStackLayers: blank_value must be finite, got {blank_value}")
    # Validate all layers have matching 2D shapes
    if len(layers) > 1:
        ref_shape = layers[0].shape[:2]
        for i, layer in enumerate(layers[1:], start=1):
            if layer.shape[:2] != ref_shape:
                raise ValueError(
                    f"CStackLayers: layer {i} shape {layer.shape[:2]} does not match "
                    f"reference shape {ref_shape}"
                )
    # Validate result is 3D before indexing shape[2] below
    if result.ndim != 3:
        raise ValueError(f"CStackLayers: result must be 3-dimensional, got {result.ndim}d")
    # F-06: the C++ stack_layers writes result cells at [0, nx) x [0, ny)
    # per layer (stack_layers.h:29-32,57,71). A result whose x/y dims are
    # smaller than the layer dims is a heap OOB WRITE — reject it instead
    # of only validating nz.
    ref_shape = layers[0].shape[:2]
    if result.shape[0] != ref_shape[0] or result.shape[1] != ref_shape[1]:
        raise ValueError(
            f"CStackLayers: result x/y shape {result.shape[:2]} does not match "
            f"layer shape {ref_shape}"
        )
    # Validate nz fits within result array dimensions
    if nz > result.shape[2]:
        raise ValueError(f"CStackLayers: nz ({nz}) exceeds result.shape[2] ({result.shape[2]})")
    # Validate markers count matches layers count
    if len(markers) != len(layers):
        raise ValueError(
            f"CStackLayers: len(markers) ({len(markers)}) must match len(layers) ({len(layers)})"
        )
    # Validate dtypes before raw ctypes pointer cast prevents silent garbage results
    for i, layer in enumerate(layers):
        if not isinstance(layer, numpy.ndarray) or layer.dtype != numpy.float32:
            raise TypeError(
                f"CStackLayers: layer {i} must be a float32 ndarray, "
                f"got {type(layer).__name__}"
            )
    if not isinstance(result, numpy.ndarray) or result.dtype != numpy.float32:
        raise TypeError(
            f"CStackLayers: result must be a float32 ndarray, "
            f"got {type(result).__name__}"
        )
    # F-08: non-contiguous/sliced layer arrays produce strides-based
    # map_index values beyond the C++ cumulative_k buffer (OOB write,
    # stack_layers.h:33,38-39). Copy layers to contiguous float32; a
    # sliced view is copied, so the C++ sees a safe contiguous buffer.
    layers = [numpy.ascontiguousarray(layer) for layer in layers]
    # The result is the OUTPUT buffer — the full FFI output-buffer contract
    # (got-20260802015741): contiguity + exact size + writeability. The C++
    # kernel writes result cells at [0, nx) x [0, ny) per layer
    # (stack_layers.h:29-32,57,71); a non-contiguous or short result would
    # OOB-write. The 3D shape checks above (result.ndim, x/y match, nz cap)
    # establish the expected element count. Full-init (III-40) is guaranteed
    # by the C++ kernel, which pre-fills the ENTIRE result buffer with
    # blank_value before the layer loop (stack_layers.h:106-112), so no
    # sentinel prefill is needed here.
    require_output_buffer(result, result.size, "CStackLayers", numpy.dtype("float32"))
    layers2 = []
    for layer in layers:
        layers2.append(_create_float_data(layer))

    result2 = _create_float_data(result)
    with _cvar_error_guard("CStackLayers"):
        cvar.cvar_stack_layers(
            _c_array(float_data_t, len(layers2), layers2),
            _c_array(C.c_int, len(markers), markers),
            len(layers2),
            nz,
            scalez,
            blank_value,
            C.byref(result2),
        )

    # Post-call validation: check for NaN/Inf in the result array
    if numpy.any(numpy.isnan(result)) or numpy.any(numpy.isinf(result)):
        raise RuntimeError(
            f"CStackLayers: C function produced NaN or Inf in result array "
            f"(nz={nz}, nlayers={len(layers2)})"
        )
