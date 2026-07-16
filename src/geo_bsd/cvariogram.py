# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import contextlib
import ctypes as C
import threading

import numpy

# NumPy 2.0+ compatibility
from numpy import ctypeslib as NC

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

MAX_NUM_LAGS = 10000
MAX_POINT_SET_SIZE = 1_000_000


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
    _snapshot_cvar_error()
    try:
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
    """
    with _cvar_error_snapshot_lock:
        _cvar_error_local._cvar_error_snapshot = cvar.cvar_get_last_error()


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
                # Same error as pre-call snapshot — stale, suppress.
                return
            err_str = err.decode("utf-8", errors="replace")
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
    NC.ndpointer(dtype=numpy.float32),
    C.c_int,
    C.c_int,
]

cvar.calc_variograms_from_point_set.restype = None
cvar.calc_variograms_from_point_set.argtypes = [
    C.POINTER(variogram_search_template_t),
    C.POINTER(cont_point_set_t),
    NC.ndpointer(dtype=numpy.float32),
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


def CalcVariograms(templ, hard_data, percent=100):
    if templ.num_lags <= 0:
        raise ValueError("CalcVariograms: num_lags must be positive")
    if percent < 1 or percent > 100:
        raise ValueError(f"CalcVariograms: percent must be in [1, 100], got {percent}")
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
    variogram = numpy.array([0] * templ.num_lags, dtype="float32")

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
        cvar.calc_variograms(C.byref(templ.templ), C.byref(hd), variogram, variogram.size, percent)

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
    if variogram is None:
        variogram = numpy.array([0] * templ.num_lags, dtype="float32")

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
    # Validate nz fits within result array dimensions
    if nz > result.shape[2]:
        raise ValueError(f"CStackLayers: nz ({nz}) exceeds result.shape[2] ({result.shape[2]})")
    # Validate markers count matches layers count
    if len(markers) != len(layers):
        raise ValueError(
            f"CStackLayers: len(markers) ({len(markers)}) must match len(layers) ({len(layers)})"
        )
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
