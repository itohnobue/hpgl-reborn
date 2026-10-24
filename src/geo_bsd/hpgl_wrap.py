# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
r"""Low-level ctypes wrappers for the HPGL C shared library (libhpgl).

This module defines the ctypes structures and function prototypes for
all HPGL C API functions.  It is consumed by :mod:`geo_bsd.geo` which
provides the user-facing Python API.

.. note::

   **Kriging variance is not exposed through the Python API.**

   13 of the 27 exports use ``restype = None`` (void return); the other 14
   declare a non-void ``restype``: the C entry points that return a status
   code (``c_int``, 10 exports), a string (``c_char_p``, 3 exports), or the
   kriging-stats struct by value (1 export).  The C++ kriging functions
   write the kriging variance to an internal field on the kriging output
   type, but the C API boundary does not include a variance-returning
   function, and no Python wrapper exposes kriging variance (see
   ``test_regression_v150.py:136-138`` for the self-documented gap).

   If kriging variance access is needed from Python, the C API would
   need a new function (e.g. ``hpgl_get_kriging_variance``) that
   returns the per-cell variance as a float array, and a corresponding
   wrapper would be added here.

"""

from __future__ import annotations

import ctypes as C
import hashlib
import logging
import os
import pathlib
import sys

import numpy

# NumPy 2.0+ compatibility
from numpy import ctypeslib as NC


# Since numpy>=2.0 is required, always use direct ctypes.CDLL
def _load_lib_func(libpath: str) -> C.CDLL:
    return C.CDLL(str(libpath))


ndpointer = NC.ndpointer

hpgl_output_handler = C.CFUNCTYPE(C.c_int, C.c_char_p, C.py_object)
hpgl_progress_handler = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_int, C.py_object)


class _HPGL_MEAN_KIND:
    stationary_auto = 0
    stationary = 1
    varying = 2


class _HPGL_KRIGING_KIND:
    ordinary = 0
    simple = 1


class _HPGL_SHAPE(C.Structure):
    _fields_ = [("m_data", C.c_int * 3), ("m_strides", C.c_int * 3)]


class _HPGL_CONT_MASKED_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("shape", _HPGL_SHAPE),
    ]


class _HPGL_IND_MASKED_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_ubyte)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("shape", _HPGL_SHAPE),
        ("indicator_count", C.c_int),
    ]


class _HPGL_UBYTE_ARRAY(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_ubyte)), ("shape", _HPGL_SHAPE)]


class _HPGL_FLOAT_ARRAY(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("shape", _HPGL_SHAPE)]


class _HPGL_OK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
    ]


class _HPGL_SK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("automatic_mean", C.c_ubyte),
        ("mean", C.c_double),
    ]


class _HPGL_SGS_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("kriging_kind", C.c_int),
        ("seed", C.c_int64),
        ("min_neighbours", C.c_int),
    ]


class _HPGL_MEDIAN_IK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("marginal_probs", C.c_double * 2),
    ]


class _HPGL_IK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("marginal_prob", C.c_double),
    ]


class __hpgl_cov_params_t(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
    ]


class __hpgl_cockriging_m1_params_t(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("primary_mean", C.c_double),
        ("secondary_mean", C.c_double),
        ("secondary_variance", C.c_double),
        ("correlation_coef", C.c_double),
    ]


# Module-level reference to __hpgl_cov_params_t to avoid name mangling in class _fields_
_hpgl_cov_params_t_ref = __hpgl_cov_params_t


class __hpgl_cockriging_m2_params_t(C.Structure):
    _fields_ = [
        ("primary_cov_params", _hpgl_cov_params_t_ref),
        ("secondary_cov_params", _hpgl_cov_params_t_ref),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("primary_mean", C.c_double),
        ("secondary_mean", C.c_double),
        ("correlation_coef", C.c_double),
    ]


class hpgl_non_parametric_cdf_t(C.Structure):
    _fields_ = [
        ("values", C.POINTER(C.c_float)),
        ("probs", C.POINTER(C.c_float)),
        ("size", C.c_longlong),
    ]


class _HPGLKrigingStats(C.Structure):
    """ctypes mirror of C hpgl_kriging_stats_t (api.h:160-167).

    Fields use c_ulong for unsigned long (4 bytes on Windows, 8 bytes
    on Unix — matching the platform C convention).
    """

    _fields_ = [
        ("m_points_calculated", C.c_ulong),
        ("m_points_without_neighbours", C.c_ulong),
        ("m_points_singularity", C.c_ulong),
        ("m_mean", C.c_double),
        ("m_speed_nps", C.c_double),
    ]


_hpgl_so = None

# Expected SHA-256 hashes of known-good library builds.
# Populated from current builds at development time. Mismatch triggers
# a warning (not a hard error) to support development rebuilds.
_EXPECTED_LIBRARY_HASHES: dict[str, str] = {}

# Symbols that a FRESH library build must export. A loaded library missing
# these is stale (predates the API) or the wrong binary — the freshness
# check (F-03) warns so a stale-lib shadow cannot pass silently. This is the
# meaningful replacement for the empty-dict hash guard (which was a permanent
# no-op).
#
# F-071: the list must cover EVERY hpgl_* symbol this module dereferences at
# module scope (the ``_hpgl_so.<name>`` block below). Checking a single
# symbol — and the one whose absence the module is built to tolerate, at that —
# made the gate blind to a symbol-identical stale engine (live instance: an
# engine stamped 2.0.5 loaded against a 2.0.6 package with zero diagnostics).
# What remains open is artifact IDENTITY: nothing records the hash of the
# validated build, so the ``_EXPECTED_LIBRARY_HASHES`` gate above is still a
# no-op until the build publishes that manifest.
_HPGL_FATAL_SYMBOLS: tuple[str, ...] = (
    "hpgl_get_last_exception_message",
    "hpgl_set_output_handler",
    "hpgl_set_progress_handler",
    "hpgl_ordinary_kriging",
    "hpgl_simple_kriging",
    "hpgl_lvm_kriging",
    "hpgl_simple_kriging_weights",
    "hpgl_sgs_simulation",
    "hpgl_sgs_lvm_simulation",
    "hpgl_median_ik",
    "hpgl_indicator_kriging",
    "hpgl_set_thread_num",
    "hpgl_get_thread_num",
    "hpgl_get_api_validation_registry_count",
    "hpgl_get_api_validation_registry_name",
    "hpgl_get_api_validation_registry_validation",
    "hpgl_read_inc_file_float",
    "hpgl_read_inc_file_byte",
    "hpgl_write_inc_file_float",
    "hpgl_write_inc_file_byte",
    "hpgl_write_gslib_cont_property",
    "hpgl_write_gslib_byte_property",
    "hpgl_sis_simulation",
    "hpgl_sis_simulation_lvm",
    "hpgl_simple_cokriging_mark1",
    "hpgl_simple_cokriging_mark2",
    # Tolerated at import (guarded by try/except AttributeError below), but a
    # library without it is still an older build — keep it in the gate.
    "hpgl_get_kriging_stats",
)

_EXPECTED_LIBRARY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "hpgl": _HPGL_FATAL_SYMBOLS,
    "hpgl_d": _HPGL_FATAL_SYMBOLS,
    "_cvariogram": (
        "cvar_get_last_error",
        "cvar_stack_layers",
        "cvar_clear_last_error",
        # F-071 sibling sweep: the remaining fatal _cvariogram exports
        # (cvariogram.py:259-293 dereferences them at module scope).
        "fill_ellipsoid_directions",
        "calc_variograms",
        "calc_variograms_from_point_set",
    ),
}

logger = logging.getLogger(__name__)


# Security: Validate and safely load the native library
def _safe_load_library(lib_name: str, ref_file: str):
    """
    Safely loads a native library with path validation to prevent
    directory traversal and library loading attacks.

    Args:
        lib_name: Name of the library to load (without extension/prefix)
        ref_file: Reference file path (typically __file__) used to locate library

    Returns:
        Loaded ctypes library

    Raises:
        ValueError: If library path validation fails
        OSError: If library cannot be loaded
    """
    # Validate the reference file path
    if not ref_file:
        raise ValueError("Reference file path cannot be empty")

    # Convert to absolute path and normalize
    ref_path = pathlib.Path(ref_file).resolve()

    # The library should be in the same directory as the reference file
    lib_dir = ref_path.parent

    # Try platform-specific library names.
    #
    # F-03: prefer the build-output name ({name}) over the lib-prefixed
    # name. build.sh copies the fresh binary as hpgl.dylib / _cvariogram.dylib,
    # so a leftover libhpgl.dylib / lib_cvariogram.dylib (or a stale
    # cross-platform .so) is an OLD build that must not shadow the fresh one.
    # The lib-prefixed names remain as a fallback for environments where the
    # build was installed under the conventional lib{name} name.
    lib_paths = []

    if sys.platform.startswith("win"):
        # Windows: .dll or .pyd extensions
        lib_paths.extend(
            [
                lib_dir / f"{lib_name}.dll",
                lib_dir / f"{lib_name}.pyd",
                lib_dir / f"lib{lib_name}.dll",
                lib_dir / f"{lib_name}_d.dll",  # Debug version
                lib_dir / f"lib{lib_name}_d.dll",
            ]
        )
    elif sys.platform.startswith("darwin"):
        # macOS: .dylib or .so extensions
        lib_paths.extend(
            [
                lib_dir / f"{lib_name}.dylib",
                lib_dir / f"{lib_name}.so",
                lib_dir / f"lib{lib_name}.dylib",
                lib_dir / f"lib{lib_name}.so",
            ]
        )
    else:  # Linux and others
        lib_paths.extend(
            [
                lib_dir / f"{lib_name}.so",
                lib_dir / f"lib{lib_name}.so",
            ]
        )

    # Try each library path
    load_failures: list[tuple[pathlib.Path, OSError]] = []
    for lib_path in lib_paths:
        if not lib_path.exists():
            continue
        # Validate the resolved library path is within allowed directories
        resolved_lib = lib_path.resolve()
        # Ensure library is in the same directory tree as the reference file.
        # The containment test runs BEFORE the load: a library opened and then
        # rejected would stay mapped for the process lifetime (ctypes exposes
        # no dlclose and no __del__), so nothing must be opened until its path
        # is known safe (G-39).
        try:
            # Verify the library is in a subdirectory of ref_path.parent
            resolved_lib.relative_to(lib_dir)
        except ValueError as err:
            # Library path escapes allowed directory
            raise ValueError(
                f"Library path {resolved_lib} is outside allowed directory {lib_dir}"
            ) from err
        try:
            lib = _load_lib_func(str(resolved_lib))
        except OSError as err:
            # Wrong architecture or incompatible library — try next path.
            # F-073: record the failure and log it. Falling through silently
            # handed back the next candidate — on macOS the older `lib{name}`
            # naming — so a corrupt or quarantined preferred binary made the
            # process run the WRONG engine with no record of the downgrade.
            load_failures.append((resolved_lib, err))
            logger.warning(
                "_safe_load_library: candidate %s exists but failed to load (%s); "
                "trying the next candidate",
                resolved_lib,
                err,
            )
            continue
        _verify_library_hash(lib_name, resolved_lib)
        _verify_library_freshness(lib_name, lib)
        return lib

    # If not found, try the original load_library behavior as fallback
    # but wrap it with additional validation
    fallback_path = pathlib.Path(os.path.join(str(ref_path.parent), lib_name))
    # G-39: same containment-before-load ordering as the candidate loop above.
    try:
        fallback_resolved = fallback_path.resolve()
        fallback_resolved.relative_to(lib_dir)
    except ValueError as err:
        raise ValueError(
            f"Loaded library {fallback_resolved} is outside allowed directory {lib_dir}"
        ) from err
    except OSError as err:
        raise OSError(
            f"Cannot load library '{lib_name}': {fallback_path} cannot be resolved: {err}"
        ) from err
    try:
        lib = _load_lib_func(str(fallback_path))
    except OSError as fallback_err:
        # F-072: report the candidate FILENAMES and their real per-candidate
        # errors. The previous message listed the search directory once per
        # candidate and reported only the extension-less fallback's dlopen
        # error — a path no build step ever produces — so the actionable
        # cause (wrong architecture, missing dependency, permission,
        # quarantine) was discarded.
        if load_failures:
            details = "; ".join(f"{path.name}: {err}" for path, err in load_failures)
        else:
            details = "no candidate file exists"
        candidates = ", ".join(path.name for path in lib_paths)
        raise OSError(
            f"Cannot load library '{lib_name}'. Candidates searched in {lib_dir}: "
            f"{candidates}. Candidate errors — {details}. "
            f"Extension-less fallback '{fallback_path.name}' also failed: {fallback_err}"
        ) from fallback_err

    _verify_library_hash(lib_name, fallback_path)
    _verify_library_freshness(lib_name, lib)
    return lib


def _verify_library_hash(lib_name: str, lib_path: pathlib.Path) -> None:
    """Verify the SHA-256 hash of a loaded library against expected values.

    Logs a warning if the hash doesn't match any known-good hash.
    Does NOT raise — development rebuilds need to work without
    pre-registered hashes.

    Args:
        lib_name: Logical name of the library (e.g. 'hpgl')
        lib_path: Resolved path to the library file
    """
    if not _EXPECTED_LIBRARY_HASHES:
        return  # No expected hashes registered — skip wasted I/O + CPU

    try:
        with open(lib_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        logger.debug("_verify_library_hash: cannot read %s: %s", lib_path, e)
        return

    # Look for a matching expected hash
    for expected_name, expected_hash in _EXPECTED_LIBRARY_HASHES.items():
        if file_hash == expected_hash:
            logger.debug(
                "_verify_library_hash: %s hash matches expected (%s)", lib_name, expected_name
            )
            return

    # No match found — warn but don't fail
    if _EXPECTED_LIBRARY_HASHES:
        logger.warning(
            "_verify_library_hash: %s (%s) hash %s does not match any expected hash. "
            "Library may have been modified.",
            lib_name,
            lib_path,
            file_hash,
        )


def _verify_library_freshness(lib_name: str, lib: C.CDLL) -> None:
    """Warn when a loaded library lacks symbols a fresh build must export.

    The old hash guard was a permanent no-op because
    ``_EXPECTED_LIBRARY_HASHES`` is never populated. This symbol check is
    the meaningful freshness gate (F-03): a stale binary (e.g. the Jun-27
    ``libhpgl.dylib`` predating ``hpgl_get_kriging_stats``) is detected by
    the missing symbol and reported loudly instead of silently shadowing
    the fresh build.

    Warns (does not raise) so development rebuilds keep working.

    Args:
        lib_name: Logical name of the library (e.g. 'hpgl', '_cvariogram').
        lib: The loaded ctypes CDLL object.
    """
    expected = _EXPECTED_LIBRARY_SYMBOLS.get(lib_name)
    if not expected:
        return  # Unknown library name — nothing to check
    missing = [sym for sym in expected if not hasattr(lib, sym)]
    if missing:
        logger.warning(
            "_verify_library_freshness: %s (%s) does not export expected "
            "symbols %s — the loaded binary is stale or the wrong library. "
            "Rebuild the native library.",
            lib_name,
            getattr(lib, "_name", "<unknown>"),
            missing,
        )


if "HPGL_DEBUG" in os.environ:
    try:
        _hpgl_so = _safe_load_library("hpgl_d", __file__)
    except OSError:
        # Debug-suffixed library not found: CMake only sets DEBUG_POSTFIX
        # on Windows. Non-Windows debug builds use the standard name.
        # Fall back to the unsuffixed library name.
        logger.warning(
            "HPGL_DEBUG is set but hpgl_d library not found. Falling back to 'hpgl' (release name)."
        )
        _hpgl_so = _safe_load_library("hpgl", __file__)
else:
    _hpgl_so = _safe_load_library("hpgl", __file__)

_hpgl_so.hpgl_get_last_exception_message.restype = C.c_char_p
_hpgl_so.hpgl_get_last_exception_message.argtypes = []

_hpgl_so.hpgl_set_output_handler.restype = None
_hpgl_so.hpgl_set_output_handler.argtypes = [hpgl_output_handler, C.py_object]

_hpgl_so.hpgl_set_progress_handler.restype = None
_hpgl_so.hpgl_set_progress_handler.argtypes = [hpgl_progress_handler, C.py_object]

_hpgl_so.hpgl_ordinary_kriging.restype = None
_hpgl_so.hpgl_ordinary_kriging.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_OK_PARAMS),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
]

_hpgl_so.hpgl_simple_kriging.restype = None
_hpgl_so.hpgl_simple_kriging.argtypes = [
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    C.POINTER(_HPGL_SHAPE),
    C.POINTER(_HPGL_SK_PARAMS),
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    C.POINTER(_HPGL_SHAPE),
]

_hpgl_so.hpgl_lvm_kriging.restype = None
_hpgl_so.hpgl_lvm_kriging.argtypes = [
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    C.POINTER(_HPGL_SHAPE),
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    C.POINTER(_HPGL_SHAPE),
    C.POINTER(_HPGL_OK_PARAMS),
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    C.POINTER(_HPGL_SHAPE),
]

_hpgl_so.hpgl_simple_kriging_weights.restype = C.c_int
_hpgl_so.hpgl_simple_kriging_weights.argtypes = [
    (C.c_float * 3),
    NC.ndpointer(dtype=numpy.float32),
    NC.ndpointer(dtype=numpy.float32),
    NC.ndpointer(dtype=numpy.float32),
    C.c_int,
    C.POINTER(__hpgl_cov_params_t),
    NC.ndpointer(dtype=numpy.float32),
]

_hpgl_so.hpgl_sgs_simulation.restype = None
_hpgl_so.hpgl_sgs_simulation.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_SGS_PARAMS),
    C.POINTER(hpgl_non_parametric_cdf_t),
    C.POINTER(C.c_double),
    C.POINTER(_HPGL_UBYTE_ARRAY),
]

_hpgl_so.hpgl_sgs_lvm_simulation.restype = None
_hpgl_so.hpgl_sgs_lvm_simulation.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_SGS_PARAMS),
    C.POINTER(hpgl_non_parametric_cdf_t),
    C.POINTER(_HPGL_FLOAT_ARRAY),
    C.POINTER(_HPGL_UBYTE_ARRAY),
]

_hpgl_so.hpgl_median_ik.restype = None
_hpgl_so.hpgl_median_ik.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_MEDIAN_IK_PARAMS),
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
]

_hpgl_so.hpgl_indicator_kriging.restype = None
_hpgl_so.hpgl_indicator_kriging.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.c_int,
]

_hpgl_so.hpgl_set_thread_num.restype = C.c_int
_hpgl_so.hpgl_set_thread_num.argtypes = [C.c_int]

_hpgl_so.hpgl_get_thread_num.restype = C.c_int
_hpgl_so.hpgl_get_thread_num.argtypes = []

# C-API validation registry (pat-20260802223236). The generated completeness
# test (test_api_validation_registry.py) walks the exported hpgl_* symbols and
# fails when any entry point is missing a registry row — a new entry point
# without mirror-validation is a test-time failure, not a silent recurrence.
_hpgl_so.hpgl_get_api_validation_registry_count.restype = C.c_int
_hpgl_so.hpgl_get_api_validation_registry_count.argtypes = []

_hpgl_so.hpgl_get_api_validation_registry_name.restype = C.c_char_p
_hpgl_so.hpgl_get_api_validation_registry_name.argtypes = [C.c_int]

_hpgl_so.hpgl_get_api_validation_registry_validation.restype = C.c_char_p
_hpgl_so.hpgl_get_api_validation_registry_validation.argtypes = [C.c_int]

_hpgl_so.hpgl_read_inc_file_float.restype = C.c_int
_hpgl_so.hpgl_read_inc_file_float.argtypes = [
    C.c_char_p,
    C.c_float,
    C.c_int,
    NC.ndpointer(dtype=numpy.float32, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
]

_hpgl_so.hpgl_read_inc_file_byte.restype = C.c_int
_hpgl_so.hpgl_read_inc_file_byte.argtypes = [
    C.c_char_p,
    C.c_int,
    C.c_int,
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    NC.ndpointer(dtype=numpy.ubyte, flags=["F", "W", "A"]),
    C.c_int,
]

_hpgl_so.hpgl_write_inc_file_float.restype = C.c_int
_hpgl_so.hpgl_write_inc_file_float.argtypes = [
    C.c_char_p,
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.c_float,
    C.c_char_p,
]

_hpgl_so.hpgl_write_inc_file_byte.restype = C.c_int
_hpgl_so.hpgl_write_inc_file_byte.argtypes = [
    C.c_char_p,
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.c_int,
    C.c_char_p,
    C.POINTER(C.c_ubyte),
    C.c_int,
]

_hpgl_so.hpgl_write_gslib_cont_property.restype = C.c_int
_hpgl_so.hpgl_write_gslib_cont_property.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.c_char_p,
    C.c_char_p,
    C.c_double,
]

_hpgl_so.hpgl_write_gslib_byte_property.restype = C.c_int
_hpgl_so.hpgl_write_gslib_byte_property.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.c_char_p,
    C.c_char_p,
    C.c_double,
    C.POINTER(C.c_ubyte),
    C.c_int,
]


_hpgl_so.hpgl_sis_simulation.restype = None
_hpgl_so.hpgl_sis_simulation.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.c_int,
    C.c_int64,
    C.POINTER(_HPGL_UBYTE_ARRAY),
]

_hpgl_so.hpgl_sis_simulation_lvm.restype = None
_hpgl_so.hpgl_sis_simulation_lvm.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.POINTER(_HPGL_FLOAT_ARRAY),
    C.c_int,
    C.c_int64,
    C.POINTER(_HPGL_UBYTE_ARRAY),
    C.c_int,
]

_hpgl_so.hpgl_simple_cokriging_mark1.restype = None
_hpgl_so.hpgl_simple_cokriging_mark1.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(__hpgl_cockriging_m1_params_t),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
]

_hpgl_so.hpgl_simple_cokriging_mark2.restype = None
_hpgl_so.hpgl_simple_cokriging_mark2.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(__hpgl_cockriging_m2_params_t),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
]

try:
    _hpgl_so.hpgl_get_kriging_stats.restype = _HPGLKrigingStats
    _hpgl_so.hpgl_get_kriging_stats.argtypes = []
    _HAS_KRIGING_STATS = True
except AttributeError:
    _HAS_KRIGING_STATS = False


def get_kriging_stats() -> dict[str, int | float]:
    """Return kriging statistics from the most recent kriging call as a dict.

    The C++ engine zero-initializes the thread-local stats at the top of
    every kriging/simulation entry point (F-N2: ``reset_kriging_stats`` in
    api.cpp), and each entry point whose algorithm calls
    ``set_kriging_stats`` (OK/SK/LVM/cokriging markI/II, median_ik, IK,
    SGS, SIS) populates them. So after any kriging/simulation call this
    returns THAT call's stats — never stale data from a prior, different
    call (matches the api.h:188-193 promise). A call to an entry point
    that does not populate (e.g. ``simple_kriging_weights``) leaves the
    zero-initialized values.

    Returns:
        dict with keys: points_calculated, points_without_neighbours,
        points_singularity, mean, speed_nps

    Raises:
        NotImplementedError: if the native library does not export
            hpgl_get_kriging_stats (requires a library rebuild with
            EX-006 fix applied).
    """
    if not _HAS_KRIGING_STATS:
        raise NotImplementedError(
            "hpgl_get_kriging_stats is not available in the current library build. "
            "Rebuild the native library after applying the EX-006 fix "
            "(set_kriging_stats integration)."
        )
    raw = _hpgl_so.hpgl_get_kriging_stats()
    return {
        "points_calculated": raw.m_points_calculated,
        "points_without_neighbours": raw.m_points_without_neighbours,
        "points_singularity": raw.m_points_singularity,
        "mean": raw.m_mean,
        "speed_nps": raw.m_speed_nps,
    }
