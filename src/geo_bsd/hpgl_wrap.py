# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
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
    _fields_ = [("m_data", C.c_int * 3 ),
            ("m_strides", C.c_int * 3)]

class _HPGL_CONT_MASKED_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("shape", _HPGL_SHAPE)]

class _HPGL_IND_MASKED_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_ubyte)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("shape", _HPGL_SHAPE),
        ("indicator_count", C.c_int)]

class _HPGL_UBYTE_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_ubyte)),
        ("shape", _HPGL_SHAPE)]

class _HPGL_FLOAT_ARRAY(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("shape", _HPGL_SHAPE)]

class _HPGL_OK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int)]

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
        ("mean", C.c_double)]

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
        ("marginal_probs", C.c_double * 2)]

class _HPGL_IK_PARAMS(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double),
        ("radiuses", C.c_int * 3),
        ("max_neighbours", C.c_int),
        ("marginal_prob", C.c_double)]

class __hpgl_cov_params_t(C.Structure):
    _fields_ = [
        ("covariance_type", C.c_int),
        ("ranges", C.c_double * 3),
        ("angles", C.c_double * 3),
        ("sill", C.c_double),
        ("nugget", C.c_double)]

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
        ("correlation_coef", C.c_double)]

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
        ("correlation_coef", C.c_double)]

class hpgl_non_parametric_cdf_t(C.Structure):
    _fields_ = [
        ("values", C.POINTER(C.c_float)),
        ("probs", C.POINTER(C.c_float)),
        ("size", C.c_longlong)]

_hpgl_so = None

# Expected SHA-256 hashes of known-good library builds.
# Populated from current builds at development time. Mismatch triggers
# a warning (not a hard error) to support development rebuilds.
_EXPECTED_LIBRARY_HASHES: dict[str, str] = {}

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

    # Try platform-specific library names
    lib_paths = []

    if sys.platform.startswith('win'):
        # Windows: .dll or .pyd extensions
        lib_paths.extend([
            lib_dir / f"{lib_name}.dll",
            lib_dir / f"{lib_name}.pyd",
            lib_dir / f"lib{lib_name}.dll",
            lib_dir / f"{lib_name}_d.dll",  # Debug version
            lib_dir / f"lib{lib_name}_d.dll",
        ])
    elif sys.platform.startswith('darwin'):
        # macOS: .dylib or .so extensions
        lib_paths.extend([
            lib_dir / f"lib{lib_name}.dylib",
            lib_dir / f"lib{lib_name}.so",
            lib_dir / f"{lib_name}.dylib",
            lib_dir / f"{lib_name}.so",
        ])
    else:  # Linux and others
        lib_paths.extend([
            lib_dir / f"lib{lib_name}.so",
            lib_dir / f"{lib_name}.so",
        ])

    # Try each library path
    for lib_path in lib_paths:
        if lib_path.exists():
            # Validate the resolved library path is within allowed directories
            resolved_lib = lib_path.resolve()
            # Ensure library is in the same directory tree as the reference file
            try:
                # Verify the library is in a subdirectory of ref_path.parent
                resolved_lib.relative_to(lib_dir)
                lib = _load_lib_func(str(resolved_lib))
                _verify_library_hash(lib_name, resolved_lib)
                return lib
            except ValueError as err:
                # Library path escapes allowed directory
                raise ValueError(
                    f"Library path {resolved_lib} is outside allowed directory {lib_dir}"
                ) from err

    # If not found, try the original load_library behavior as fallback
    # but wrap it with additional validation
    try:
        lib = _load_lib_func(os.path.join(str(ref_path.parent), lib_name))
        # Verify the loaded library path is safe
        if hasattr(lib, '_name'):
            loaded_path = pathlib.Path(lib._name)
            # If the library name is relative, resolve relative to lib_dir
            if not loaded_path.is_absolute():
                loaded_path = lib_dir / loaded_path
            if loaded_path.exists():
                resolved = loaded_path.resolve()
                try:
                    resolved.relative_to(lib_dir)
                except ValueError as err:
                    raise ValueError(
                        f"Loaded library {resolved} is outside allowed directory {lib_dir}"
                    ) from err
        _verify_library_hash(lib_name, pathlib.Path(os.path.join(str(ref_path.parent), lib_name)))
        return lib
    except OSError as e:
        # Library not found or cannot be loaded
        lib_dirs_str = ', '.join(str(p.parent) for p in lib_paths)
        raise OSError(
            f"Cannot load library '{lib_name}'. Searched in: {lib_dirs_str}. "
            f"Original error: {e}"
        ) from e

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
        with open(lib_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        logger.debug("_verify_library_hash: cannot read %s: %s", lib_path, e)
        return

    # Look for a matching expected hash
    for expected_name, expected_hash in _EXPECTED_LIBRARY_HASHES.items():
        if file_hash == expected_hash:
            logger.debug("_verify_library_hash: %s hash matches expected (%s)", lib_name, expected_name)
            return

    # No match found — warn but don't fail
    if _EXPECTED_LIBRARY_HASHES:
        logger.warning(
            "_verify_library_hash: %s (%s) hash %s does not match any expected hash. "
            "Library may have been modified.",
            lib_name, lib_path, file_hash
        )

if 'HPGL_DEBUG' in os.environ:
    try:
        _hpgl_so = _safe_load_library('hpgl_d', __file__)
    except OSError:
        # Debug-suffixed library not found: CMake only sets DEBUG_POSTFIX
        # on Windows. Non-Windows debug builds use the standard name.
        # Fall back to the unsuffixed library name.
        logger.warning(
            "HPGL_DEBUG is set but hpgl_d library not found. "
            "Falling back to 'hpgl' (release name)."
        )
        _hpgl_so = _safe_load_library('hpgl', __file__)
else:
    _hpgl_so = _safe_load_library('hpgl', __file__)

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
    C.POINTER(_HPGL_CONT_MASKED_ARRAY)]

_hpgl_so.hpgl_simple_kriging.restype = None
_hpgl_so.hpgl_simple_kriging.argtypes =  [
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    C.POINTER(_HPGL_SHAPE),
    C.POINTER(_HPGL_SK_PARAMS),
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    C.POINTER(_HPGL_SHAPE)]

_hpgl_so.hpgl_lvm_kriging.restype = None
_hpgl_so.hpgl_lvm_kriging.argtypes = [
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    C.POINTER(_HPGL_SHAPE),
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    C.POINTER(_HPGL_SHAPE),
    C.POINTER(_HPGL_OK_PARAMS),
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    C.POINTER(_HPGL_SHAPE)]

_hpgl_so.hpgl_simple_kriging_weights.restype = C.c_int
_hpgl_so.hpgl_simple_kriging_weights.argtypes = [
    (C.c_float * 3),
    NC.ndpointer(dtype = numpy.float32),
    NC.ndpointer(dtype = numpy.float32),
    NC.ndpointer(dtype = numpy.float32),
    C.c_int,
    C.POINTER(__hpgl_cov_params_t),
    NC.ndpointer(dtype = numpy.float32)]

_hpgl_so.hpgl_sgs_simulation.restype = None
_hpgl_so.hpgl_sgs_simulation.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_SGS_PARAMS),
    C.POINTER(hpgl_non_parametric_cdf_t),
    C.POINTER(C.c_double),
    C.POINTER(_HPGL_UBYTE_ARRAY)]

_hpgl_so.hpgl_sgs_lvm_simulation.restype = None
_hpgl_so.hpgl_sgs_lvm_simulation.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_SGS_PARAMS),
    C.POINTER(hpgl_non_parametric_cdf_t),
    C.POINTER(_HPGL_FLOAT_ARRAY),
    C.POINTER(_HPGL_UBYTE_ARRAY)]

_hpgl_so.hpgl_median_ik.restype = None
_hpgl_so.hpgl_median_ik.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_MEDIAN_IK_PARAMS),
    C.POINTER(_HPGL_IND_MASKED_ARRAY)
    ]

_hpgl_so.hpgl_indicator_kriging.restype = None
_hpgl_so.hpgl_indicator_kriging.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.c_int]

_hpgl_so.hpgl_set_thread_num.restype = C.c_int
_hpgl_so.hpgl_set_thread_num.argtypes = [C.c_int]

_hpgl_so.hpgl_get_thread_num.restype = C.c_int
_hpgl_so.hpgl_get_thread_num.argtypes = []

_hpgl_so.hpgl_read_inc_file_float.restype = C.c_int
_hpgl_so.hpgl_read_inc_file_float.argtypes = [
    C.c_char_p,
    C.c_float,
    C.c_int,
    NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A'])]

_hpgl_so.hpgl_read_inc_file_byte.restype = C.c_int
_hpgl_so.hpgl_read_inc_file_byte.argtypes = [
    C.c_char_p,
    C.c_int,
    C.c_int,
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
    C.c_int]

_hpgl_so.hpgl_write_inc_file_float.restype = C.c_int
_hpgl_so.hpgl_write_inc_file_float.argtypes = [
    C.c_char_p,
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.c_float,
    C.c_char_p
]

_hpgl_so.hpgl_write_inc_file_byte.restype = C.c_int
_hpgl_so.hpgl_write_inc_file_byte.argtypes = [
    C.c_char_p,
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.c_int,
    C.c_char_p,
    C.POINTER(C.c_ubyte),
    C.c_int]

_hpgl_so.hpgl_write_gslib_cont_property.restype = C.c_int
_hpgl_so.hpgl_write_gslib_cont_property.argtypes = [
        C.POINTER(_HPGL_CONT_MASKED_ARRAY),
        C.c_char_p,
        C.c_char_p,
        C.c_double]

_hpgl_so.hpgl_write_gslib_byte_property.restype = C.c_int
_hpgl_so.hpgl_write_gslib_byte_property.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.c_char_p,
    C.c_char_p,
    C.c_double,
    C.POINTER(C.c_ubyte),
    C.c_int]


_hpgl_so.hpgl_sis_simulation.restype = None
_hpgl_so.hpgl_sis_simulation.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.c_int,
    C.c_int64,
    C.POINTER(_HPGL_UBYTE_ARRAY)]

_hpgl_so.hpgl_sis_simulation_lvm.restype = None
_hpgl_so.hpgl_sis_simulation_lvm.argtypes = [
    C.POINTER(_HPGL_IND_MASKED_ARRAY),
    C.POINTER(_HPGL_IK_PARAMS),
    C.POINTER(_HPGL_FLOAT_ARRAY),
    C.c_int,
    C.c_int64,
    C.POINTER(_HPGL_UBYTE_ARRAY),
    C.c_int
    ]

_hpgl_so.hpgl_simple_cokriging_mark1.restype = None
_hpgl_so.hpgl_simple_cokriging_mark1.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(__hpgl_cockriging_m1_params_t),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY)]

_hpgl_so.hpgl_simple_cokriging_mark2.restype = None
_hpgl_so.hpgl_simple_cokriging_mark2.argtypes = [
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY),
    C.POINTER(__hpgl_cockriging_m2_params_t),
    C.POINTER(_HPGL_CONT_MASKED_ARRAY)]
