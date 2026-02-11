import os
import ctypes as C
import numpy

# NumPy 2.0+ compatibility: try new location first, fall back to old
try:
    from numpy import ctypeslib as NC
    # NumPy 2.0+ changed load_library signature
    # Old: load_library(libname, path)
    # New: load_library(filepath, loader_path=None)
    import numpy as np
    if tuple(map(int, np.__version__.split('.')[:2])) >= (2, 0):
        # NumPy 2.0+: use direct ctypes.CDLL with full path
        _load_lib_func = lambda libpath: C.CDLL(str(libpath))
    else:
        # NumPy < 2.0: use the original load_library
        _load_lib_func = lambda libpath: NC.load_library(libpath)
except ImportError:
    from numpy import _ctypeslib as NC
    import numpy as np
    if tuple(map(int, np.__version__.split('.')[:2])) >= (2, 0):
        _load_lib_func = lambda libpath: C.CDLL(str(libpath))
    else:
        _load_lib_func = lambda libpath: NC.load_library(libpath)

# In NumPy 2.0+, ndpointer might be in different location
try:
    ndpointer = NC.ndpointer
except AttributeError:
    from numpy.ctypeslib import ndpointer

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
	_fields_ = [("data", C.c_int * 3 ),
		    ("strides", C.c_int * 3)]

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
		("seed", C.c_long),
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

class __hpgl_cockriging_m2_params_t(C.Structure):
	_fields_ = [
		("primary_cov_params", globals()["__hpgl_cov_params_t"]),
		("secondary_cov_params", globals()["__hpgl_cov_params_t"]),
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
	import pathlib

	# Validate the reference file path
	if not ref_file:
		raise ValueError("Reference file path cannot be empty")

	# Convert to absolute path and normalize
	ref_path = pathlib.Path(ref_file).resolve()

	# Check for path traversal in the reference path itself
	ref_str = str(ref_path)
	if '..' in ref_str.split(os.sep):
		# This is OK after resolve() as long as it's within allowed directories
		pass

	# The library should be in the same directory as the reference file
	lib_dir = ref_path.parent

	# Try platform-specific library names
	import sys
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
				return _load_lib_func(str(resolved_lib))
			except ValueError:
				# Library path escapes allowed directory
				raise ValueError(
					f"Library path {resolved_lib} is outside allowed directory {lib_dir}"
				)

	# If not found, try the original load_library behavior as fallback
	# but wrap it with additional validation
	try:
		lib = _load_lib_func(os.path.join(str(ref_path.parent), lib_name))
		# Verify the loaded library path is safe
		if hasattr(lib, '_name'):
			loaded_path = pathlib.Path(lib._name)
			if loaded_path.exists():
				resolved = loaded_path.resolve()
				try:
					resolved.relative_to(lib_dir)
				except ValueError:
					raise ValueError(
						f"Loaded library {resolved} is outside allowed directory {lib_dir}"
					)
		return lib
	except OSError as e:
		# Library not found or cannot be loaded
		lib_dirs_str = ', '.join(str(p.parent) for p in lib_paths)
		raise OSError(
			f"Cannot load library '{lib_name}'. Searched in: {lib_dirs_str}. "
			f"Original error: {e}"
		) from e

if 'HPGL_DEBUG' in os.environ:
	_hpgl_so = _safe_load_library('hpgl_d', __file__)
else:
	_hpgl_so = _safe_load_library('hpgl', __file__)

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

_hpgl_so.hpgl_set_thread_num.restype = None
_hpgl_so.hpgl_set_thread_num.argtypes = [C.c_int]

_hpgl_so.hpgl_get_thread_num.restype = C.c_int
_hpgl_so.hpgl_get_thread_num.argtypes = []

_hpgl_so.hpgl_read_inc_file_float.restype = None
_hpgl_so.hpgl_read_inc_file_float.argtypes = [
	C.c_char_p,
	C.c_float,
	C.c_int,
	NC.ndpointer(dtype = numpy.float32, flags=['F', 'W', 'A']),
	NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A'])]

_hpgl_so.hpgl_read_inc_file_byte.restype = None
_hpgl_so.hpgl_read_inc_file_byte.argtypes = [
	C.c_char_p,
	C.c_int,
	C.c_int,
	NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
	NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
	NC.ndpointer(dtype = numpy.ubyte, flags=['F', 'W', 'A']),
	C.c_int]

_hpgl_so.hpgl_write_inc_file_float.restype = None
_hpgl_so.hpgl_write_inc_file_float.argtypes = [
	C.c_char_p,
	C.POINTER(_HPGL_CONT_MASKED_ARRAY),
	C.c_float,
	C.c_char_p
]

_hpgl_so.hpgl_write_inc_file_byte.restype = None
_hpgl_so.hpgl_write_inc_file_byte.argtypes = [
	C.c_char_p,
	C.POINTER(_HPGL_IND_MASKED_ARRAY),
	C.c_int,
	C.c_char_p,
	C.POINTER(C.c_ubyte),
	C.c_int]

_hpgl_so.hpgl_write_gslib_cont_property.restype = None
_hpgl_so.hpgl_write_gslib_cont_property.argtypes = [
		C.POINTER(_HPGL_CONT_MASKED_ARRAY),
		C.c_char_p,
		C.c_char_p,
		C.c_double]

_hpgl_so.hpgl_write_gslib_byte_property.restype = None
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
	C.c_int,
	C.POINTER(_HPGL_UBYTE_ARRAY)]

_hpgl_so.hpgl_sis_simulation_lvm.restype = None
_hpgl_so.hpgl_sis_simulation_lvm.argtypes = [
	C.POINTER(_HPGL_IND_MASKED_ARRAY),
	C.POINTER(_HPGL_IK_PARAMS),
	C.POINTER(_HPGL_FLOAT_ARRAY),
	C.c_int,
	C.c_int,
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
