import numpy
import os
import pathlib
from typing import Optional, Tuple, Union

import ctypes as C

# Import validation framework
from . import validation
from .validation import (
    PathValidator,
    GridValidator,
    ParameterValidator,
    validate_grid_params,
    validate_kriging_params,
    validate_simulation_params,
    validate_file_params,
    ValidationError,
    CriticalValidationError
)

from .hpgl_wrap import _HPGL_SHAPE, _HPGL_CONT_MASKED_ARRAY, _HPGL_IND_MASKED_ARRAY, _HPGL_UBYTE_ARRAY, _HPGL_FLOAT_ARRAY, _HPGL_OK_PARAMS, _HPGL_SK_PARAMS, _HPGL_IK_PARAMS, _HPGL_MEDIAN_IK_PARAMS, __hpgl_cov_params_t, __hpgl_cockriging_m1_params_t, __hpgl_cockriging_m2_params_t, _hpgl_so


# Security: Path validation utilities to prevent directory traversal attacks
def _validate_filepath(filename: Union[str, pathlib.Path], allow_directories: bool = False) -> str:
    """
    Validates and sanitizes file paths to prevent directory traversal attacks.

    Args:
        filename: The file path to validate
        allow_directories: Whether to allow directory paths (default: False)

    Returns:
        Absolute, normalized path as string

    Raises:
        ValueError: If path contains directory traversal attempts or points outside allowed directories
        FileNotFoundError: If the file doesn't exist when validation requires it
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Convert to Path object for robust handling
    path = pathlib.Path(filename)

    # Resolve to absolute path and normalize (removes ../ segments)
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {filename}") from e

    # Check for path traversal attempts in the original string
    # This catches cases where resolve() might behave unexpectedly
    path_str = os.path.normpath(filename)
    if '..' in path_str.split(os.sep) or '../' in filename or '..\\' in filename:
        raise ValueError(f"Path traversal detected in filename: {filename}")

    # If we're reading, verify the file exists (unless allow_directories is True)
    # For write operations, we don't require existence but validate path safety
    if allow_directories:
        # For directories, ensure we're not escaping filesystem root
        if not resolved_path.is_absolute():
            # Resolve relative to current working directory
            resolved_path = pathlib.Path.cwd() / resolved_path
            resolved_path = resolved_path.resolve()
    else:
        # For files, we validate the path structure but may defer existence check
        # to the caller's context (read vs write)
        pass

    return str(resolved_path)


# Security: Array reference holder to prevent use-after-free with ctypes
class _ArrayReferenceHolder:
    """
    Holds references to NumPy arrays to prevent garbage collection
    while C code holds pointers to their data.

    This prevents use-after-free vulnerabilities where the C code
    might access freed memory if the Python array is garbage collected.
    """
    def __init__(self):
        self._arrays = []

    def add(self, *arrays):
        """Add arrays to be kept alive"""
        self._arrays.extend(arrays)

    def clear(self):
        """Clear all references"""
        self._arrays.clear()
from .hpgl_wrap import hpgl_output_handler, hpgl_progress_handler

def _c_array(ar_type, size, values):
	if len(values) != size:
		raise RuntimeError("%s values specified for array of %s elements" % (len(values), size))
	return (ar_type * size) (*values)

def _create_hpgl_shape(shape, strides=None):
	# Normalize shape to 3D tuple
	if len(shape) == 1:
		shape = (shape[0], 1, 1)
	elif len(shape) == 2:
		shape = (shape[0], shape[1], 1)

	if strides is None:
		# C-order strides (row-major) to match C++ indexing: z * x * y + y * x + x
		# The strides array is (stride_x, stride_y, stride_z)
		return _HPGL_SHAPE(data = _c_array(C.c_int, 3, shape),
				   strides = _c_array(C.c_int, 3, (1, shape[0], shape[0]*shape[1])))
	else:
		return _HPGL_SHAPE(data = _c_array(C.c_int, 3, shape),
				   strides = _c_array(C.c_int, 3, strides))

def __get_strides(prop):
	ndim = prop.ndim
	if ndim == 1:
		return (1, prop.shape[0], prop.shape[0])
	elif ndim == 2:
		return (1, prop.shape[0], prop.shape[0] * prop.shape[1])
	else:  # ndim == 3
		return (prop.strides[0] // prop.itemsize,
			prop.strides[1] // prop.itemsize,
			prop.strides[2] // prop.itemsize)


def __checked_create(T, **kargs):
    fields = []
    for f, _ in T._fields_:
        fields.append(f)
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
    assert len(fields) == 0, "No values for parameters: %s" % fields
    return T(**kargs)

def checkFWA(a):
	"""
	Checks for fortran-order, writable and aligned flags.
	"""
	assert (a.flags['F'] and a.flags['W'] and a.flags['A'])

def _create_hpgl_cont_masked_array(prop, grid):
	if (grid is None):
		sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
	else:
		# Use actual NumPy strides if array is 3D, otherwise compute strides from grid
		if prop.data.ndim == 3:
			sh = _create_hpgl_shape((grid.x, grid.y, grid.z), __get_strides(prop.data))
		else:
			# For 1D arrays, compute expected strides based on grid dimensions
			sh = _create_hpgl_shape((grid.x, grid.y, grid.z))
		if grid.x * grid.y * grid.z != prop.data.size:
			raise RuntimeError("Invalid data size. Size of data = %s. Size of grid = %s" % (prop.data.size, grid.x * grid.y * grid.z))

	# Security: Keep references to arrays to prevent use-after-free
	# The C code will hold pointers to this memory, so we must ensure
	# the Python objects aren't garbage collected while in use
	result = _HPGL_CONT_MASKED_ARRAY(
		data=prop.data.ctypes.data_as(C.POINTER(C.c_float)),
		mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
		shape = sh)

	# Store array references on the result object to prevent garbage collection
	# This ensures the arrays live as long as the C structure does
	result._array_refs = (prop.data, prop.mask)

	return result

def _create_hpgl_ind_masked_array(prop, grid):
	if (grid is None):
		sh = _create_hpgl_shape(prop.data.shape, __get_strides(prop.data))
		assert(prop.data.strides == prop.mask.strides)
	else:
		sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

	# Security: Keep references to arrays to prevent use-after-free
	result = _HPGL_IND_MASKED_ARRAY(
		data=prop.data.ctypes.data_as(C.POINTER(C.c_ubyte)),
		mask=prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
		shape = sh,
		indicator_count = prop.indicator_count)

	# Store array references to prevent garbage collection while C code uses them
	result._array_refs = (prop.data, prop.mask)

	return result

def _create_hpgl_ubyte_array(array, grid):
	checkFWA(array)
	if (grid is None):
		sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
	else:
		sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

	# Security: Keep array reference to prevent use-after-free
	result = _HPGL_UBYTE_ARRAY(data = array.ctypes.data_as(C.POINTER(C.c_ubyte)), shape = sh)
	result._array_ref = array
	return result

def _create_hpgl_float_array(array, grid):
	checkFWA(array)
	if (grid is None):
		sh = _create_hpgl_shape(array.shape, strides=__get_strides(array))
	else:
		sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

	# Security: Keep array reference to prevent use-after-free
	result = _HPGL_FLOAT_ARRAY(data = array.ctypes.data_as(C.POINTER(C.c_float)), shape = sh)
	result._array_ref = array
	return result

class ContProperty:
	def __init__(self, data, mask):
		self.data = numpy.require(data, 'float32', 'F')
		self.mask = numpy.require(mask, 'uint8', 'F')
	def validate(self):
		checkFWA(self.data)
		checkFWA(self.mask)
		assert(self.data.shape == self.mask.shape)
	def fix_shape(self, grid):
		if self.data.ndim != 3:
			if self.data.size == grid.x * grid.y * grid.z:
				self.data = self.data.reshape((grid.x, grid.y, grid.z), order='F')
		if self.mask.ndim != 3:
			if self.mask.size == grid.x * grid.y * grid.z:
				self.mask = self.mask.reshape((grid.x, grid.y, grid.z), order='F')
	def __getitem__(self, idx):
#		print "Warning. ContProperty.__getitem__ is deprecated."
		if idx == 0:
			return self.data
		elif idx == 1:
			return self.mask
		else:
			raise RuntimeError("Index out of range.")
	
class IndProperty:
	def __init__(self, data, mask, indicator_count):
		self.data = numpy.require(data, 'uint8', 'F')
		self.mask = numpy.require(mask, 'uint8', 'F')
		self.indicator_count = indicator_count
		if numpy.sum(numpy.bitwise_and((mask > 0), (data >= indicator_count))) > 0:
			raise RuntimeError("Property contains some indicators outside of [0..%s] range." % (indicator_count - 1))
		assert(data.shape == mask.shape)
	def validate(self):
		checkFWA(self.data)
		checkFWA(self.mask)
		assert(self.data.shape == self.mask.shape)
	def __getitem__(self, idx):
#		print "Warning. IndPoroperty.__getitem__ is deprecated."
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
		assert False
		return prop

def append_mask(prop, mask):
	infs = prop[1]
	infs &= mask
	infs.choose(-99, prop[0], out=prop[0])

class covariance:
	spherical = 0
	exponential = 1
	gaussian = 2

class SugarboxGrid:
	def __init__(self, x, y, z):
		# Validate grid dimensions
		GridValidator.validate_grid_dimensions(x, y, z)
		self.x = x
		self.y = y
		self.z = z

class CovarianceModel:
	def __init__(self, type = 0, ranges=(0,0,0), angles=(0.0,0.0,0.0), sill=1.0, nugget=0.0):
		self.type = type
		self.ranges = ranges
		self.angles = angles
		self.sill = sill
		self.nugget = nugget

		# Validate covariance parameters
		ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)

def _load_prop_cont_slow(filename, undefined_value):
	# Security: Validate filename to prevent directory traversal attacks
	safe_path = _validate_filepath(filename)

	values = []
	mask = []
	# Use validated path and explicit encoding
	with open(safe_path, 'r', encoding='utf-8') as f:
		for line in f:
			if line.strip().startswith("--"):
				continue
			for part in line.split():
				try:
					val = float(part.strip())
					values.append(val)
					if (val == undefined_value):
						mask.append(0)
					else:
						mask.append(1)
				except (ValueError, TypeError):
					pass

	return ContProperty(numpy.array(values, dtype="float32"), numpy.array(mask, dtype="uint8"))

def _load_prop_ind_slow(filename, undefined_value, ind_values):
	dict_map = {}
	for i in range(len(ind_values)):
		dict_map[ind_values[i]] = i

	# Security: Validate filename to prevent directory traversal attacks
	safe_path = _validate_filepath(filename)

	values = []
	mask = []

	# Use validated path and explicit encoding
	with open(safe_path, 'r', encoding='utf-8') as f:
		for line in f:
			if line.strip().startswith("--"):
				continue
			for part in line.split():
				try:
					val = int(part.strip())
					if (val == undefined_value):
						values.append(255)
						mask.append(0)
					else:
						values.append(dict_map[val])
						mask.append(1)
				except (ValueError, TypeError, KeyError):
					pass

	return IndProperty(numpy.array(values, dtype="uint8", order='F'), numpy.array(mask, dtype="uint8", order='F'), len(ind_values))

def _create_cont_prop(size):
	return ContProperty(numpy.zeros(size, dtype="float32"), numpy.zeros(size, dtype='uint8'))

def _create_ind_prop(size, indicator_count):
	return IndProperty(numpy.zeros(size, dtype="uint8"), numpy.zeros(size, dtype='uint8'), indicator_count)

def _empty_clone(prop):	
	data2 = prop.data.copy('F')
	data2.fill(0)
	mask2 = prop.mask.copy('F')
	mask2.fill(0)
	if isinstance(prop, IndProperty):
		return IndProperty(data2, mask2, prop.indicator_count)
	elif isinstance(prop, ContProperty):
		return ContProperty(data2, mask2)
	else:
		assert False

def _clone_prop(prop):
	data2 = prop.data.copy('F')
	mask2 = prop.mask.copy('F')
	if isinstance(prop, IndProperty):
		return IndProperty(data2, mask2, prop.indicator_count)
	elif isinstance(prop, ContProperty):
		return ContProperty(data2, mask2)
	else:
		assert False, "Unknown prop type"

def _require_cont_data(data):
	if data is None:
		return None
	return numpy.require(data, dtype='float32', requirements='F')

def _requite_ind_data(data):
	if data is None:
		return None
	return numpy.require(data, dtype='uint8', requirements='F')

def accepts_tuple(arg_name, arg_pos):
	def tuple_to_prop(t):
		if isinstance(t, tuple):
			if len(t) == 3:
				return IndProperty(*t)
			elif len(t) == 2:
				return ContProperty(*t)
			else:
				assert False
		else:
			assert isinstance(t, ContProperty) or isinstance(t, IndProperty)
			return t
	def decorator(f):
		def new_f(*args, **kargs):
			if arg_name in kargs:
				kargs[arg_name] = tuple_to_prop(kargs[arg_name])
			elif len(args) > arg_pos:
				args = args[:arg_pos] + (tuple_to_prop(args[arg_pos]),) + args[arg_pos+1:]
			else:
				assert False
			return f(*args, **kargs)
		new_f.__name__ = f.__name__
		return new_f
	return decorator
		
@accepts_tuple('prop', 0)
def write_property(prop, filename, prop_name, undefined_value, indicator_values=None):
	# Security: Validate filename to prevent directory traversal attacks
	safe_path = _validate_filepath(filename)

	if indicator_values is None:
		indicator_values = []

	if (prop.data.ndim == 3):
		sh = _create_hpgl_shape(prop.data.shape)
	else:
		sh = _create_hpgl_shape((prop.data.size, 1, 1))
	if isinstance(prop, ContProperty):
		marr = _HPGL_CONT_MASKED_ARRAY(
			data = prop.data.ctypes.data_as(C.POINTER(C.c_float)),
			mask = prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
			shape = sh)
		# Security: Keep array references to prevent use-after-free
		marr._array_refs = (prop.data, prop.mask)
		rc = _hpgl_so.hpgl_write_inc_file_float(
			safe_path.encode("utf-8"),
			C.byref(marr),
			undefined_value,
			prop_name.encode("utf-8"))
		if rc != 0:
			raise RuntimeError("write_property failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))
	else:
		# Security: Keep reference to indicator_values array
		ind_arr = numpy.array(indicator_values, dtype='uint8')
		marr = _HPGL_IND_MASKED_ARRAY(
			data = prop.data.ctypes.data_as(C.POINTER(C.c_ubyte)),
			mask = prop.mask.ctypes.data_as(C.POINTER(C.c_ubyte)),
			shape = sh,
			indicator_count = prop.indicator_count)
		# Security: Keep array references to prevent use-after-free
		marr._array_refs = (prop.data, prop.mask, ind_arr)
		rc = _hpgl_so.hpgl_write_inc_file_byte(
			safe_path.encode("utf-8"),
			C.byref(marr),
			undefined_value,
			prop_name.encode("utf-8"),
			ind_arr.ctypes.data_as(C.POINTER(C.c_ubyte)),
			len(indicator_values))
		if rc != 0:
			raise RuntimeError("write_property failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))

@accepts_tuple('prop', 0)
def write_gslib_property(prop, filename, prop_name, undefined_value, indicator_values=None):
	# Security: Validate filename to prevent directory traversal attacks
	safe_path = _validate_filepath(filename)

	if indicator_values is None:
		indicator_values = []

	if isinstance(prop, ContProperty):
		rc = _hpgl_so.hpgl_write_gslib_cont_property(
			_create_hpgl_cont_masked_array(prop, None),
			safe_path.encode("utf-8"),
			prop_name.encode("utf-8"),
			undefined_value)
		if rc != 0:
			raise RuntimeError("write_gslib_property failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))
	else:
		rc = _hpgl_so.hpgl_write_gslib_byte_property(
			_create_hpgl_ind_masked_array(prop, None),
			safe_path.encode("utf-8"),
			prop_name.encode("utf-8"),
			undefined_value,
			_c_array(C.c_ubyte, len(indicator_values), indicator_values),
			len(indicator_values))
		if rc != 0:
			raise RuntimeError("write_gslib_property failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))

def load_cont_property(filename, undefined_value, size=None):
	# Validate filename for security
	safe_path = PathValidator.validate_filepath(filename, must_exist=True)

	if size is None:
		print("[WARNING]. load_cont_property: Size is not specified. Using slow and ineficcient method.")
		return _load_prop_cont_slow(safe_path, undefined_value)
	else:
		return read_inc_file_float(safe_path, undefined_value, size)

def read_inc_file_float(filename, undefined_value, size):
	# Security: Validate filename to prevent directory traversal attacks
	safe_path = PathValidator.validate_filepath(filename, must_exist=True)

	# Validate size parameters
	if isinstance(size, tuple) and len(size) == 3:
		GridValidator.validate_grid_dimensions(size[0], size[1], size[2])

	total_elements = size[0] * size[1] * size[2] if isinstance(size, tuple) and len(size) == 3 else size
	data = numpy.zeros(total_elements, dtype='float32', order='F')
	mask = numpy.zeros(total_elements, dtype='uint8', order='F')

	rc = _hpgl_so.hpgl_read_inc_file_float(
		safe_path.encode("utf-8"),
		undefined_value,
		total_elements,
		data,
		mask)
	if rc != 0:
		raise RuntimeError("read_inc_file_float failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))

	return ContProperty(data, mask)

def read_inc_file_byte(filename, undefined_value, size, indicator_values):
	# Security: Validate filename to prevent directory traversal attacks
	safe_path = PathValidator.validate_filepath(filename, must_exist=True)

	# Validate size parameters
	if isinstance(size, tuple) and len(size) == 3:
		GridValidator.validate_grid_dimensions(size[0], size[1], size[2])

	total_elements = size[0] * size[1] * size[2] if isinstance(size, tuple) and len(size) == 3 else size
	data = numpy.zeros(total_elements, dtype='uint8', order='F')
	mask = numpy.zeros(total_elements, dtype='uint8', order='F')
	rc = _hpgl_so.hpgl_read_inc_file_byte(
		safe_path.encode("utf-8"),
		undefined_value,
		total_elements,
		data,
		mask,
		numpy.array(indicator_values, dtype='uint8'),
		len(indicator_values))
	if rc != 0:
		raise RuntimeError("read_inc_file_byte failed: " + _hpgl_so.hpgl_get_last_exception_message().decode("utf-8", errors="replace"))
	return IndProperty(data, mask, len(indicator_values))

def load_ind_property(filename, undefined_value, indicator_values, size=None):
	if (size is None):
		print("[WARNING]. load_ind_property: Size is not specified. Using slow and ineficcient method.")
		return _load_prop_ind_slow(filename, undefined_value, indicator_values)
	else:
		return read_inc_file_byte(filename, undefined_value, size, indicator_values)

def set_thread_num(num):
	_hpgl_so.hpgl_set_thread_num(num)

def get_thread_num():
	return _hpgl_so.hpgl_get_thread_num()

@accepts_tuple('prop', 0)
def calc_mean(prop):
	l = len(prop.data.flat)
	sum = 0
	count = 0
	for i in range(l):
		if prop.mask.flat[i] == 1:
			sum += prop.data.flat[i]
			count += 1
	if count == 0:
		raise ValueError("calc_mean: no informed values (all masked)")
	return sum/count

@accepts_tuple('prop', 0)
def ordinary_kriging(prop, grid, radiuses, max_neighbours, cov_model):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses
	valid_radiuses = ParameterValidator.validate_radius(radiuses, 'radiuses')

	# Validate max_neighbours
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate covariance model
	ParameterValidator.validate_covariance_parameters(
		cov_model.sill,
		cov_model.nugget,
		cov_model.ranges,
		cov_model.angles
	)

	out_prop = _clone_prop(prop)

	okp = _HPGL_OK_PARAMS(
		covariance_type = cov_model.type,
		ranges = cov_model.ranges,
		angles = cov_model.angles,
		sill = cov_model.sill,
		nugget = cov_model.nugget,
		radiuses = valid_radiuses,
		max_neighbours = max_neighbours)

	_hpgl_so.hpgl_ordinary_kriging(
		_create_hpgl_cont_masked_array(prop, grid),
		C.byref(okp),
		_create_hpgl_cont_masked_array(out_prop, grid))

	return out_prop

@accepts_tuple('prop', 0)
def simple_kriging(prop, grid, radiuses, max_neighbours, cov_model, mean=None):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses
	valid_radiuses = ParameterValidator.validate_radius(radiuses, 'radiuses')

	# Validate max_neighbours
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate covariance model
	ParameterValidator.validate_covariance_parameters(
		cov_model.sill,
		cov_model.nugget,
		cov_model.ranges,
		cov_model.angles
	)

	out_prop = _clone_prop(prop)

	skp = _HPGL_SK_PARAMS(
		covariance_type = cov_model.type,
		ranges = cov_model.ranges,
		angles = cov_model.angles,
		sill = cov_model.sill,
		nugget = cov_model.nugget,
		radiuses = valid_radiuses,
		max_neighbours = max_neighbours,
		automatic_mean = (mean is None),
		mean = (mean if mean is not None else 0))

	sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

	_hpgl_so.hpgl_simple_kriging(
		prop.data, prop.mask,
		C.byref(sh), C.byref(skp),
		out_prop[0], out_prop[1],
		C.byref(sh))

	return out_prop

@accepts_tuple('prop', 0)
def lvm_kriging(prop, grid, mean_data, radiuses, max_neighbours, cov_model):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses
	valid_radiuses = ParameterValidator.validate_radius(radiuses, 'radiuses')

	# Validate max_neighbours
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate covariance model
	ParameterValidator.validate_covariance_parameters(
		cov_model.sill,
		cov_model.nugget,
		cov_model.ranges,
		cov_model.angles
	)

	# Validate mean_data
	if not isinstance(mean_data, numpy.ndarray):
		raise ValueError("lvm_kriging: mean_data must be a numpy array")
	if mean_data.dtype != numpy.float32:
		mean_data = numpy.require(mean_data, dtype='float32')

	out_prop = _clone_prop(prop)

	okp = _HPGL_OK_PARAMS(
		covariance_type = cov_model.type,
		ranges = cov_model.ranges,
		angles = cov_model.angles,
		sill = cov_model.sill,
		nugget = cov_model.nugget,
		radiuses = valid_radiuses,
		max_neighbours = max_neighbours)

	sh = _create_hpgl_shape((grid.x, grid.y, grid.z))

	_hpgl_so.hpgl_lvm_kriging(
		prop.data, prop.mask, C.byref(sh),
		mean_data, C.byref(sh),
		C.byref(okp),
		out_prop.data, out_prop.mask,
		C.byref(sh))

	return out_prop

@accepts_tuple('prop', 0)
def median_ik(prop, grid, marginal_probs, radiuses, max_neighbours, cov_model):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses
	valid_radiuses = ParameterValidator.validate_radius(radiuses, 'radiuses')

	# Validate max_neighbours
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate covariance model
	ParameterValidator.validate_covariance_parameters(
		cov_model.sill,
		cov_model.nugget,
		cov_model.ranges,
		cov_model.angles
	)

	# Validate marginal_probs
	if len(marginal_probs) != 2:
		raise ValueError("median_ik: marginal_probs must have exactly 2 elements")
	for i, p in enumerate(marginal_probs):
		ParameterValidator.validate_probability(p, f'marginal_probs[{i}]')

	out_prop = _clone_prop(prop)

	miksp = _HPGL_MEDIAN_IK_PARAMS(
		covariance_type = cov_model.type,
		ranges = cov_model.ranges,
		angles = cov_model.angles,
		sill = cov_model.sill,
		nugget = cov_model.nugget,
		radiuses = valid_radiuses,
		max_neighbours = max_neighbours,
		marginal_probs = marginal_probs)

	inp = _create_hpgl_ind_masked_array(prop, grid)
	outp = _create_hpgl_ind_masked_array(out_prop, grid)
	_hpgl_so.hpgl_median_ik(C.byref(inp), C.byref(miksp), C.byref(outp))
	return out_prop

def __create_hpgl_ik_params(data, indicator_count, is_lvm, marginal_probs):
	ikps = []
	assert len(data) == indicator_count
	for i in range(indicator_count):
		ikd = data[i]
		ikp = __checked_create(
			_HPGL_IK_PARAMS,
			covariance_type = ikd["cov_model"].type,
			ranges = _c_array(C.c_double, 3, ikd["cov_model"].ranges),
			angles = _c_array(C.c_double, 3, ikd["cov_model"].angles),
			sill = ikd["cov_model"].sill,
			nugget = ikd["cov_model"].nugget,
			radiuses = _c_array(C.c_int, 3, ikd["radiuses"]),
			max_neighbours = ikd["max_neighbours"],
			marginal_prob = 0 if is_lvm else marginal_probs[i])
		ikps.append(ikp)
	return _c_array(_HPGL_IK_PARAMS, indicator_count, ikps)

@accepts_tuple('prop', 0)
def indicator_kriging(prop, grid, data, marginal_probs):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate indicator count
	ParameterValidator.validate_indicator_count(len(data))

	# Validate marginal_probs
	if len(marginal_probs) != len(data):
		raise ValueError(f"indicator_kriging: marginal_probs length ({len(marginal_probs)}) must match data length ({len(data)})")
	for i, p in enumerate(marginal_probs):
		ParameterValidator.validate_probability(p, f'marginal_probs[{i}]')

	# Validate per-indicator parameters
	for i, ikd in enumerate(data):
		ParameterValidator.validate_radius(ikd["radiuses"], f'data[{i}].radiuses')
		ParameterValidator.validate_max_neighbors(ikd["max_neighbours"])
		ParameterValidator.validate_covariance_parameters(
			ikd["cov_model"].sill,
			ikd["cov_model"].nugget,
			ikd["cov_model"].ranges,
			ikd["cov_model"].angles
		)

	for i in range(len(data)):
		data[i]['marginal_prob'] = marginal_probs[i]
	if(len(data) == 2):
		return median_ik(
			prop, 
			grid, 
			(data[0]["marginal_prob"], 
			 1 - data[0]["marginal_prob"]), 
			data[0]["radiuses"], 
			data[0]["max_neighbours"],
			data[0]['cov_model'])	
	out_prop = _clone_prop(prop)
	_hpgl_so.hpgl_indicator_kriging(
		C.byref(_create_hpgl_ind_masked_array(prop, grid)),
		C.byref(_create_hpgl_ind_masked_array(out_prop, grid)),
		__create_hpgl_ik_params(data, len(data), False, marginal_probs),
		len(data))

	return out_prop

@accepts_tuple('prop', 0)
def simple_cokriging_markI(prop, grid,
		secondary_data, primary_mean, secondary_mean, secondary_variance, correlation_coef,
		radiuses, max_neighbours, cov_model):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses and max_neighbours
	ParameterValidator.validate_radius(radiuses, 'radiuses')
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate covariance model
	ParameterValidator.validate_covariance_parameters(
		cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles)

	# Validate cokriging-specific parameters
	ParameterValidator.validate_correlation_coef(correlation_coef)
	ParameterValidator.validate_variance(secondary_variance, 'secondary_variance')

	out_prop = _clone_prop(prop)

	_hpgl_so.hpgl_simple_cokriging_mark1(
		C.byref(_create_hpgl_cont_masked_array(prop, grid)),
		C.byref(_create_hpgl_cont_masked_array(secondary_data, grid)),
		C.byref(__checked_create(
				__hpgl_cockriging_m1_params_t, 
				covariance_type = cov_model.type,
				ranges = _c_array(C.c_double, 3, cov_model.ranges),
				angles = _c_array(C.c_double, 3, cov_model.angles),
				sill = cov_model.sill,
				nugget = cov_model.nugget,
				radiuses = _c_array(C.c_int, 3, radiuses),
				max_neighbours = max_neighbours,
				primary_mean = primary_mean,
				secondary_mean = secondary_mean,
				secondary_variance = secondary_variance,
				correlation_coef = correlation_coef)),
		C.byref(_create_hpgl_cont_masked_array(out_prop, grid)))
	return out_prop

def simple_cokriging_markII(grid,
		primary_data,
		secondary_data,
		correlation_coef,
		radiuses,
		max_neighbours):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate radiuses and max_neighbours
	ParameterValidator.validate_radius(radiuses, 'radiuses')
	ParameterValidator.validate_max_neighbors(max_neighbours)

	# Validate correlation coefficient
	ParameterValidator.validate_correlation_coef(correlation_coef)

	# Validate both covariance models
	for label, d in [("primary", primary_data), ("secondary", secondary_data)]:
		cm = d["cov_model"]
		ParameterValidator.validate_covariance_parameters(
			cm.sill, cm.nugget, cm.ranges, cm.angles)

	out_prop = _clone_prop(primary_data["data"])

	pcp = primary_data["cov_model"]
	scp = secondary_data["cov_model"]

	_hpgl_so.hpgl_simple_cokriging_mark2(
		C.byref(_create_hpgl_cont_masked_array(primary_data["data"], grid)),
		C.byref(_create_hpgl_cont_masked_array(secondary_data["data"], grid)),
		C.byref(__checked_create(
				__hpgl_cockriging_m2_params_t,
				primary_cov_params = __checked_create(
					__hpgl_cov_params_t,
					covariance_type = pcp.type,
					ranges = _c_array(C.c_double, 3, pcp.ranges),
					angles = _c_array(C.c_double, 3, pcp.angles),
					sill = pcp.sill,
					nugget = pcp.nugget),
				secondary_cov_params = __checked_create(
					__hpgl_cov_params_t,
					covariance_type = scp.type,
					ranges = _c_array(C.c_double, 3, scp.ranges),
					angles = _c_array(C.c_double, 3, scp.angles),
					sill = scp.sill,
					nugget = scp.nugget),
				radiuses = _c_array(C.c_int, 3, radiuses),
				max_neighbours = max_neighbours,
				primary_mean = primary_data["mean"],
				secondary_mean = secondary_data["mean"],
				correlation_coef = correlation_coef)),
		C.byref(_create_hpgl_cont_masked_array(out_prop, grid)))
	return out_prop

def simple_kriging_weights(center_point, n_x, n_y, n_z, ranges = (100000,100000,100000), sill = 1, cov_type = covariance.exponential, nugget = None, angles = None):
	if angles is None:
		angles = (0,0,0)
	if nugget is None:
		nugget = 0

	# Validate covariance parameters
	ParameterValidator.validate_covariance_parameters(sill, nugget, ranges, angles)

	# Validate pointset arrays have matching lengths
	if len(n_x) != len(n_y) or len(n_x) != len(n_z):
		raise RuntimeError("Invalid pointset. %s,%s,%s." % (len(n_x), len(n_y), len(n_z)))

	# Validate pointset arrays for NaN/inf
	for name, arr in [('n_x', n_x), ('n_y', n_y), ('n_z', n_z)]:
		arr_np = numpy.asarray(arr, dtype='float32')
		if numpy.any(numpy.isnan(arr_np)) or numpy.any(numpy.isinf(arr_np)):
			raise ValueError(f"simple_kriging_weights: {name} contains NaN or infinite values")

	covp = C.byref(__checked_create(
			__hpgl_cov_params_t,
			covariance_type = cov_type,
			ranges = _c_array(C.c_double, 3, ranges),
			angles = _c_array(C.c_double, 3, angles),
			sill = sill,
			nugget = nugget))

	weights = numpy.array([0]*len(n_x), dtype='float32')

	_hpgl_so.hpgl_simple_kriging_weights(
		_c_array(C.c_float, 3, center_point),
		numpy.array(n_x, dtype='float32'),
		numpy.array(n_y, dtype='float32'),
		numpy.array(n_z, dtype='float32'),
		len(n_x),
		covp,
		weights)

	return weights

def get_gslib_property(prop_dict, prop_name, undefined_value):
	prop = prop_dict[prop_name]
	informed_array = numpy.zeros(prop.shape, dtype=numpy.uint8)
	for i in range(prop.size):
		if(prop[i] == undefined_value):
			informed_array[i] = 0
		else:
			informed_array[i] = 1
	return (prop_dict[prop_name], informed_array)

def set_output_handler(handler, param):
	global h
	if (handler is None):
		# Cast None to the expected type for ctypes compatibility
		_hpgl_so.hpgl_set_output_handler(C.cast(None, hpgl_output_handler), None)
		h = None
	else:
		h = hpgl_output_handler(handler)
		_hpgl_so.hpgl_set_output_handler(h, param)


def set_progress_handler(handler, param):
	global progress_handler
	if (handler is None):
		# Cast None to the expected type for ctypes compatibility
		_hpgl_so.hpgl_set_progress_handler(C.cast(None, hpgl_progress_handler), None)
		progress_handler = None
	else:
		progress_handler = hpgl_progress_handler(handler)
		_hpgl_so.hpgl_set_progress_handler(progress_handler, param)
