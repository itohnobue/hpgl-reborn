import numpy
import ctypes as C

from .geo import _clone_prop, _empty_clone, _require_cont_data, _requite_ind_data, __checked_create, _create_hpgl_ind_masked_array, _create_hpgl_ubyte_array, _create_hpgl_float_array
from .geo import _c_array

from .geo import accepts_tuple

from .hpgl_wrap import _HPGL_IK_PARAMS, _HPGL_FLOAT_ARRAY, _hpgl_so

# Import validation framework
from . import validation
from .validation import (
    GridValidator,
    ParameterValidator,
    validate_simulation_params
)

def __prepare_sis(prop, data, marginal_probs, mask, use_harddata):
	is_lvm = not numpy.isscalar(marginal_probs[0])
	if use_harddata:
		out_prop = _clone_prop(prop)
	else:
		out_prop = _empty_clone(prop)

	if is_lvm:
		marginal_probs = [_require_cont_data(m) for m in marginal_probs]
	for i in range(len(data)):
		if is_lvm:
			data[i]['marginal_prob'] = 0
		else:
			data[i]['marginal_prob'] = marginal_probs[i]
	if not mask is None:
		mask = _requite_ind_data(mask)

	return out_prop, is_lvm, marginal_probs, mask

def __create_hpgl_ik_params(data, indicator_count, is_lvm, marginal_probs):
	ikps = []
	assert len(data) == indicator_count
	for i in range(indicator_count):
		ikd = data[i]
		ikp = __checked_create(
			_HPGL_IK_PARAMS,
			covariance_type = ikd["cov_model"].type,
			ranges = (C.c_double * 3)(*ikd["cov_model"].ranges),
			angles = (C.c_double * 3)(*ikd["cov_model"].angles),
			sill = ikd["cov_model"].sill,
			nugget = ikd["cov_model"].nugget,
			radiuses = (C.c_int * 3)(*ikd["radiuses"]),
			max_neighbours = ikd["max_neighbours"],
			marginal_prob = 0 if is_lvm else marginal_probs[i])
		ikps.append(ikp)
	return _c_array(_HPGL_IK_PARAMS, indicator_count, ikps)

@accepts_tuple('prop', 0)
def sis_simulation(prop, grid, data, seed, marginal_probs, use_correlogram=True, mask=None, force_single_thread=False, force_parallel=False, use_harddata=True, use_regions=False, region_size = (), min_neighbours = 0):
	# Validate grid dimensions
	GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

	# Validate indicator count
	ParameterValidator.validate_indicator_count(len(data))

	# Validate seed
	ParameterValidator.validate_seed(seed)

	# Validate marginal probabilities
	is_lvm = not numpy.isscalar(marginal_probs[0])
	if not is_lvm:
		ParameterValidator.validate_probability_sum(marginal_probs)

	# Validate each indicator's parameters
	for i, ikd in enumerate(data):
		# Validate covariance parameters
		ParameterValidator.validate_covariance_parameters(
			ikd["cov_model"].sill,
			ikd["cov_model"].nugget,
			ikd["cov_model"].ranges,
			ikd["cov_model"].angles
		)

		# Validate radiuses
		ParameterValidator.validate_radius(ikd["radiuses"], f'ik_data[{i}].radiuses')

		# Validate max_neighbours
		ParameterValidator.validate_max_neighbors(ikd["max_neighbours"])

		# Validate marginal probability
		if not is_lvm:
			ParameterValidator.validate_probability(marginal_probs[i], f'marginal_probs[{i}]')

	out_prop, is_lvm, marginal_probs, mask = __prepare_sis(prop, data, marginal_probs, mask, use_harddata)

	# Update indicator_count to match the number of categories in data
	out_prop.indicator_count = len(data)
	prop_2 = _create_hpgl_ind_masked_array(out_prop, grid)

	ikps = __create_hpgl_ik_params(data, len(data), is_lvm, marginal_probs)

	means = []
	if is_lvm:
		for i in range(len(data)):
			means.append(_create_hpgl_float_array(marginal_probs[i], grid))

	if not is_lvm:
		_hpgl_so.hpgl_sis_simulation(
			prop_2,
			ikps,
			len(data),
			seed,
			_create_hpgl_ubyte_array(mask, grid) if mask is not None else None)
#		hpgl.sis_simulation(_prop_to_tuple(out_prop), grid.grid, data, seed, False, use_correlogram, mask)
	else:
		_hpgl_so.hpgl_sis_simulation_lvm(
			prop_2,
			ikps,
			_c_array(_HPGL_FLOAT_ARRAY, len(data), means),
			len(data),
			seed,
			_create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
			use_correlogram
			)
		#hpgl.sis_simulation_lvm(_prop_to_tuple(out_prop), grid.grid, data, seed, marginal_probs, use_correlogram, mask)
	return out_prop
