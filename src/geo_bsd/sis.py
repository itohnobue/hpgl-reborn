import ctypes as C

import numpy

# Import validation framework
from .geo import (
    __checked_create,
    _c_array,
    _check_hpgl_error,
    _clone_prop,
    _create_hpgl_float_array,
    _create_hpgl_ind_masked_array,
    _create_hpgl_ubyte_array,
    _empty_clone,
    _require_cont_data,
    _require_ind_data,
    _snapshot_hpgl_error,
    accepts_tuple,
)
from .hpgl_wrap import _HPGL_FLOAT_ARRAY, _HPGL_IK_PARAMS, _hpgl_so
from .validation import GridValidator, ParameterValidator


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
    if mask is not None:
        mask = _require_ind_data(mask)

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
    """Performs Sequential Indicator Simulation (SIS).

Parameters:
-----------
prop : IndProperty
    Input indicator property with data, mask, and indicator_count.
grid : SugarboxGrid
    Simulation grid defining (x, y, z) dimensions.
data : list of dict
    List of per-indicator dictionaries, each containing:
    ``{"cov_model": CovarianceModel, "radiuses": tuple, "max_neighbours": int}``.
seed : int
    Seed for the random number generator.
marginal_probs : list of float or list of numpy.ndarray
    Marginal probabilities for each indicator category. For simple (stationary)
    simulation, a list of scalar floats that sum to 1.0. For locally varying
    mean (LVM) simulation, a list of 3D float32 ndarrays matching the grid size.
use_correlogram : bool, optional
    If ``True``, use correlogram-based simulation. Only applicable in LVM mode.
    Default: ``True``.
mask : numpy.ndarray or None, optional
    3D uint8 array where ``1`` marks cells to simulate and ``0`` marks cells
    to skip. If ``None``, all cells are simulated. Default: ``None``.
force_single_thread : bool, optional
    Force single-threaded execution. Default: ``False``.
force_parallel : bool, optional
    Force parallel execution. Default: ``False``.
use_harddata : bool, optional
    If ``True``, use source data values for simulation. If ``False``,
    ignore source data values. Default: ``True``.
use_regions : bool, optional
    If ``True``, use region-based simulation partitioning. Default: ``False``.
region_size : tuple, optional
    Size of each simulation region when `use_regions=True`. Default: ``()``.
min_neighbours : int, optional
    Minimum number of neighbours required for kriging. Default: ``0``.

Returns:
--------
IndProperty
    Simulated indicator property.

Raises:
-------
CriticalValidationError
    If any parameter fails validation.
ValueError
    If marginal probabilities do not sum to 1.0 (non-LVM mode).
RuntimeError
    If the underlying C++ simulation fails."""
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
        _snapshot_hpgl_error()
        _hpgl_so.hpgl_sis_simulation(
            prop_2,
            ikps,
            len(data),
            seed,
            _create_hpgl_ubyte_array(mask, grid) if mask is not None else None)
        _check_hpgl_error("sis_simulation")
#        hpgl.sis_simulation(_prop_to_tuple(out_prop), grid.grid, data, seed, False, use_correlogram, mask)
    else:
        _snapshot_hpgl_error()
        _hpgl_so.hpgl_sis_simulation_lvm(
            prop_2,
            ikps,
            _c_array(_HPGL_FLOAT_ARRAY, len(data), means),
            len(data),
            seed,
            _create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
            use_correlogram
            )
        _check_hpgl_error("sis_simulation_lvm")
        #hpgl.sis_simulation_lvm(_prop_to_tuple(out_prop), grid.grid, data, seed, marginal_probs, use_correlogram, mask)
    return out_prop
