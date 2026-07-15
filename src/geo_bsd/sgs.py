# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import ctypes as C

import numpy

# Import validation framework
from .cdf import CdfData
from .geo import (
    CovarianceModel,
    __checked_create,
    _clone_prop,
    _create_hpgl_cont_masked_array,
    _create_hpgl_float_array,
    _create_hpgl_ubyte_array,
    _empty_clone,
    _hpgl_error_guard,
    _require_cont_data,
    _require_ind_data,
    accepts_tuple,
)
from .hpgl_wrap import _HPGL_KRIGING_KIND, _HPGL_SGS_PARAMS, _hpgl_so, hpgl_non_parametric_cdf_t
from .validation import GridValidator, ParameterValidator


def __prepare_sgs(prop, mean=None, use_harddata=True, mask=None):
    if use_harddata:
        out_prop = _clone_prop(prop)
    else:
        out_prop = _empty_clone(prop)
    if mean is not None and not numpy.isscalar(mean):
        mean = _require_cont_data(mean)
    if mask is not None:
        mask = _require_ind_data(mask)
    return out_prop, mean, mask


def _create_hpgl_nonparam_cdf(cdf_data):
    cd2 = cdf_data
    if not isinstance(cdf_data, CdfData):
        raise TypeError(
            f"_create_hpgl_nonparam_cdf: expected CdfData, got {type(cdf_data).__name__}"
        )
    result = __checked_create(
        hpgl_non_parametric_cdf_t,
        values=cd2.values.ctypes.data_as(C.POINTER(C.c_float)),
        probs=cd2.probs.ctypes.data_as(C.POINTER(C.c_float)),
        size=cd2.values.size,
    )
    # Preserve references to numpy arrays to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = (cd2.values, cd2.probs)
    return result


def normed_cov_model(cov_model):
    coef = cov_model.sill
    if coef == 0.0:
        raise ValueError("normed_cov_model: sill cannot be zero (division by zero)")
    return CovarianceModel(
        cov_model.type,
        cov_model.ranges,
        cov_model.angles,
        cov_model.sill / coef,
        cov_model.nugget / coef,
    )


@accepts_tuple("prop", 0)
def sgs_simulation(
    prop,
    grid,
    cdf_data,
    radiuses,
    max_neighbours,
    cov_model,
    seed,
    kriging_type="sk",
    mean=None,
    use_harddata=True,
    mask=None,
    min_neighbours=0,
    **params,
):
    """Performs Sequential Gaussian Simulation (SGS).

    Parameters:
    -----------
    prop : ContProperty
        Input continuous property with data and mask arrays.
    grid : SugarboxGrid
        Simulation grid defining (x, y, z) dimensions.
    cdf_data : CdfData or None
        Pre-computed cumulative distribution data for normal-score transform.
        The underlying `_create_hpgl_nonparam_cdf` asserts CdfData type.
        If None, no CDF transformation is performed.
    radiuses : tuple of (int, int, int)
        Search radiuses in (X, Y, Z) directions.
    max_neighbours : int
        Maximum number of neighbour points to use for kriging.
    cov_model : CovarianceModel
        Covariance model (type, ranges, angles, sill, nugget).
    seed : int
        Seed for the random number generator.
    kriging_type : str, optional
        Kriging method: ``"sk"`` for Simple Kriging or ``"ok"`` for Ordinary Kriging.
        Default: ``"sk"``.
    mean : None, float, or numpy.ndarray, optional
        Stationary mean value. If ``None``, the mean is calculated automatically
        from source data. If a non-scalar ndarray, it is used as a locally
        varying mean (LVM). Default: ``None``.
    use_harddata : bool, optional
        If ``True``, use source data values for simulation. If ``False``,
        ignore source data values. Default: ``True``.
    mask : numpy.ndarray or None, optional
        3D array where ``1`` marks cells to simulate and ``0`` marks cells to skip.
        If ``None``, all cells are simulated. Default: ``None``.
    min_neighbours : int, optional
        Minimum number of neighbours required for kriging. Default: ``0``.

    Returns:
    --------
    ContProperty
        Simulated continuous property.

    Raises:
    -------
    CriticalValidationError
        If any parameter fails validation."""
    # Raise on unexpected keyword arguments to catch parameter name typos
    if params:
        raise TypeError(
            f"sgs_simulation() got unexpected keyword arguments: {', '.join(sorted(params.keys()))}"
        )

    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate radiuses - convert to int for ctypes compatibility
    valid_radiuses = ParameterValidator.validate_radius(radiuses, "radiuses")
    # Ensure radiuses are integers for ctypes (c_int * 3)
    valid_radiuses = tuple(int(r) for r in valid_radiuses)

    # Validate max_neighbours
    ParameterValidator.validate_max_neighbors(max_neighbours)

    # Validate covariance model
    ParameterValidator.validate_covariance_parameters(
        cov_model.sill, cov_model.nugget, cov_model.ranges, cov_model.angles
    )

    # Validate seed
    ParameterValidator.validate_seed(seed)

    # Validate min_neighbours
    ParameterValidator.validate_min_neighbors(min_neighbours, max_neighbours)

    prop.fix_shape(grid)
    cov_model = normed_cov_model(cov_model)

    out_prop, mean, mask = __prepare_sgs(prop=prop, mean=mean, use_harddata=use_harddata, mask=mask)

    kriging_kind_map = {"sk": _HPGL_KRIGING_KIND.simple, "ok": _HPGL_KRIGING_KIND.ordinary}
    if kriging_type not in kriging_kind_map:
        raise ValueError(
            f"sgs_simulation: invalid kriging_type '{kriging_type}'. "
            f"Choose from: {', '.join(sorted(kriging_kind_map.keys()))}"
        )
    sgsp = _HPGL_SGS_PARAMS(
        covariance_type=cov_model.type,
        ranges=cov_model.ranges,
        angles=cov_model.angles,
        sill=cov_model.sill,
        nugget=cov_model.nugget,
        radiuses=valid_radiuses,
        max_neighbours=max_neighbours,
        kriging_kind=kriging_kind_map[kriging_type],
        seed=seed,
        min_neighbours=min_neighbours,
    )

    if cdf_data is None:
        hpgl_cdf = None
    else:
        _cdf_struct = _create_hpgl_nonparam_cdf(cdf_data)
        hpgl_cdf = C.byref(_cdf_struct)

    if mask is not None:
        _mask_struct = _create_hpgl_ubyte_array(mask, grid)
        hpgl_mask = C.byref(_mask_struct)
    else:
        hpgl_mask = None

    if mean is None or numpy.isscalar(mean):
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        _c_mean = C.c_double(mean) if mean is not None else None
        with _hpgl_error_guard("sgs_simulation"):
            _hpgl_so.hpgl_sgs_simulation(
                C.byref(_cont_marr),
                C.byref(sgsp),
                hpgl_cdf,
                C.byref(_c_mean) if _c_mean is not None else None,
                hpgl_mask,
            )

    else:
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        GridValidator.validate_array_size(mean, (grid.x, grid.y, grid.z))
        _float_arr = _create_hpgl_float_array(mean, grid)
        with _hpgl_error_guard("sgs_lvm_simulation"):
            _hpgl_so.hpgl_sgs_lvm_simulation(
                C.byref(_cont_marr), C.byref(sgsp), hpgl_cdf, C.byref(_float_arr), hpgl_mask
            )

    return out_prop
