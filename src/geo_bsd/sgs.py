# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import numpy

# Import validation framework
from .config import SGSConfig
from .ffi_adapter import (
    _HPGL_KRIGING_KIND,
    _HPGL_SGS_PARAMS,
    call_sgs_lvm_simulation,
    call_sgs_simulation,
)
from .ffi_adapter import (
    create_cont_masked_array as _create_hpgl_cont_masked_array,
)
from .ffi_adapter import (
    create_float_array as _create_hpgl_float_array,
)
from .ffi_adapter import (
    create_ubyte_array as _create_hpgl_ubyte_array,
)
from .geo import (
    ContProperty,
    CovarianceModel,
    _clone_prop,
    _empty_clone,
    _require_cont_data,
    _require_ind_data,
    accepts_tuple,
)
from .validation import (
    GridValidator,
    ParameterValidator,
    validate_kriging_params,
    validate_simulation_params,
)


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
    """Create hpgl_non_parametric_cdf_t from CdfData (delegates to adapter)."""
    from .ffi_adapter import create_nonparam_cdf

    return create_nonparam_cdf(cdf_data)


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
    config=None,
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
    config : SGSConfig or None, optional
        Pre-configured SGS parameters as a frozen dataclass.  When provided,
        its values override the corresponding keyword arguments above.
        Default: ``None``.

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

    # When config is provided, override parameter values from config
    if config is not None:
        if not isinstance(config, SGSConfig):
            raise TypeError(
                f"sgs_simulation: config must be SGSConfig, got {type(config).__name__}"
            )
        kriging_type = config.kriging_type
        seed = config.seed
        min_neighbours = config.min_neighbours
        max_neighbours = config.max_neighbours
        radiuses = config.radiuses
        use_harddata = config.use_harddata

    # Validate grid dimensions, radiuses, max_neighbours, covariance
    valid_radiuses = validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model
    )
    # Ensure radiuses are integers for ctypes (c_int * 3)
    valid_radiuses = tuple(int(r) for r in valid_radiuses)

    # Validate simulation-specific parameters
    validate_simulation_params(seed=seed, min_neighbours=min_neighbours, max_neighbours=max_neighbours)

    ParameterValidator.validate_property_type(prop, ContProperty, "sgs_simulation")

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
        _cdf_struct = None
    else:
        _cdf_struct = _create_hpgl_nonparam_cdf(cdf_data)

    # Validate prop.data for NaN/Inf before C++ call.
    # The ContProperty.data setter does NOT validate isfinite (only
    # __init__ does), so NaN/Inf can reach this point via reassignment.
    # All 7 kriging functions have this guard; SGS/SIS must match.
    if not numpy.all(numpy.isfinite(prop.data)):
        raise ValueError(
            "sgs_simulation: prop.data contains NaN or Inf values"
        )

    if mask is not None:
        _mask_struct = _create_hpgl_ubyte_array(mask, grid)
    else:
        _mask_struct = None

    if mean is None or numpy.isscalar(mean):
        if mean is not None:
            ParameterValidator.validate_scalar_mean(mean, "sgs_simulation")
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        call_sgs_simulation(_cont_marr, sgsp, _cdf_struct, mean, _mask_struct)

    else:
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        GridValidator.validate_array_size(mean, (grid.x, grid.y, grid.z))
        if not numpy.all(numpy.isfinite(mean)):
            raise ValueError(
                "sgs_simulation: LVM mean array contains NaN or Inf values"
            )
        _float_arr = _create_hpgl_float_array(mean, grid)
        call_sgs_lvm_simulation(_cont_marr, sgsp, _cdf_struct, _float_arr, _mask_struct)

    # Validate output data for NaN/Inf after C++ computation.
    # C++ simulation can return NaN/Inf from degenerate matrices,
    # zero neighbours, or division by zero.
    # All 7 kriging functions have this check; SGS/SIS must match.
    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "sgs_simulation: output data contains NaN or Inf after C++ computation"
        )

    return out_prop
