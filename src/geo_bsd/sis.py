# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import numpy

# Import validation framework
from .config import SISConfig
from .ffi_adapter import (
    _HPGL_FLOAT_ARRAY,
    _c_array,
    call_sis_simulation,
    call_sis_simulation_lvm,
    create_ik_params,
)
from .ffi_adapter import (
    create_float_array as _create_hpgl_float_array,
)
from .ffi_adapter import (
    create_ind_masked_array as _create_hpgl_ind_masked_array,
)
from .ffi_adapter import (
    create_ubyte_array as _create_hpgl_ubyte_array,
)
from .geo import (
    IndProperty,
    _clone_prop,
    _empty_clone,
    _require_cont_data,
    _require_ind_data,
    accepts_tuple,
)
from .validation import (
    GridValidator,
    ParameterValidator,
    ValidationConstants,
)


def __prepare_sis(prop, data, marginal_probs, mask, use_harddata):
    is_lvm = not numpy.isscalar(marginal_probs[0])
    if use_harddata:
        out_prop = _clone_prop(prop)
    else:
        out_prop = _empty_clone(prop)

    if is_lvm:
        marginal_probs = [_require_cont_data(m) for m in marginal_probs]
    if mask is not None:
        mask = _require_ind_data(mask)

    return out_prop, is_lvm, marginal_probs, mask


def __create_hpgl_ik_params(data, indicator_count, is_lvm, marginal_probs):
    """Create IK params array (delegates to consolidated adapter helper)."""
    return create_ik_params(data, indicator_count, is_lvm, marginal_probs)


@accepts_tuple("prop", 0)
def sis_simulation(
    prop,
    grid,
    data,
    seed,
    marginal_probs,
    use_correlogram=True,
    mask=None,
    use_harddata=True,
    config=None,
    **params,
):
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
    use_harddata : bool, optional
        If ``True``, use source data values for simulation. If ``False``,
        ignore source data values. Default: ``True``.
    config : SISConfig or None, optional
        Pre-configured SIS parameters as a frozen dataclass.  When provided,
        its values override the corresponding keyword arguments above.
        Default: ``None``.

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
    # Raise on unexpected keyword arguments to catch parameter name typos
    if params:
        raise TypeError(
            f"sis_simulation() got unexpected keyword arguments: {', '.join(sorted(params.keys()))}"
        )

    # When config is provided, override parameter values from config
    if config is not None:
        if not isinstance(config, SISConfig):
            raise TypeError(
                f"sis_simulation: config must be SISConfig, got {type(config).__name__}"
            )
        seed = config.seed
        use_correlogram = config.use_correlogram
        use_harddata = config.use_harddata
        if config.marginal_probs is not None:
            marginal_probs = list(config.marginal_probs)
        # Apply config radiuses / max_neighbours to each indicator's data dict
        # only if the dict doesn't already specify its own value
        for ikd in data:
            if "radiuses" not in ikd:
                ikd["radiuses"] = config.radiuses
            if "max_neighbours" not in ikd:
                ikd["max_neighbours"] = config.max_neighbours

    # Validate grid dimensions
    GridValidator.validate_grid_dimensions(grid.x, grid.y, grid.z)

    # Validate indicator count
    ParameterValidator.validate_indicator_count(len(data))

    # Validate seed
    ParameterValidator.validate_seed(seed)

    # Validate marginal probabilities
    if len(marginal_probs) == 0:
        raise ValueError("sis_simulation: marginal_probs must not be empty")
    if len(marginal_probs) != len(data):
        raise ValueError(
            f"sis_simulation: marginal_probs length ({len(marginal_probs)}) "
            f"must match data length ({len(data)})"
        )
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
            ikd["cov_model"].angles,
        )

        # Validate radiuses
        ParameterValidator.validate_radius(ikd["radiuses"], f"ik_data[{i}].radiuses")

        # Validate max_neighbours
        ParameterValidator.validate_max_neighbors(ikd["max_neighbours"])

        # Validate marginal probability
        if not is_lvm:
            ParameterValidator.validate_probability(marginal_probs[i], f"marginal_probs[{i}]")

    out_prop, is_lvm, marginal_probs, mask = __prepare_sis(
        prop, data, marginal_probs, mask, use_harddata
    )

    ParameterValidator.validate_property_type(out_prop, IndProperty, "sis_simulation")

    # Validate prop.data for NaN/Inf before C++ call (defensive consistency).
    # IndProperty uses uint8 data which cannot hold NaN/Inf natively,
    # but the guard matches the pattern used by all kriging functions
    # and future-proofs against potential dtype changes.
    if not numpy.all(numpy.isfinite(
        numpy.asarray(prop.data, dtype=numpy.float32)
    )):
        raise ValueError(
            "sis_simulation: prop.data contains NaN or Inf values"
        )

    # Update indicator_count to match the number of categories in data
    out_prop.indicator_count = len(data)
    prop_2 = _create_hpgl_ind_masked_array(out_prop, grid)

    ikps = __create_hpgl_ik_params(data, len(data), is_lvm, marginal_probs)

    means = []
    if is_lvm:
        # Per-cell probability sum validation: in LVM mode, each cell's
        # indicator probabilities must sum to approximately 1.0. The global
        # validate_probability_sum check is intentionally skipped for LVM
        # arrays (each cell has its own distribution), so we validate
        # per-cell sums here with floating-point tolerance.
        lvm_prob_sum = numpy.sum(marginal_probs, axis=0)
        if not numpy.allclose(lvm_prob_sum, 1.0, atol=ValidationConstants.PROBABILITY_SUM_TOLERANCE):
            max_dev = numpy.max(numpy.abs(lvm_prob_sum - 1.0))
            raise ValueError(
                f"sis_simulation: LVM per-cell probability sum deviates from 1.0. "
                f"Max deviation: {max_dev:.6e}, tolerance: {ValidationConstants.PROBABILITY_SUM_TOLERANCE}. "
                f"Some cells have Σ P_i(cell) ≠ 1.0"
            )
        # Per-cell probability range validation: each cell's probability
        # for each indicator must be in [0, 1]. The sum check above ensures
        # cells sum to ~1.0 but cannot catch cells like [1.5, -0.5] = 1.0.
        for i in range(len(marginal_probs)):
            if numpy.any(marginal_probs[i] < 0) or numpy.any(marginal_probs[i] > 1):
                raise ValueError(
                    f"sis_simulation: LVM per-cell probabilities must be in [0, 1]. "
                    f"marginal_probs[{i}] contains values outside this range."
                )
        for i in range(len(data)):
            GridValidator.validate_array_size(marginal_probs[i], (grid.x, grid.y, grid.z))
            if not numpy.all(numpy.isfinite(marginal_probs[i])):
                raise ValueError(
                    f"sis_simulation: LVM marginal_probs[{i}] "
                    f"contains NaN or Inf values"
                )
            means.append(_create_hpgl_float_array(marginal_probs[i], grid))

    if not is_lvm:
        call_sis_simulation(
            prop_2,
            ikps,
            len(data),
            seed,
            _create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
        )
    else:
        call_sis_simulation_lvm(
            prop_2,
            ikps,
            _c_array(_HPGL_FLOAT_ARRAY, len(data), means),
            len(data),
            seed,
            _create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
            use_correlogram,
        )

    # Validate output data for NaN/Inf after C++ computation.
    # C++ simulation can return NaN/Inf from degenerate matrices,
    # zero neighbours, or division by zero.
    # All 7 kriging functions have this check; SGS/SIS must match.
    if not numpy.all(numpy.isfinite(
        numpy.asarray(out_prop.data, dtype=numpy.float32)
    )):
        raise RuntimeError(
            "sis_simulation: output data contains NaN or Inf after C++ computation"
        )

    return out_prop
