# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import numpy

# Import validation framework
from . import geo as _geo_module
from .config import SISConfig
from .ffi_adapter import (
    _HPGL_FLOAT_ARRAY,
    _c_array,
    _hpgl_call_lock,
    call_sis_simulation,
    call_sis_simulation_lvm,
    create_ik_params,
    normalize_mask_binary,
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

# Simulation failure statistics ARE now populated by the C++ SIS path
# (F-M6: sequential_indicator_simulation.cpp calls set_kriging_stats with
# the kriging outcome counters). The Python wrapper consumes them via
# geo._finalize_kriging_stats (populate + raise on failure counters)
# under _hpgl_call_lock, mirroring the continuous-kriging wrappers.


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
        3D uint8 array where non-zero marks cells to simulate and ``0`` marks
        cells to skip (the library-wide "non-zero = informed" mask contract,
        got-20260803180153 — non-binary values like 2 are normalized to 1 at
        the boundary). If ``None``, all cells are simulated. Default: ``None``.
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
    # Reset the documented kriging-stats inspection point and populate from
    # the C++ SIS stats under _hpgl_call_lock (F-M6/F-N12) — see the module
    # comment. The reset must run INSIDE the lock so each call's sentinel
    # write is atomic with respect to other kriging/simulation threads.

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
        # M-25: never mutate the caller's data dicts. The injection below
        # adds radiuses/max_neighbours entries the caller did not provide;
        # writing them into the caller-owned dicts silently leaves stale
        # config values behind for dict reuse. Inject into shallow copies
        # (cov_model references are shared read-only).
        data = [dict(ikd) for ikd in data]
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
        # E-M9: SIS simulation allows max_neighbours=0 — the C++ contract
        # documents 0 as "unconditional simulation" (api.cpp:144-167,
        # validate_max_neighbours_or_throw accepts 0; kriging entries use
        # a separate >=1 gate).
        ParameterValidator.validate_max_neighbors(
            ikd["max_neighbours"], min_neighbors=0
        )

        # Validate marginal probability
        if not is_lvm:
            ParameterValidator.validate_probability(marginal_probs[i], f"marginal_probs[{i}]")

    out_prop, is_lvm, marginal_probs, mask = __prepare_sis(
        prop, data, marginal_probs, mask, use_harddata
    )

    # Mask semantics (got-20260803180153): the library-wide contract is
    # "non-zero = informed". The C++ SIS kernel gates simulation on
    # mask[node] == 1 (sequential_indicator_simulation.cpp:114) while the
    # Python expected-cell count counts mask != 0 — a mask value like 2 is
    # counted as "simulate" in Python but silently skipped by C++. The
    # centralized normalization converts any non-zero mask to binary 1 at
    # the boundary so both sides agree on the SAME cell set.
    mask = normalize_mask_binary(mask, "sis_simulation")

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

    # Validate indicator_count against the clone's configured value instead
    # of silently overriding it. The number of categories in data must match
    # prop.indicator_count — a mismatch would create an inconsistent output
    # contract (lost categories) between the property struct passed to C++
    # and the IK params array.
    if len(data) != out_prop.indicator_count:
        raise ValueError(
            f"sis_simulation: len(data) ({len(data)}) does not match "
            f"prop.indicator_count ({out_prop.indicator_count})"
        )
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
            # II-18: equal-volume per-dimension shape mismatch on an LVM
            # marginal-probability field silently permutes the probability
            # field — the C++ LVM provider consumes the buffer by flat node
            # index, so a (2,2,2) marginal_probs[i] on a (1,2,4) grid (both
            # volume 8) is misread with no exception. Mirror the lvm_kriging
            # R-13 guard (geo.py:1919-1927). 1D (flat) vectors are covered
            # by the size check and carry no per-dim meaning.
            # E2-01/R-09: the guard above is 3D-only — an equal-volume 2D
            # marginal_probs[i] (e.g. (2,2) on a (1,2,2) grid) sails past
            # the size check and the ndim==3 guard, then the F-order flat
            # layout of the 2D array is consumed by flat node index with a
            # silently permuted probability field — exactly the defect class
            # fixed for lvm_kriging (geo.py:1971-1982). Only 1D flat and
            # exactly grid-shaped 3D arrays are unambiguous; reject every
            # other ndim for the same reason.
            if marginal_probs[i].ndim not in (1, 3):
                raise ValueError(
                    f"sis_simulation: LVM marginal_probs[{i}] must be 1D flat "
                    f"(size {grid.x * grid.y * grid.z}) or 3D with shape "
                    f"({grid.x}, {grid.y}, {grid.z}), got "
                    f"{marginal_probs[i].ndim}D shape {marginal_probs[i].shape}"
                )
            if marginal_probs[i].ndim == 3 and (
                marginal_probs[i].shape[0] != grid.x
                or marginal_probs[i].shape[1] != grid.y
                or marginal_probs[i].shape[2] != grid.z
            ):
                raise ValueError(
                    f"sis_simulation: 3D LVM marginal_probs[{i}] shape "
                    f"{marginal_probs[i].shape} does not match grid "
                    f"dimensions ({grid.x}, {grid.y}, {grid.z})"
                )
            if not numpy.all(numpy.isfinite(marginal_probs[i])):
                raise ValueError(
                    f"sis_simulation: LVM marginal_probs[{i}] "
                    f"contains NaN or Inf values"
                )
            means.append(_create_hpgl_float_array(marginal_probs[i], grid))

    # Expected number of kriging evaluations the simulation should perform.
    # The C++ SIS counting differs by branch (sequential_indicator_simulation.cpp):
    # the 2-category median-SIS branch does ONE kriging evaluation per node
    # on indicator 0 only (:122-155) — indicator 1's neighbourhood parameters
    # are never consulted (nblookups[1] is constructed at :56 but unused in
    # that branch); the 3+-category branch does one evaluation per indicator
    # category per node (:158-188). points_calculated is a SINGLE shared
    # counter for the whole run (:76), incremented once per KI_SUCCESS
    # evaluation (:140/:175). The expected count must mirror that per-branch
    # counting: (uninformed, unmasked cells) × evaluations_per_node.
    # Computed before the lock (cheap numpy sum) — keeps the lock hold minimal;
    # the C++ SIS counters decide whether the failure warning fires.
    # E-M9/R-20: an indicator with max_neighbours=0 is in the C++-documented
    # "unconditional simulation" mode — its neighbourhood is empty for every
    # node (sugarbox_neighbour_lookup.h:107-108, m_max_neighbours <= 0), every
    # one of its evaluations takes the KI_NO_NEIGHBOURS marginal-probability-
    # substitution path (kriging_interpolation.h:581-582) and contributes ZERO
    # to points_calculated (:141/:176 — KI_NO_NEIGHBOURS goes to
    # points_without_neighbours, not points_calculated). The exemption is
    # therefore BRANCH-AWARE: expected=0 suppresses the spurious warning only
    # for the evaluations that are genuinely unconditional, while the
    # conditional indicators' expected successes stay visible so their real
    # under-kriging still warns. (any()/all() over the indicators are both
    # wrong: any() hides genuine failures in mixed configs; all() spuriously
    # warns in the 2-category branch, which kriges only indicator 0.)
    # Genuine singular failures still raise via _finalize_kriging_stats
    # (points_singularity > 0, independent of expected).
    uninformed = grid.x * grid.y * grid.z - int(numpy.sum(out_prop.mask > 0))
    if mask is not None:
        uninformed = int(numpy.sum((out_prop.mask.ravel() == 0) & (mask.ravel() != 0)))
    if len(data) == 2:
        # Median-SIS branch: only indicator 0 is evaluated (one KI eval per
        # node). Unconditional indicator 0 → every node KI_NO_NEIGHBOURS →
        # points_calculated stays 0 → expected=0 (no warning; the marginal
        # draw IS the requested mode). Conditional indicator 0 → one success
        # per node is genuinely expected; shortfalls still warn. Indicator 1's
        # max_neighbours is irrelevant in this branch (never consulted).
        if data[0]["max_neighbours"] == 0:
            expected = 0
        else:
            expected = uninformed
    else:
        # 3+-category branch: one KI eval per indicator per node. An
        # unconditional indicator contributes KI_NO_NEIGHBOURS (zero
        # successes); conditional indicators contribute their real successes.
        # Expected = uninformed × (number of conditional indicators).
        # All-unconditional → expected=0 → no warning (requested mode).
        # Mixed → the conditional indicators' genuine failures still fire
        # the warning.
        expected = uninformed * sum(
            1 for ikd in data if ikd["max_neighbours"] > 0
        )

    if not is_lvm:
        with _hpgl_call_lock:
            _geo_module._reset_kriging_stats()
            call_sis_simulation(
                prop_2,
                ikps,
                len(data),
                seed,
                _create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
            )
            _geo_module._finalize_kriging_stats(expected, "sis_simulation")
    else:
        with _hpgl_call_lock:
            _geo_module._reset_kriging_stats()
            call_sis_simulation_lvm(
                prop_2,
                ikps,
                _c_array(_HPGL_FLOAT_ARRAY, len(data), means),
                len(data),
                seed,
                _create_hpgl_ubyte_array(mask, grid) if mask is not None else None,
                use_correlogram,
            )
            _geo_module._finalize_kriging_stats(expected, "sis_simulation")

    # geo._last_kriging_stats was populated from the C++ SIS stats inside
    # the lock above (see module comment) — the sentinel now carries the
    # simulation's failure counters.

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
