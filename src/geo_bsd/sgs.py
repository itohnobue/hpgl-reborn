# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import logging
import numpy

logger = logging.getLogger(__name__)

# Import validation framework
from . import geo as _geo_module
from .config import SGSConfig
from .ffi_adapter import (
    _HPGL_KRIGING_KIND,
    _HPGL_SGS_PARAMS,
    _hpgl_call_lock,
    call_sgs_lvm_simulation,
    call_sgs_simulation,
    normalize_mask_binary,
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

# Simulation failure statistics ARE now populated by the C++ SGS path
# (F-M6: sequential_simulation.h calls set_kriging_stats with the
# kriging outcome counters). The Python wrapper consumes them via
# geo._finalize_kriging_stats (populate + raise on failure counters)
# under _hpgl_call_lock, mirroring the continuous-kriging wrappers.


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


def _warn_all_ndmin_skipped(min_neighbours, expected_uninformed):
    """E-M57: signal an all-ndmin-skipped run programmatically.

    With min_neighbours > 0 the wrapper passes expected=0 to
    ``_finalize_kriging_stats`` (partial ndmin skips must not fire the
    spurious "could not be kriged" warning — test-pinned F11), which means
    an ALL-skipped run — every cell the run should have simulated left
    unsimulated by the ndmin gate — was previously completely silent: all
    C++ counters stay zero (points_calculated == points_without_neighbours
    == points_singularity == 0), indistinguishable from a successful no-op.
    That zero-attempt signature IS the all-skipped signal when the run had
    cells it should have simulated (expected_uninformed > 0); warn loudly
    so the no-output condition is observable programmatically (caplog /
    log capture), not just via the C++ stderr counter.
    """
    stats = getattr(_geo_module, "_last_kriging_stats", None)
    if min_neighbours <= 0 or expected_uninformed <= 0 or stats is None:
        return
    calculated = int(stats.get("points_calculated", 0))
    no_neighbours = int(stats.get("points_without_neighbours", 0))
    singular = int(stats.get("points_singularity", 0))
    if calculated == 0 and no_neighbours == 0 and singular == 0:
        logger.warning(
            "sgs_simulation: no nodes were simulated — all %d expected "
            "uninformed cells were left unsimulated by the ndmin gate "
            "(fewer than min_neighbours=%d ORIGINAL conditioning data in "
            "the search radius). Output remains at its initial (masked) "
            "state. stats=%s",
            expected_uninformed, min_neighbours, stats,
        )


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
        Kriging method: ``"sk"`` for Simple Kriging or ``"ok"`` for Ordinary
        Kriging.  Default: ``"sk"``.  Only these two values are accepted; any
        other value raises ``ValueError``.
    mean : None, float, or numpy.ndarray, optional
        Stationary mean value. If ``None``, the mean is calculated automatically
        from source data. If a non-scalar ndarray, it is used as a locally
        varying mean (LVM). Default: ``None``.

        .. note::
            The ``mean`` argument is honored for both ``kriging_type``
            values.  With ``kriging_type="sk"`` (simple kriging) the
            user-supplied mean is used as the stationary mean in the
            kriging estimate.  With ``kriging_type="ok"`` (ordinary
            kriging) the kriged estimate is computed from the
            conditioning data without an explicit mean term (the OK
            weight calculator solves for the local mean implicitly), but
            the user-supplied mean IS applied on the failure fallback:
            nodes that cannot be kriged (no neighbours in the search
            radius, or a singular kriging system) draw from
            N(mean, 1.0) rather than N(0, 1) (GSLIB sgsim semantics:
            ``cmean = gmean; cstdev = 1.0``).  A non-scalar (LVM) mean
            array is always used as the local varying mean, and the LVM
            kernel performs simple kriging against it.
    use_harddata : bool, optional
        If ``True``, use source data values for simulation. If ``False``,
        ignore source data values. Default: ``True``.
    mask : numpy.ndarray or None, optional
        3D array where non-zero marks cells to simulate and ``0`` marks cells
        to skip (the library-wide "non-zero = informed" mask contract,
        got-20260803180153 — non-binary values like 2 are normalized to 1 at
        the boundary). If ``None``, all cells are simulated. Default: ``None``.
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
    # Reset the documented kriging-stats inspection point and populate from
    # the C++ SGS stats under _hpgl_call_lock (F-M6/F-N12) — see the module
    # comment. The reset must run INSIDE the lock so each call's sentinel
    # write is atomic with respect to other kriging/simulation threads.

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
    # E-M9: simulation allows max_neighbours=0 — the C++ contract documents
    # 0 as "unconditional simulation" (api.cpp:144-167,
    # validate_max_neighbours_or_throw accepts 0; kriging entries use a
    # separate >=1 gate). The default min_neighbors=1 in
    # validate_kriging_params applies to kriging entries only.
    valid_radiuses = validate_kriging_params(
        grid, radiuses, max_neighbours, cov_model, min_neighbors=0
    )
    # Ensure radiuses are integers for ctypes (c_int * 3)
    valid_radiuses = tuple(int(r) for r in valid_radiuses)

    # Validate simulation-specific parameters
    validate_simulation_params(seed=seed, min_neighbours=min_neighbours, max_neighbours=max_neighbours)

    ParameterValidator.validate_property_type(prop, ContProperty, "sgs_simulation")

    prop.fix_shape(grid)
    cov_model = normed_cov_model(cov_model)

    out_prop, mean, mask = __prepare_sgs(prop=prop, mean=mean, use_harddata=use_harddata, mask=mask)

    # Mask semantics (got-20260803180153): the library-wide contract is
    # "non-zero = informed". The C++ SGS kernel gates simulation on
    # mask[node] == 1 (sequential_simulation.h:124) while the Python
    # expected-cell count below counts mask != 0 — a mask value like 2 is
    # counted as "simulate" in Python but silently skipped by C++. The
    # centralized normalization converts any non-zero mask to binary 1 at
    # the boundary, so both sides agree on the SAME cell set after
    # normalization (instead of rejecting the mask as invalid).
    mask = normalize_mask_binary(mask, "sgs_simulation")

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

    # Expected number of cells the simulation should calculate = uninformed
    # cells (out_prop.mask == 0) that are not masked out by the simulation
    # mask. Computed before the lock (cheap numpy sum) — keeps the lock hold
    # minimal; the C++ SGS counters decide whether the failure warning fires.
    expected_uninformed = grid.x * grid.y * grid.z - int(numpy.sum(out_prop.mask > 0))
    if mask is not None:
        expected_uninformed = int(numpy.sum((out_prop.mask.ravel() == 0) & (mask.ravel() != 0)))

    # GSLIB ndmin semantics (sequential_simulation.h): when
    # min_neighbours > 0, C++ deliberately leaves nodes with fewer than
    # min_neighbours conditioning data unsimulated, and those nodes are
    # excluded from ALL stats counters (points_calculated /
    # points_without_neighbours / points_singularity). The ndmin skip count
    # is carried in the C++ kriging_stats_t (m_points_ndmin_skipped) but is
    # not copied into the Python-visible C-API stats dict (api.cpp
    # hpgl_get_kriging_stats copies fields by name), so Python cannot
    # compute a matching expected count a priori — an expected based on
    # uninformed cells would spuriously fire the "could not be kriged"
    # warning on every sparse-data run with min_neighbours > 0. Pass
    # expected=0 in that configuration to suppress the misleading warning;
    # genuine numerical failures still raise via _finalize_kriging_stats
    # (points_singularity > 0) and C++ reports the ndmin skip count itself.
    # E-M57: the ALL-skipped case (no node kriged at all) must still
    # produce a programmatic signal — _warn_all_ndmin_skipped below detects
    # the zero-attempt signature in the returned stats and warns.
    # E-M9/R-20: max_neighbours=0 is the C++-documented "unconditional
    # simulation" mode (api.cpp:144-167) — find() returns an empty
    # neighbourhood for every node (sugarbox_neighbour_lookup.h:94-96),
    # every node takes the KI_NO_NEIGHBOURS marginal-draw path, and
    # points_calculated stays 0. Passing expected=0 (the same suppression
    # mechanism as the min_neighbours > 0 case above) exempts the run from
    # the spurious "N of N cells could not be kriged" warning — the
    # marginal draw IS the requested mode, not a failure. Genuine singular
    # failures still raise via _finalize_kriging_stats
    # (points_singularity > 0, independent of expected).
    if min_neighbours > 0 or max_neighbours == 0:
        expected = 0
    else:
        expected = expected_uninformed

    if mean is None or numpy.isscalar(mean):
        if mean is not None:
            ParameterValidator.validate_scalar_mean(mean, "sgs_simulation")
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        with _hpgl_call_lock:
            _geo_module._reset_kriging_stats()
            call_sgs_simulation(_cont_marr, sgsp, _cdf_struct, mean, _mask_struct)
            _geo_module._finalize_kriging_stats(expected, "sgs_simulation")
            _warn_all_ndmin_skipped(min_neighbours, expected_uninformed)

    else:
        _cont_marr = _create_hpgl_cont_masked_array(out_prop, grid)
        GridValidator.validate_array_size(mean, (grid.x, grid.y, grid.z))
        # II-18: equal-volume per-dimension shape mismatch on the LVM mean
        # silently permutes the mean field — the C++ LVM provider consumes
        # the buffer by flat node index, so a (2,2,2) mean on a (1,2,4)
        # grid (both volume 8) is misread with no exception. Mirror the
        # lvm_kriging R-13 guard (geo.py:1919-1927). 1D (flat) mean vectors
        # are covered by the size check and carry no per-dim meaning.
        # E2-01/R-09: the guard above is 3D-only — an equal-volume 2D mean
        # (e.g. (2,2) on a (1,2,2) grid) sails past the size check and the
        # ndim==3 guard, then the F-order flat layout of the 2D array is
        # consumed by flat node index with a silently permuted mean field —
        # exactly the defect class fixed for lvm_kriging (geo.py:1971-1982).
        # Only 1D flat and exactly grid-shaped 3D arrays are unambiguous;
        # reject every other ndim for the same reason.
        if mean.ndim not in (1, 3):
            raise ValueError(
                f"sgs_simulation: LVM mean must be 1D flat (size "
                f"{grid.x * grid.y * grid.z}) or 3D with shape "
                f"({grid.x}, {grid.y}, {grid.z}), got {mean.ndim}D "
                f"shape {mean.shape}"
            )
        if mean.ndim == 3 and (
            mean.shape[0] != grid.x
            or mean.shape[1] != grid.y
            or mean.shape[2] != grid.z
        ):
            raise ValueError(
                f"sgs_simulation: 3D LVM mean shape {mean.shape} does not match "
                f"grid dimensions ({grid.x}, {grid.y}, {grid.z})"
            )
        if not numpy.all(numpy.isfinite(mean)):
            raise ValueError(
                "sgs_simulation: LVM mean array contains NaN or Inf values"
            )
        _float_arr = _create_hpgl_float_array(mean, grid)
        with _hpgl_call_lock:
            _geo_module._reset_kriging_stats()
            call_sgs_lvm_simulation(_cont_marr, sgsp, _cdf_struct, _float_arr, _mask_struct)
            _geo_module._finalize_kriging_stats(expected, "sgs_simulation")
            _warn_all_ndmin_skipped(min_neighbours, expected_uninformed)

    # geo._last_kriging_stats was populated from the C++ SGS stats inside
    # the lock above (see module comment) — the sentinel now carries the
    # simulation's failure counters.

    # Validate output data for NaN/Inf after C++ computation.
    # C++ simulation can return NaN/Inf from degenerate matrices,
    # zero neighbours, or division by zero.
    # All 7 kriging functions have this check; SGS/SIS must match.
    if not numpy.all(numpy.isfinite(out_prop.data)):
        raise RuntimeError(
            "sgs_simulation: output data contains NaN or Inf after C++ computation"
        )

    return out_prop
