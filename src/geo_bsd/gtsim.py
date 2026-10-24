# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
# GTSIM for 2 indicators (facies)

import logging
import math
import warnings

import numpy as np

from .cdf import calc_cdf
from .config import GTSIMConfig
from .geo import ContProperty, simple_kriging
from .sgs import sgs_simulation
from .validation import CriticalValidationError, GridValidator, ParameterValidator

logger = logging.getLogger(__name__)


def pseudo_gaussian_transform(prop, pk_prop, rng=None):
    # NOTE: modifies prop.data in-place. Returns the same prop object.
    if rng is None:
        # E-M6: the default RandomState was unseeded — every direct call
        # produced a different transform, defeating reproducibility. Seed
        # the default so standalone calls are deterministic; callers can
        # still pass their own rng (gtsim_2ind passes RandomState(seed)).
        rng = np.random.RandomState(0)
    # Use ravel(order='K') for safe flat indexing of Fortran-ordered arrays.
    prop_flat = prop.data.ravel(order="K")
    pk_flat = pk_prop.data.ravel(order="K")

    # Vectorized: generate random values for all zero/one cells at once.
    zero_mask = prop_flat == 0
    if np.any(zero_mask):
        prop_flat[zero_mask] = rng.uniform(0.0, pk_flat[zero_mask])
    one_mask = prop_flat == 1
    if np.any(one_mask):
        prop_flat[one_mask] = rng.uniform(pk_flat[one_mask], 1.0)

    # Clamp output to [0, 1] as a safety net
    prop_flat[:] = np.clip(prop_flat, 0.0, 1.0)
    return prop


def _norm_ppf(p):
    """Inverse CDF (percent point function) for the standard normal distribution.

    Uses the Abramowitz & Stegun 26.2.23 rational approximation, accurate to
    ~4.4e-4 (max abs error measured 4.44e-4 at p=0.3577) for probabilities in
    [1e-15, 1-1e-15]. Vectorized over numpy arrays.

    Parameters
    ----------
    p : numpy.ndarray
        Probabilities in (0, 1). Values are clipped to avoid singularities.

    Returns
    -------
    numpy.ndarray
        Standard normal quantiles returning Φ⁻¹(p).
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise ValueError("_norm_ppf: input contains NaN or Inf values")
    with np.errstate(divide="ignore", invalid="ignore"):
        # Clip to avoid log(0) and log(negative)
        p = p.clip(1e-15, 1.0 - 1e-15)
        # Split at 0.5 for numerical stability
        q = np.where(p > 0.5, 1.0 - p, p)
        upper = p > 0.5

        t = np.sqrt(-2.0 * np.log(q))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        # Φ⁻¹(p) is negative for p < 0.5, positive for p > 0.5
        z[~upper] = -z[~upper]
        return z


def _norm_cdf(x):
    """Standard normal cumulative distribution function Φ(x).

    Vectorized over numpy arrays using ``math.erf``:
    Φ(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Parameters
    ----------
    x : numpy.ndarray
        Standard normal quantiles.

    Returns
    -------
    numpy.ndarray
        Cumulative probabilities Φ(x) in (0, 1).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def tk_calculation(pk_prop, mean=0.0, std_dev=1.0):
    """
    Calculate truncation thresholds using the inverse normal CDF.

    Given the probability p = P(indicator=1) from simple kriging, computes
    the threshold t such that P(N(mean, std_dev²) >= t) = p.

    The threshold is: t = mean - std_dev * Φ⁻¹(p)

    Parameters:
    -----------
    pk_prop : ContProperty
        Probability property from simple kriging (values in ~[0, 1]).
    mean : float, optional
        Mean of the Gaussian distribution (default: 0.0)
    std_dev : float, optional
        Standard deviation of the Gaussian distribution (default: 1.0)

    Returns:
    --------
    ContProperty
        Same property object with data overwritten by threshold values.

    Raises:
    -------
    ValueError
        If std_dev <= 0
    """
    # Input validation
    if not np.isfinite(mean):
        raise ValueError(f"mean must be finite, got {mean}")
    if not np.isfinite(std_dev) or std_dev <= 0:
        raise ValueError(f"std_dev must be positive and finite, got {std_dev}")

    # NOTE: modifies pk_prop.data in-place via pk_flat[:] assignment.
    # Uses ravel(order='K') for safe flat indexing of Fortran-ordered arrays.
    pk_flat = pk_prop.data.ravel(order="K")
    if np.any(~np.isfinite(pk_flat)):
        raise ValueError("tk_calculation: pk_prop.data contains NaN or Inf values")
    # Compute inverse normal CDF: t = mean - std_dev * Φ⁻¹(p)
    z = _norm_ppf(pk_flat)
    pk_flat[:] = mean - std_dev * z
    return pk_prop


def gtsim_2ind(
    grid,
    prop,
    sk_params,
    do_sk=True,
    pk_prop=None,
    sgs_params=None,
    tk_mean=0.0,
    tk_std_dev=1.0,
    seed=3439275,
    config=None,
):
    """
    Gaussian Truncated Simulation for 2 indicators (facies).

    Parameters:
    -----------
    grid : Grid
        Simulation grid
    prop : ContProperty
        Initial property holding the hard facies. Informed cells must carry
        ``0`` or ``1`` — the pseudo-Gaussian transform and the truncation are
        defined for binary facies only, and any other informed value is
        rejected with ``ValueError`` (a continuous field is not a 2-facies
        model). Masked-out cells may hold any value (e.g. a sentinel such as
        ``-99``); the simulation overwrites them.
    sk_params : dict
        Simple kriging parameters
    do_sk : bool, optional
        Deprecated — this parameter is unused. Simple kriging is always performed
        unless pk_prop is provided. Kept for backward compatibility.
        (default: True)
    pk_prop : ContProperty, optional
        Pre-computed probability property (if None, will compute via SK).
        Values slightly outside [0, 1] (kriging overshoots) are clamped to
        [0, 1] on an internal copy — the caller's pk_prop.data array is
        never mutated (F-M15).
    sgs_params : dict, optional
        Sequential Gaussian Simulation parameters (if None, uses sk_params)
    tk_mean : float, optional
        Mean for threshold calculation Gaussian PDF (default: 0.0)
    tk_std_dev : float, optional
        Standard deviation for threshold calculation (default: 1.0)
        For standard normal distribution, use 1.0
    seed : int, optional
        Seed for the random number generator (default: 3439275). The package's
        full non-negative int64 seed range is accepted: the value drives the
        engine (SGS) directly and is folded to the 32-bit word numpy's
        ``RandomState`` requires for the pseudo-Gaussian transform, so a seed
        recorded from ``sgs_simulation``/``sis_simulation`` replays here.
    config : GTSIMConfig or None, optional
        Pre-configured GTSIM parameters as a frozen dataclass.  When provided,
        its values override the corresponding keyword arguments above.
        Default: ``None``.

    Returns:
    --------
    ContProperty
        Simulated indicator property with binary values (0 or 1)
    """
    # When config is provided, override parameter values from config
    if config is not None:
        if not isinstance(config, GTSIMConfig):
            raise TypeError(
                f"gtsim_2ind: config must be GTSIMConfig, got {type(config).__name__}"
            )
        tk_mean = config.tk_mean
        tk_std_dev = config.tk_std_dev
        seed = config.seed
    # prop carries the hard facies (0/1 at informed cells) — see the gate
    # after the pk_prop contract below.

    # 1. calculate pk_prop
    # check pk_prop, if presented, use it, if not - do SK

    if not do_sk:
        warnings.warn(
            "The 'do_sk' parameter is deprecated and will be removed in a future version. "
            "Simple kriging is always performed unless pk_prop is provided.",
            FutureWarning,
            stacklevel=2,
        )
        if pk_prop is None:
            raise ValueError(
                "gtsim_2ind: do_sk=False requires pk_prop to be provided, "
                "since simple kriging is the only way to compute pk_prop."
            )

    if pk_prop is None:
        logger.info("Testing SK...")
        pk_prop = simple_kriging(prop, grid, **sk_params)
        logger.info("Done.")
    else:
        if not isinstance(pk_prop, ContProperty):
            raise TypeError(
                "gtsim_2ind: pk_prop must be ContProperty, "
                f"got {type(pk_prop).__name__}"
            )
        logger.info("Using provided pk_prop.")

    # Validate pk_prop regardless of source (SK or user-provided).
    if not isinstance(pk_prop, ContProperty):
        raise TypeError(
            "gtsim_2ind: pk_prop must be ContProperty, "
            f"got {type(pk_prop).__name__}"
        )
    pk_flat = pk_prop.data.ravel(order="K")
    if np.any(~np.isfinite(pk_flat)):
        raise ValueError("gtsim_2ind: pk_prop.data contains NaN or Inf values")

    # F-007: pk_prop is consumed pairwise with the grid by FLAT node index
    # (pseudo_gaussian_transform pairs it with prop's flat mask, and the
    # truncation indexes thresholds the same way), so its volume — and, for 3D
    # data, its per-dimension shape — must match the grid. Without this
    # contract an equal-volume but differently-shaped pk_prop completed
    # silently with the thresholds landing on the wrong cells (measured: 4 of
    # 50 cells flipped facies on a smooth probability gradient), while a
    # wrong-VOLUME pk_prop died inside numpy with an opaque IndexError.
    # Mirrors the sibling guards sgs.py:346-375 (LVM mean) and sis.py:305-344;
    # a 1D flat probability vector of the grid volume stays legal.
    GridValidator.validate_array_size(pk_prop.data, (grid.x, grid.y, grid.z))
    if pk_prop.data.ndim not in (1, 3):
        raise ValueError(
            f"gtsim_2ind: pk_prop.data must be 1D flat (size "
            f"{grid.x * grid.y * grid.z}) or 3D with shape "
            f"({grid.x}, {grid.y}, {grid.z}), got {pk_prop.data.ndim}D "
            f"shape {pk_prop.data.shape}"
        )
    if pk_prop.data.ndim == 3 and pk_prop.data.shape != (grid.x, grid.y, grid.z):
        raise ValueError(
            f"gtsim_2ind: pk_prop.data shape {pk_prop.data.shape} does not "
            f"match grid dimensions ({grid.x}, {grid.y}, {grid.z})"
        )

    # G-26: prop must hold BINARY facies at informed cells. The algorithm is
    # defined for 0/1 only: pseudo_gaussian_transform transforms exactly the
    # cells equal to 0 or 1 and then silently clips everything else into
    # [0, 1], so continuous input (the input the pre-fix docstring invited)
    # saturated to a 100 % single-facies field with no exception and no
    # warning (measured P(category 1) = 1.0000 for the data-unit values
    # 12/27). Reject rather than silently degrade. Masked-out cells are
    # exempt: they may hold a sentinel such as -99 and are overwritten by the
    # simulation.
    prop_gate_flat = prop.data.ravel(order="K")
    informed_gate = prop.mask.ravel(order="K") != 0
    non_binary = informed_gate & (prop_gate_flat != 0) & (prop_gate_flat != 1)
    if np.any(non_binary):
        raise ValueError(
            f"gtsim_2ind: prop must hold binary facies (0 or 1) at informed "
            f"cells, got {int(non_binary.sum())} informed cell(s) with other "
            f"values (e.g. {float(prop_gate_flat[non_binary][0])!r})"
        )

    # M-24 / F-M15 / 2-M-12: never mutate the caller's pk_prop object or its
    # data buffer. All downstream in-place transforms (the overshoot clamp,
    # tk_calculation, the pseudo-Gaussian transform) run on an INTERNAL
    # working copy, never on pk_prop.data. Pre-fix, the no-overshoot path
    # called tk_calculation directly on the caller's array, writing the
    # inverse-CDF thresholds into it in place through the ravel('K') view;
    # the later attribute restore hid the damage from pk_prop itself, but any
    # external buffer aliasing pk_prop.data was permanently corrupted (M-24).
    # Kriging with negative weights can produce probabilities slightly
    # outside [0, 1] (e.g. -0.04 or 1.02). These are legitimate kriging
    # overshoots, not invalid inputs: GSLIB's gtsim clamps probabilities
    # before the inverse-CDF threshold calculation. Clamp rather than
    # reject so partially-informed data works (a hard reject previously
    # made gtsim_2ind unusable with realistic partially-informed props).
    # The clamp is applied to the working copy's data, matching GSLIB's
    # clamp semantics without touching the caller's array.
    if np.any((pk_flat < 0.0) | (pk_flat > 1.0)):
        pk_work_data = np.clip(pk_prop.data, 0.0, 1.0)
    else:
        pk_work_data = pk_prop.data.copy(order="F")
    pk_work = ContProperty(pk_work_data, pk_prop.mask.copy(order="F"))

    # 2. calculate tk_prop
    # t0_prop = 0
    # t1_prop = tk_calculation(pk_prop)
    # (for 2 indicators)

    logger.info("Calculate tk_prop...")
    # Save original probability data (tk_calculation overwrites pk_work.data
    # in-place — on the internal copy, never the caller's buffer).
    # Use copy(order="F") to preserve Fortran (column-major) order expected by
    # the C++ backend. Default C-order copy would corrupt 3D array layout.
    original_pk_data = pk_work.data.copy(order="F")
    tk_prop = tk_calculation(pk_work, mean=tk_mean, std_dev=tk_std_dev)
    # Extract threshold data for truncation (tk_prop.data contains inverse CDF thresholds)
    threshold_data = tk_prop.data.copy(order="F")
    # Restore original probabilities for pseudo_gaussian_transform
    pk_work.data = original_pk_data
    logger.info("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with pk_prop
    # del(pk_prop)

    logger.info("Pseudo gaussian transforming...")
    # G-27: GTSIM was the only simulation entry point with no seed gate, and
    # np.random.RandomState cannot represent the package's int64 seed domain —
    # it raised numpy's raw "Seed must be between 0 and 2**32 - 1" before any
    # HPGL-side validation, naming neither GTSIM nor the accepted range.
    # Validate through the package validator first (so a bool/negative/bad-type
    # seed is reported against gtsim_2ind), then fold the full 64-bit seed into
    # the 32-bit word RandomState requires — the same width-preserving split
    # the C++ engine performs with std::seed_seq (bs_random_generator.cpp:9-21)
    # — so a seed recorded from SGS/SIS/variogram replays here. The unfolded
    # seed is still forwarded to sgs_simulation below.
    try:
        ParameterValidator.validate_seed(seed)
    except CriticalValidationError as err:
        raise CriticalValidationError(f"gtsim_2ind: {err.message}", "seed") from err
    except TypeError as err:
        raise TypeError(f"gtsim_2ind: {err}") from err
    rng = np.random.RandomState((int(seed) ^ (int(seed) >> 32)) & 0xFFFFFFFF)
    # 2-M-12: transform an INTERNAL working copy of prop, never the caller's
    # prop object. Pre-fix, `prop.data = prop.data.copy()` REBOUND the
    # caller's attribute (mutating the caller's object) and the comment
    # claiming "avoid mutating caller's data (I2F-11)" was false —
    # pseudo_gaussian_transform then transformed the rebound array in place.
    prop_work = ContProperty(prop.data.copy(order="F"), prop.mask.copy(order="F"))
    prop_work = pseudo_gaussian_transform(prop_work, pk_work, rng)
    del pk_work
    logger.info("Done.")

    # 4. SGS on prop_work (after transform in 3)
    # if sgs_params defined - use it
    # if not, use sk_params
    # sill of covariance must be 1

    if sgs_params is None:
        sgs_params = sk_params
    logger.info("Computing CDF...")
    cdf_data = calc_cdf(prop_work)
    logger.info("Done.")
    logger.info("Testing SGS...")
    prop1 = sgs_simulation(prop_work, grid, cdf_data, seed=seed, **sgs_params)
    logger.info("Done.")

    # 5. Truncation
    # if sgs_result(u) >= tk_prop(u) -> sgs_result(u) = 1
    # if sgs_result(u) < tk_prop(u) -> sgs_result(u) = 0

    logger.info("Truncation.")
    # Use ravel(order='K') for safe flat indexing of Fortran-ordered arrays.
    prop1_flat = prop1.data.ravel(order="K")
    tk_flat = threshold_data.ravel(order="K")
    # Detect NaN/Inf early: IEEE 754 NaN fails both >= and <, silently
    # landing in the ~mask branch where it becomes 0 — indistinguishable
    # from a legitimate output 0. Reject NaN input to preserve correctness.
    if np.any(~np.isfinite(prop1_flat)):
        raise RuntimeError("gtsim_2ind: NaN or Inf in SGS output (prop1.data)")
    if np.any(~np.isfinite(tk_flat)):
        raise RuntimeError("gtsim_2ind: NaN or Inf in threshold data")

    # F-02 (HIGH): the SGS output prop1 lives in DATA space — the C++
    # sequential_gaussian_simulation back-transforms the simulated
    # standard-normal field through the in-scope empirical CDF
    # (transform_cdf_p(output, gaussian_cdf_t(), ncdf) at
    # sequential_gaussian_simulation.cpp:165), so prop1 = F⁻¹(Φ(Z)).
    # The tk thresholds computed by tk_calculation live in NORMAL-SCORE
    # space. Comparing data-space output against normal-score thresholds
    # grossly misclassifies categories (pre-fix live repro: pk=0.5 ->
    # facies-1 proportion 1.000 instead of ~0.5). Map the threshold into
    # the SAME data space via the same empirical CDF used by the
    # back-transform: tk_data = F⁻¹(Φ(tk_std)).
    #
    # II-39: non-default tk_mean/tk_std_dev silently distort proportions.
    # tk_calculation returns tk = tk_mean - tk_std_dev·Φ⁻¹(p); the engine
    # simulates a STANDARD-normal field, so the effective threshold is
    # (tk - tk_mean)/tk_std_dev = -Φ⁻¹(p). Normalize tk params to
    # standard-normal space before the CDF mapping — the F-02 "same space"
    # fix alone (tk_data = F⁻¹(Φ(tk)) with raw tk) does NOT repair the
    # tk-param distortion.
    tk_std = (tk_flat - tk_mean) / tk_std_dev
    tk_data = cdf_data.inverse(_norm_cdf(tk_std))
    # Vectorized thresholding.
    mask = prop1_flat >= tk_data
    prop1_flat[mask] = 1
    prop1_flat[~mask] = 0

    # E2-32: preserve hard-data facies through the truncation. The
    # per-cell threshold F⁻¹(1−p) separates classes only when the SGS
    # output at the cell follows the empirical distribution with
    # probability p — at hard-data cells the output is the degenerate
    # back-transformed conditioning value, so heterogeneous p silently
    # flips the original facies (Monte Carlo reproduction: 96% flip rate).
    # Restore the original 0/1 facies at informed cells after thresholding
    # (sibling sample-script P-02 documents this exact fix class).
    prop_orig_flat = prop.data.ravel(order="K")
    prop_mask_flat = prop.mask.ravel(order="K")
    hard_cells = (prop_mask_flat != 0) & (
        (prop_orig_flat == 0) | (prop_orig_flat == 1)
    )
    prop1_flat[hard_cells] = prop_orig_flat[hard_cells]
    logger.info("Done.")
    return prop1
