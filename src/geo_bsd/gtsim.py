# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
# GTSIM for 2 indicators (facies)

import logging
import warnings

import numpy as np

from .cdf import calc_cdf
from .config import GTSIMConfig
from .geo import ContProperty, simple_kriging
from .sgs import sgs_simulation

logger = logging.getLogger(__name__)


def pseudo_gaussian_transform(prop, pk_prop, rng=None):
    # NOTE: modifies prop.data in-place. Returns the same prop object.
    if rng is None:
        rng = np.random.RandomState()
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

    Uses the Beasley-Springer-Moro approximation, accurate to ~4.5e-5
    for probabilities in [1e-15, 1-1e-15]. Vectorized over numpy arrays.

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
        Initial property (continuous values)
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
        Seed for the random number generator (default: 3439275)
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
    # prop must be continious!

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
    # Kriging with negative weights can produce probabilities slightly
    # outside [0, 1] (e.g. -0.04 or 1.02). These are legitimate kriging
    # overshoots, not invalid inputs: GSLIB's gtsim clamps probabilities
    # before the inverse-CDF threshold calculation. Clamp rather than
    # reject so partially-informed data works (a hard reject previously
    # made gtsim_2ind unusable with realistic partially-informed props).
    #
    # F-M15: the clamp is applied to a NEW array, never the caller's
    # pk_prop.data. ravel(order="K") returns a VIEW of pk_prop.data (1D
    # arrays are both C- and F-contiguous, so ContProperty's require("F")
    # returns the same object), and np.clip(..., out=view) would write the
    # clamped values back into the caller's array in place — permanently
    # altering a user-supplied pk_prop whose overshoots were legitimate.
    # Replacing the reference (np.clip without out=) leaves the caller's
    # original array object untouched; every downstream step (tk_calculation,
    # the restore below, pseudo_gaussian_transform) then sees the clamped
    # copy, matching the GSLIB clamp semantics.
    if np.any((pk_flat < 0.0) | (pk_flat > 1.0)):
        pk_prop.data = np.clip(pk_prop.data, 0.0, 1.0)

    # 2. calculate tk_prop
    # t0_prop = 0
    # t1_prop = tk_calculation(pk_prop)
    # (for 2 indicators)

    logger.info("Calculate tk_prop...")
    # Save original probability data (tk_calculation overwrites pk_prop.data in-place)
    # Use copy(order="F") to preserve Fortran (column-major) order expected by
    # the C++ backend. Default C-order copy would corrupt 3D array layout.
    original_pk_data = pk_prop.data.copy(order="F")
    tk_prop = tk_calculation(pk_prop, mean=tk_mean, std_dev=tk_std_dev)
    # Extract threshold data for truncation (tk_prop.data contains inverse CDF thresholds)
    threshold_data = tk_prop.data.copy(order="F")
    # Restore original probabilities for pseudo_gaussian_transform
    pk_prop.data = original_pk_data
    logger.info("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with pk_prop
    # del(pk_prop)

    logger.info("Pseudo gaussian transforming...")
    rng = np.random.RandomState(seed)
    # Copy prop data to avoid mutating caller's data (I2F-11)
    prop.data = prop.data.copy()
    prop = pseudo_gaussian_transform(prop, pk_prop, rng)
    del pk_prop
    logger.info("Done.")

    # 4. SGS on prop (after transfrom in 3)
    # if sgs_params defined - use it
    # if not, use sk_params
    # sill of covariance must be 1

    if sgs_params is None:
        sgs_params = sk_params
    logger.info("Computing CDF...")
    cdf_data = calc_cdf(prop)
    logger.info("Done.")
    logger.info("Testing SGS...")
    prop1 = sgs_simulation(prop, grid, cdf_data, seed=seed, **sgs_params)
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
    # Vectorized thresholding.
    mask = prop1_flat >= tk_flat
    prop1_flat[mask] = 1
    prop1_flat[~mask] = 0
    logger.info("Done.")
    return prop1
