# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
# GTSIM for 2 indicators (facies)

import logging
import warnings

import numpy as np

from .cdf import calc_cdf
from .geo import simple_kriging
from .sgs import sgs_simulation

logger = logging.getLogger(__name__)


def pseudo_gaussian_transform(prop, pk_prop, rng=None):
    # NOTE: modifies prop.data in-place. Returns the same prop object.
    if rng is None:
        rng = np.random.RandomState()
    # Use ravel(order='K') for safe flat indexing of Fortran-ordered arrays.
    prop_flat = prop.data.ravel(order="K")
    pk_flat = pk_prop.data.ravel(order="K")
    for i in range(pk_prop.data.size):
        if prop_flat[i] == 0:
            prop_flat[i] = rng.uniform(0.0, pk_flat[i])
        if prop_flat[i] == 1:
            prop_flat[i] = rng.uniform(pk_flat[i], 1.0)
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
    with np.errstate(divide="ignore", invalid="ignore"):
        # Force a copy and clip to avoid log(0) and log(negative)
        p = np.asarray(p, dtype=np.float64).clip(1e-15, 1.0 - 1e-15)
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
    if std_dev <= 0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    # NOTE: modifies pk_prop.data in-place via pk_flat[:] assignment.
    # Uses ravel(order='K') for safe flat indexing of Fortran-ordered arrays.
    pk_flat = pk_prop.data.ravel(order="K")
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
        Pre-computed probability property (if None, will compute via SK)
    sgs_params : dict, optional
        Sequential Gaussian Simulation parameters (if None, uses sk_params)
    tk_mean : float, optional
        Mean for threshold calculation Gaussian PDF (default: 0.0)
    tk_std_dev : float, optional
        Standard deviation for threshold calculation (default: 1.0)
        For standard normal distribution, use 1.0

    Returns:
    --------
    ContProperty
        Simulated indicator property with binary values (0 or 1)
    """
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
        logger.info("Using provided pk_prop.")

    # 2. calculate tk_prop
    # t0_prop = 0
    # t1_prop = tk_calculation(pk_prop)
    # (for 2 indicators)

    logger.info("Calculate tk_prop...")
    # Save original probability data (tk_calculation overwrites pk_prop.data in-place)
    original_pk_data = pk_prop.data.copy()
    tk_prop = tk_calculation(pk_prop, mean=tk_mean, std_dev=tk_std_dev)
    # Extract threshold data for truncation (tk_prop.data contains inverse CDF thresholds)
    threshold_data = tk_prop.data.copy()
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
    # Vectorized thresholding. IEEE 754 NaN fails both >= and <, so
    # NaN passes through to the ~mask branch and becomes 0 — this
    # fixes the old for-loop which silently preserved NaN unchanged.
    mask = prop1_flat >= tk_flat
    prop1_flat[mask] = 1
    prop1_flat[~mask] = 0
    logger.info("Done.")
    return prop1
