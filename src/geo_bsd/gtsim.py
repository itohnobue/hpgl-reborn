# GTSIM for 2 indicators (facies)
from .geo import *
from numpy import *
try:
    from scipy import *
except ImportError:
    pass
import os

def pseudo_gaussian_transform(prop, pk_prop):
    for i in range(pk_prop.data.size):
        if (prop.data.flat[i] == 0):
            prop.data.flat[i] = random.uniform(0.0, pk_prop.data.flat[i])
        if (prop.data.flat[i] == 1):
            prop.data.flat[i] = random.uniform(pk_prop.data.flat[i], 1.0)
    return prop

def tk_calculation(pk_prop, mean=0.0, std_dev=1.0):
    """
    Calculate threshold probabilities using Gaussian PDF.

    Parameters:
    -----------
    pk_prop : ContProperty
        Probability property to transform
    mean : float, optional
        Mean of the Gaussian distribution (default: 0.0)
    std_dev : float, optional
        Standard deviation (quad_diff) of the Gaussian distribution (default: 1.0)
        For standard normal distribution, use std_dev=1.0

    Returns:
    --------
    ContProperty
        Transformed property with Gaussian PDF values

    Raises:
    -------
    ValueError
        If std_dev <= 0
    """
    # Input validation
    if std_dev <= 0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    values = zeros(pk_prop.data.size, dtype=float)
    for i in range(pk_prop.data.size):
        values[i] = 1./(std_dev*sqrt(2*pi))*exp(-(1./2)*((pk_prop.data.flat[i]-mean)/std_dev)*((pk_prop.data.flat[i]-mean)/std_dev))
    for i in range(pk_prop.data.size):
        pk_prop.data.flat[i] = values[i]
    return pk_prop

def gtsim_2ind(grid, prop, sk_params, do_sk=True, pk_prop=None, sgs_params=None,
               tk_mean=0.0, tk_std_dev=1.0):
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
        Whether to perform simple kriging (default: True)
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

    if (pk_prop is None):
        print("Testing SK...")
        pk_prop = simple_kriging(prop, grid, **sk_params)
        print("Done.")
    else:
        print("Using provided pk_prop.")

    # 2. calculate tk_prop
    # t0_prop = 0
    # t1_prop = tk_calculation(pk_prop)
    # (for 2 indicators)

    print("Calculate tk_prop...")
    tk_prop = tk_calculation(pk_prop, mean=tk_mean, std_dev=tk_std_dev)
    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with pk_prop
    # del(pk_prop)

    print("Pseudo gaussian transforming...")
    prop = pseudo_gaussian_transform(prop,pk_prop)
    del(pk_prop)
    print("Done.")

    # 4. SGS on prop (after transfrom in 3)
    # if sgs_params defined - use it
    # if not, use sk_params
    # sill of covariance must be 1

    if (sgs_params is None):
        sgs_params = sk_params
    print("Testing SGS...")
    prop1 = sgs_simulation(prop, grid, seed=3439275, **sgs_params)
    print("Done.")
    write_property(prop1,"2.INC","PROP1",-99)

    # 5. Truncation
    # if sgs_result(u) >= tk_prop(u) -> sgs_result(u) = 1
    # if sgs_result(u) < tk_prop(u) -> sgs_result(u) = 0

    print("Truncation.")
    for i in range(prop.data.size):
        if (prop1.data.flat[i] >= tk_prop.data.flat[i]):
            prop1.data.flat[i] = 1
        if (prop1.data.flat[i] < tk_prop.data.flat[i]):
            prop1.data.flat[i] = 0
    write_property(prop1,"3.INC","prop1",-99)
    print("Done.")
    return prop1
