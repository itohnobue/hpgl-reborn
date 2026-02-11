# GTSIM for 2 indicators (facies)
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    ContProperty, SugarboxGrid, CovarianceModel,
    simple_kriging, write_property, _clone_prop,
)
from geo_bsd.sgs import sgs_simulation
from geo_bsd.cdf import calc_cdf
from gaussian_cdf import inverse_normal_score


def pseudo_gaussian_transform(prop, tk_prop):
    pg_prop = _clone_prop(prop)

    for i in range(tk_prop.data.size):
        if pg_prop.data.flat[i] == 0:
            pg_prop.data.flat[i] = np.random.uniform(inverse_normal_score(0.0), tk_prop.data.flat[i])
        if pg_prop.data.flat[i] == 1:
            pg_prop.data.flat[i] = np.random.uniform(tk_prop.data.flat[i], inverse_normal_score(1.0))

    return pg_prop


def tk_calculation(pk_prop):
    tk_prop = _clone_prop(pk_prop)
    for i in range(tk_prop.data.size):
        value = inverse_normal_score(tk_prop.data.flat[i])
        tk_prop.data.flat[i] = value
    return tk_prop


def gtsim_2ind(grid, prop, sk_params=None, sgs_params=None, pk_prop=None):
    # prop must be continuous!

    print("Starting GTSIM for 2 Indicator variables...")

    # 1. calculate pk_prop
    # check pk_prop, if presented, use it, if not - do SK

    print("Extracting probability information...")

    if pk_prop is None:
        print("User-defined probability properties NOT FOUND.")
        if sk_params is None:
            print("Simple Kriging parameters NOT FOUND.")
            print("ERROR: Cannot retrieve probability information.")
            return
        print("Simple Kriging parameters FOUND, doing SK...")
        pk_prop = simple_kriging(prop, grid, **sk_params)
    else:
        if isinstance(pk_prop, ContProperty):
            print("User-defined probability properties FOUND.")
        else:
            print("ERROR: WRONG TYPE of user-defined probability properties")
            return

    # 2. calculate tk_prop
    print("Calculating threshold curves (tk)...")
    write_property(pk_prop, "results/GTSIM_PKPROP.INC", "PKPROP", -99)
    tk_prop = tk_calculation(pk_prop)
    write_property(tk_prop, "results/GTSIM_TKPROP.INC", "TKPROP", -99)
    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with pk_prop
    print("Pseudo gaussian transform of initial property (hard data)...")
    prop_pg = pseudo_gaussian_transform(prop, tk_prop)
    write_property(prop, "results/GTSIM_TRANSFORMED_PROP.INC", "TRANSPROP", -99)
    del pk_prop
    print("Done.")

    # 4. SGS on prop (after transform in 3)
    print("Starting SGS on transformed property...")
    if sgs_params is None:
        sgs_params = sk_params
    cdf = calc_cdf(prop_pg)
    prop_sgs = sgs_simulation(prop_pg, grid, cdf, seed=3439275, **sgs_params)

    # 5. Truncation
    print("Truncating SGS result...")
    for i in range(prop_sgs.data.size):
        if prop_sgs.data.flat[i] < tk_prop.data.flat[i]:
            prop_sgs.data.flat[i] = 0
        if prop_sgs.data.flat[i] >= tk_prop.data.flat[i]:
            prop_sgs.data.flat[i] = 1
    print("Done.")
    print("GTSIM: Finished.")
    return prop_sgs
