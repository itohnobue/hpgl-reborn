# GTSIM for K indicators with constant probabilities
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


def mean_ind(prop_init, indicator):
    prop = _clone_prop(prop_init)
    ind_size = np.zeros(indicator - 1, dtype=float)
    prop_size = 0.0
    for j in range(prop.data.size):
        if prop.mask.flat[j] > 0:
            prop_size = prop_size + 1.0
            for i in range(indicator - 1):
                if prop.data.flat[j] == i:
                    ind_size[i] = ind_size[i] + 1.0
    for i in range(indicator - 1):
        ind_size[i] = ind_size[i] / prop_size
    return ind_size


def calc_ver(pk_prop, indicator):
    s = np.zeros(indicator - 1, dtype=float)
    for i in range(indicator - 1):
        if i > 0:
            s[i] = pk_prop[i] + s[i - 1]
        else:
            s[i] = pk_prop[i]
    return s


def indicator_transform(prop_init, indicator):
    prop = _clone_prop(prop_init)
    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            if prop.data.flat[i] == indicator:
                prop.data.flat[i] = 1
            else:
                prop.data.flat[i] = 0
    return prop


def tk_calculation(p):
    return inverse_normal_score(p)


def pseudo_gaussian_transform(prop_init, tk_prop, indicator):
    prop = _clone_prop(prop_init)
    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            val = int(prop.data.flat[i])
            if val == 0:
                v = np.random.uniform(inverse_normal_score(0.0), tk_prop[val])
                prop.data.flat[i] = v
            elif val == (indicator - 1):
                v = np.random.uniform(tk_prop[val - 1], inverse_normal_score(1.0))
                prop.data.flat[i] = v
            else:
                v = np.random.uniform(tk_prop[val - 1], tk_prop[val])
                prop.data.flat[i] = v
    write_property(prop, "results/GTSIM_TRANSFORMED_PROP.INC", "TRANSPROP", -99)
    return prop


def gtsim_Kind_const_prop(grid, prop, indicator, sk_params=None, pk_prop=None, sgs_params=None):
    # prop must be continuous!

    print("Starting GTSIM for K Indicator variables...")

    # 1. calculate pk_prop
    print("Extracting probability information...")

    if pk_prop is None:
        print("User-defined probability properties NOT FOUND.")
        pk_prop = []
        if sk_params is None:
            print("Simple Kriging parameters NOT FOUND.")
            print("ERROR: Cannot retrieve probability information.")
            return
        print("Calculating pk_prop...")
        pk_prop = mean_ind(prop, indicator)
    else:
        if isinstance(pk_prop, ContProperty):
            print("User-defined probability properties FOUND.")
        else:
            print("ERROR: WRONG TYPE of user-defined probability properties")
            return
    print(pk_prop)

    # 2. Calculate tk_prop
    print("Calculating Pk...")
    p = calc_ver(pk_prop, indicator)
    print(p)
    print("Done.")
    print("Calculating threshold curves (tk)...")
    del pk_prop

    tk_prop = np.zeros(indicator - 1, dtype=float)

    for i in range(indicator - 1):
        tk_prop[i] = tk_calculation(p[i])
    print(tk_prop)
    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with tk_prop
    print("Pseudo gaussian transform of initial property (hard data)...")
    prop1 = pseudo_gaussian_transform(prop, tk_prop, indicator)
    print("Done.")

    # 4. SGS on prop (after transform in 3)
    print("Starting SGS on transformed property...")

    if sgs_params is None:
        sgs_params = sk_params
    cdf = calc_cdf(prop1)
    prop1 = sgs_simulation(prop1, grid, cdf, seed=3439275, **sgs_params)
    write_property(prop1, "results/GTSIM_SGS_RESULT.INC", "SGS_RESULT_GT", -99)

    # 5. Truncation
    print("Truncating SGS result...")

    for i in range(prop1.data.size):
        for k in range(indicator - 1):
            if prop1.data.flat[i] < tk_prop[k]:
                prop1.data.flat[i] = k
                break
            else:
                if k == (indicator - 2):
                    prop1.data.flat[i] = k + 1

    write_property(prop1, "results/GTSIM_TRUNC_RESULT.INC", "TRUNC_RESULT_GT", -99)
    print("Done.")
    print("GTSIM: Finished.")
