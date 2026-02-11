# GTSIM for K indicators
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    ContProperty, SugarboxGrid, CovarianceModel,
    simple_kriging, write_property, _clone_prop, _create_cont_prop,
)
from geo_bsd.sgs import sgs_simulation
from geo_bsd.cdf import calc_cdf
from gaussian_cdf import inverse_normal_score


def indicator_transform(prop, indicator):
    result = _clone_prop(prop)
    for i in range(result.data.size):
        if result.mask.flat[i] > 0:
            if result.data.flat[i] == indicator:
                result.data.flat[i] = 1
            else:
                result.data.flat[i] = 0
    return result


def tk_calculation(p):
    result = _clone_prop(p)
    for i in range(result.data.size):
        result.data.flat[i] = inverse_normal_score(result.data.flat[i])
    return result


def pseudo_gaussian_transform(prop, tk_prop, indicator):
    result = _clone_prop(prop)
    for i in range(result.data.size):
        for j in range(indicator):
            if result.mask.flat[i] > 0:
                if j == 0:
                    result.data.flat[i] = np.random.uniform(
                        inverse_normal_score(0.0), tk_prop[j].data.flat[i])
                elif j == (indicator - 1):
                    result.data.flat[i] = np.random.uniform(
                        tk_prop[j - 1].data.flat[i], inverse_normal_score(1.0))
                else:
                    result.data.flat[i] = np.random.uniform(
                        tk_prop[j - 1].data.flat[i], tk_prop[j].data.flat[i])
    return result


def gtsim_Kind(grid, prop, indicator, sk_params=None, pk_prop=None, sgs_params=None):
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
        print("Simple Kriging parameters FOUND, doing SK...")
        for k in range(indicator):
            ind_prop = indicator_transform(prop, k)
            pk_prop.append(simple_kriging(ind_prop, grid, **sk_params))
    else:
        if isinstance(pk_prop, ContProperty):
            print("User-defined probability properties FOUND.")
        else:
            print("ERROR: WRONG TYPE of user-defined probability properties")
            return

    # 2. Calculate tk_prop
    print("Calculating Pk...")
    p = []
    for k in range(indicator - 1):
        if k == 0:
            p.append(_clone_prop(pk_prop[0]))
        else:
            p.append(_create_cont_prop(prop.data.size))

    for i in range(indicator - 1):
        if i > 0:
            for j in range(prop.data.size):
                p[i].data.flat[j] = pk_prop[i].data.flat[j] + p[i - 1].data.flat[j]
    print("Done.")
    print("Calculating threshold curves (tk)...")
    del pk_prop

    tk_prop = []
    for k in range(indicator - 1):
        tk_prop.append(tk_calculation(p[k]))

    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with tk_prop
    print("Pseudo gaussian transform of initial property (hard data)...")
    prop2 = pseudo_gaussian_transform(prop, tk_prop, indicator)
    print("Done.")

    # 4. SGS on prop (after transform in 3)
    print("Starting SGS on transformed property...")
    if sgs_params is None:
        sgs_params = sk_params
    cdf = calc_cdf(prop2)
    prop1 = sgs_simulation(prop2, grid, cdf, seed=3439275, **sgs_params)

    # 5. Truncation
    print("Truncating SGS result...")
    for i in range(prop1.data.size):
        for k in range(indicator - 1):
            if prop1.data.flat[i] <= tk_prop[k].data.flat[i]:
                prop1.data.flat[i] = k
                break
            else:
                if prop1.data.flat[i] > tk_prop[k].data.flat[i]:
                    prop1.data.flat[i] = k + 1

    write_property(prop1, "results/GTSIM_TRUNC_RESULT.INC", "TRUNC_RESULT_GT", -99)
    print("Done.")
    print("GTSIM: Finished.")
    return prop1
