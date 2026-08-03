# GTSIM for K indicators
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    ContProperty, SugarboxGrid, CovarianceModel,
    simple_kriging, write_property,
)
# F-04: _clone_prop and _create_cont_prop are private helpers defined in
# geo_bsd.geo, not exported from the geo_bsd top level. Importing them from
# `geo_bsd` raised ImportError (verified; pre-existing breakage). Import
# from the defining module instead.
from geo_bsd.geo import _clone_prop, _create_cont_prop
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
        # III-19: the old loop overwrote `result.data.flat[i]` inside the
        # j-loop, so the LAST j survived for EVERY hard-data cell — facies
        # 0/1 were conditioned in the WRONG Gaussian interval (live probe:
        # verbatim logic, last j survives for every cell). Select the
        # interval by the cell's facies value instead (sibling
        # gtsimk_const_prob.py pattern: value-driven interval selection).
        if result.mask.flat[i] > 0:
            val = int(result.data.flat[i])
            if val == 0:
                result.data.flat[i] = np.random.uniform(
                    inverse_normal_score(0.0), tk_prop[val].data.flat[i])
            elif val == (indicator - 1):
                result.data.flat[i] = np.random.uniform(
                    tk_prop[val - 1].data.flat[i], inverse_normal_score(1.0))
            else:
                result.data.flat[i] = np.random.uniform(
                    tk_prop[val - 1].data.flat[i], tk_prop[val].data.flat[i])
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
        # II-45: the documented user-provided-probability flow passes a LIST
        # of ContProperty objects — one per indicator (index 0..indicator-1)
        # — which is then subscripted below (pk_prop[0], pk_prop[i]). The
        # old isinstance(ContProperty) check accepted a single ContProperty
        # and then crashed with AttributeError on `_clone_prop(pk_prop[0])`
        # (cp[0] returns an ndarray; _clone_prop requires a property object),
        # so the user-pk path could never work (live probe II-45). Accept a
        # list/tuple of indicator ContProperties.
        if (
            isinstance(pk_prop, (list, tuple))
            and len(pk_prop) == indicator
            and all(isinstance(p, ContProperty) for p in pk_prop)
        ):
            print("User-defined probability properties FOUND.")
        else:
            print("ERROR: WRONG TYPE of user-defined probability properties")
            print("Expected a list of ContProperty, one per indicator "
                  f"({indicator} indicators).")
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
    # III-20: the pk_prop-only path (user provided probability properties,
    # sk_params omitted) previously left sgs_params=None → `**None`
    # TypeError at the sgs_simulation call (live probe: deterministic
    # crash on the documented user-pk flow). Derive the SGS parameters from
    # sk_params when no explicit sgs_params were given; if BOTH are missing
    # the SGS step has no covariance/neighbour configuration at all, so
    # raise a clear error instead of failing with TypeError.
    if sgs_params is None:
        sgs_params = sk_params
    if sgs_params is None:
        raise ValueError(
            "gtsim_Kind: sgs_params (or sk_params) is required for the SGS "
            "step. When providing user-defined pk_prop, pass sk_params "
            "(radiuses/max_neighbours/cov_model) or sgs_params explicitly."
        )
    cdf = calc_cdf(prop2)
    prop1 = sgs_simulation(prop2, grid, cdf, seed=3439275, **sgs_params)

    # 5. Truncation
    print("Truncating SGS result...")
    # II-44: the old loop OVERWROTE prop1.data.flat[i] with the integer
    # category k inside the loop, then the next k+1 iteration compared the
    # OVERWRITTEN integer against tk_prop[k+1] — wrong in BOTH directions
    # (tk[1]<1 and tk[1]>=1; live probe II-44; sibling gtsimk_const_prob.py
    # does it correctly). Compare every threshold against the ORIGINAL SGS
    # value, snapshot once per cell.
    for i in range(prop1.data.size):
        value = prop1.data.flat[i]
        for k in range(indicator - 1):
            if value <= tk_prop[k].data.flat[i]:
                prop1.data.flat[i] = k
                break
            else:
                if k == (indicator - 2):
                    prop1.data.flat[i] = k + 1

    write_property(prop1, "results/GTSIM_TRUNC_RESULT.INC", "TRUNC_RESULT_GT", -99)
    print("Done.")
    print("GTSIM: Finished.")
    return prop1
