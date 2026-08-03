# GTSIM for 2 indicators (facies)
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    ContProperty, SugarboxGrid, CovarianceModel,
    simple_kriging, write_property,
)
# F-04: _clone_prop is a private helper defined in geo_bsd.geo, not exported
# from the geo_bsd top level. Importing it from `geo_bsd` raised ImportError
# (verified; pre-existing breakage at restore 761f9d5). Import from the
# defining module instead.
from geo_bsd.geo import _clone_prop
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
            "gtsim_2ind: sgs_params (or sk_params) is required for the SGS "
            "step. When providing user-defined pk_prop, pass sk_params "
            "(radiuses/max_neighbours/cov_model) or sgs_params explicitly."
        )
    cdf = calc_cdf(prop_pg)
    prop_sgs = sgs_simulation(prop_pg, grid, cdf, seed=3439275, **sgs_params)

    # 5. Truncation
    print("Truncating SGS result...")
    # II-43: two independent `if`s re-compared the OVERWRITTEN value. When
    # tk <= 0 (common: any pk <= 0.5 gives tk = -Φ⁻¹(pk) <= 0), a cell set
    # to 0 by the first if was immediately re-set to 1 by the second
    # (`0 >= tk` is True for tk <= 0) → all-facies-1 output (live probe:
    # tk=-0.5 → all facies 1; PRIOR_FIX_ATTEMPT 6b2d2cd). Use if/elif so
    # each cell is classified exactly once.
    # P-02: preserve hard-data facies through the truncation. At hard cells
    # SK is an exact interpolator → pk = 0.0/1.0 exactly → tk = ∓10 (clamp,
    # gaussian_cdf.py:17-20) → the transform draws a DEGENERATE
    # uniform(-10,-10)/(10,10) → SGS reproduces the conditioning value
    # exactly → prop_sgs == tk. The strict `<` below maps that EQUALITY case
    # to facies 1, so every facies-0 hard cell was corrupted to 1
    # (gtsim_test.py hard-data check: 2540/3002 errors; 2540 == facies-0
    # count exactly). Cells whose ORIGINAL prop value is a facies (0 or 1 —
    # non-hard cells hold nodata -99, and the transform at :24-28 already
    # assumes facies are exactly 0/1) keep that facies unconditionally; only
    # non-hard cells are classified by the threshold.
    for i in range(prop_sgs.data.size):
        orig = prop.data.flat[i]
        if orig == 0 or orig == 1:
            prop_sgs.data.flat[i] = orig
        elif prop_sgs.data.flat[i] < tk_prop.data.flat[i]:
            prop_sgs.data.flat[i] = 0
        else:
            prop_sgs.data.flat[i] = 1
    print("Done.")
    print("GTSIM: Finished.")
    return prop_sgs
