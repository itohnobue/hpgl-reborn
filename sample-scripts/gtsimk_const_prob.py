# GTSIM for K indicators with constant probabilities
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel,
    simple_kriging, write_property,
)
# F-04: _clone_prop is a private helper defined in geo_bsd.geo, not exported
# from the geo_bsd top level. Importing it from `geo_bsd` raised ImportError
# (verified; pre-existing breakage). Import from the defining module instead.
from geo_bsd.geo import _clone_prop
from geo_bsd.sgs import sgs_simulation
from geo_bsd.cdf import calc_cdf
from gaussian_cdf import inverse_normal_score

# E-M13: the pseudo-Gaussian transform below draws from the numpy global RNG
# (np.random.uniform). sgs_simulation receives a fixed seed, but without
# seeding the transform draws the runs vary run-to-run (grep: 0 seed calls
# across sample-scripts/src before this fix). Seed once at script start with
# the SGS seed so the whole pipeline is reproducible.
np.random.seed(3439275)


def _norm_cdf(x):
    """Standard normal CDF Φ(x), vectorized.

    E2-41: maps the normal-score truncation thresholds into probability
    space before the empirical-CDF inversion (mirrors the library
    gtsim_2ind F-02 fix — truncation must compare in the SAME space as
    the SGS output).
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


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

    # E2-38: validate the hard-data categories up front. An out-of-range
    # category (val >= indicator) previously crashed with a bare IndexError
    # deep inside the transform (tk_prop[val]); val = -1 wrapped silently
    # through negative indexing to the last threshold, conditioning the
    # cell in the WRONG Gaussian interval. Fail loudly at the entry point.
    # indicator < 2 leaves no thresholds at all (indicator-1 == 0) and
    # crashes on the empty threshold list — reject it too.
    if indicator < 2:
        raise ValueError(
            "gtsim_Kind_const_prop: indicator must be >= 2 (at least two "
            f"categories), got {indicator}."
        )
    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            val = int(prop.data.flat[i])
            if val < 0 or val >= indicator:
                raise ValueError(
                    "gtsim_Kind_const_prop: hard-data category out of range "
                    f"— cell {i} has value {val}, expected an integer in "
                    f"[0, {indicator - 1}]."
                )

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
        # II-45: the user-provided-probability flow for the const-prob
        # variant passes a sequence of SCALAR marginal probabilities, one
        # per indicator category (0..indicator-1), consumed by calc_ver()
        # below as pk_prop[i]. The old isinstance(ContProperty) check
        # rejected that documented form and then a single ContProperty was
        # subscripted as pk_prop[i] (AttributeError 'ContProperty' object
        # has no attribute...), so the user-pk path could never work (live
        # probe II-45). Accept any sequence of indicator-1 scalars.
        try:
            pk_prop = list(pk_prop)
        except TypeError:
            print("ERROR: WRONG TYPE of user-defined probability properties")
            return
        if len(pk_prop) != indicator - 1:
            print("ERROR: WRONG NUMBER of user-defined probability properties")
            print(f"Expected {indicator - 1} marginal probabilities, got {len(pk_prop)}.")
            return
    print(pk_prop)

    # 2. Calculate tk_prop
    print("Calculating Pk...")
    p = calc_ver(pk_prop, indicator)
    # E2-40: enforce monotone cumulative probabilities in [0, 1] before the
    # inverse-CDF thresholds. A user-supplied Σ pk > 1 pushes the cumulative
    # above 1 and negative marginals (Σw > 1 kriging overshoot) produce
    # DECREASING cumulatives — non-monotone cumulatives INVERT the
    # thresholds and the middle category is NEVER simulated
    # (arithmetic-verified; E-M14 sibling: pk>1 → inverse_normal_score
    # clamps at 10 → sparse regions collapse to facies 0). Clamp each
    # cumulative to [0, 1] and enforce non-decreasing order (GSLIB gtsim
    # clamps probabilities before the inverse-CDF threshold calculation).
    p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
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
            "gtsim_Kind_const_prop: sgs_params (or sk_params) is required "
            "for the SGS step. When providing user-defined pk_prop, pass "
            "sk_params (radiuses/max_neighbours/cov_model) or sgs_params "
            "explicitly."
        )
    cdf = calc_cdf(prop1)
    prop1 = sgs_simulation(prop1, grid, cdf, seed=3439275, **sgs_params)
    write_property(prop1, "results/GTSIM_SGS_RESULT.INC", "SGS_RESULT_GT", -99)

    # 5. Truncation
    print("Truncating SGS result...")

    # E2-41: the SGS output prop1 lives in DATA space — the C++
    # sequential_gaussian_simulation back-transforms the simulated
    # standard-normal field through the in-scope empirical CDF
    # (transform_cdf_p at sequential_gaussian_simulation.cpp:165), while
    # tk_prop holds NORMAL-SCORE thresholds Φ⁻¹(pk). Comparing across
    # spaces misclassifies every cell (realized facies compressed toward
    # the marginal — E2-41 trace). Map each constant threshold through the
    # SAME empirical CDF used by the back-transform: tk_data = F⁻¹(Φ(tk))
    # (port of the library gtsim_2ind F-02 fix).
    tk_data = cdf.inverse(_norm_cdf(tk_prop))

    for i in range(prop1.data.size):
        for k in range(indicator - 1):
            if prop1.data.flat[i] < tk_data[k]:
                prop1.data.flat[i] = k
                break
            else:
                if k == (indicator - 2):
                    prop1.data.flat[i] = k + 1

    write_property(prop1, "results/GTSIM_TRUNC_RESULT.INC", "TRUNC_RESULT_GT", -99)
    print("Done.")
    print("GTSIM: Finished.")
