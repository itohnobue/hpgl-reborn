# GTSIM for K indicators
import sys
import os
import math
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
            # E2-35: tk_prop here is the CONSTANT marginal-threshold vector
            # (length indicator-1), NOT the per-cell SK thresholds. At hard
            # cells SK is an exact interpolator → the per-cell cumulative
            # probabilities collapse to 0/1 → the per-cell thresholds
            # degenerate to ±10 → the category interval becomes the full
            # range U(−10,+10) (or a point at +10) and the conditioning
            # draw is category-blind noise into the SGS (E2-35). The
            # constant marginal thresholds define proper per-category
            # intervals (sibling gtsimk_const_prob.py pattern).
            if val == 0:
                result.data.flat[i] = np.random.uniform(
                    inverse_normal_score(0.0), tk_prop[val])
            elif val == (indicator - 1):
                result.data.flat[i] = np.random.uniform(
                    tk_prop[val - 1], inverse_normal_score(1.0))
            else:
                result.data.flat[i] = np.random.uniform(
                    tk_prop[val - 1], tk_prop[val])
    return result


def gtsim_Kind(grid, prop, indicator, sk_params=None, pk_prop=None, sgs_params=None):
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
            "gtsim_Kind: indicator must be >= 2 (at least two categories), "
            f"got {indicator}."
        )
    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            val = int(prop.data.flat[i])
            if val < 0 or val >= indicator:
                raise ValueError(
                    "gtsim_Kind: hard-data category out of range — cell "
                    f"{i} has value {val}, expected an integer in "
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
    # E2-40: enforce monotone cumulative probabilities in [0, 1] before the
    # inverse-CDF thresholds. Kriging with Σw > 1 can produce pk < 0 and a
    # user-supplied cumulative Σ > 1 pushes the threshold above 1 —
    # non-monotone cumulatives INVERT the thresholds and the middle
    # category is NEVER simulated (arithmetic-verified; E-M14 sibling: the
    # gtsimk_test.py mean=1.6 made no-neighbour cells get pk=1.6, Σw<1
    # cells pk>1, and sparse regions collapsed to facies 0). Clamp each
    # cumulative to [0, 1] and enforce non-decreasing order (GSLIB gtsim
    # clamps probabilities before the inverse-CDF threshold calculation).
    cum = np.empty((indicator - 1, prop.data.size), dtype=np.float64)
    for k in range(indicator - 1):
        cum[k] = np.clip(p[k].data.flat[:], 0.0, 1.0)
    cum = np.maximum.accumulate(cum, axis=0)
    for k in range(indicator - 1):
        p[k].data.flat[:] = cum[k]
    print("Done.")
    print("Calculating threshold curves (tk)...")
    del pk_prop

    tk_prop = []
    for k in range(indicator - 1):
        tk_prop.append(tk_calculation(p[k]))

    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with tk_prop
    print("Pseudo gaussian transform of initial property (hard data)...")
    # E2-35: build the CONSTANT marginal thresholds (from the empirical
    # hard-data proportions) that condition the transform draws — see
    # pseudo_gaussian_transform. Categories 0..indicator-2 are counted
    # directly; the last category's mass is implicit (1 - sum).
    ind_size = np.zeros(indicator - 1, dtype=np.float64)
    prop_size = 0.0
    for j in range(prop.data.size):
        if prop.mask.flat[j] > 0:
            prop_size = prop_size + 1.0
            val = int(prop.data.flat[j])
            if val < indicator - 1:
                ind_size[val] = ind_size[val] + 1.0
    if prop_size > 0:
        ind_size = ind_size / prop_size
    p_cum = np.maximum.accumulate(np.clip(np.cumsum(ind_size), 0.0, 1.0))
    tk_marg = np.array([inverse_normal_score(pv) for pv in p_cum], dtype=np.float64)
    prop2 = pseudo_gaussian_transform(prop, tk_marg, indicator)
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
    # E2-41: the SGS output prop1 lives in DATA space — the C++
    # sequential_gaussian_simulation back-transforms the simulated
    # standard-normal field through the in-scope empirical CDF
    # (transform_cdf_p at sequential_gaussian_simulation.cpp:165), while
    # tk_prop holds NORMAL-SCORE thresholds Φ⁻¹(pk). Comparing across
    # spaces misclassifies every cell (realized facies compressed toward
    # the marginal — E2-41 trace). Map each per-cell threshold through the
    # SAME empirical CDF used by the back-transform: tk_data = F⁻¹(Φ(tk))
    # (port of the library gtsim_2ind F-02 fix).
    tk_data = []
    for k in range(indicator - 1):
        tk_data.append(cdf.inverse(_norm_cdf(tk_prop[k].data.flat[:])))
    # II-44: the old loop OVERWROTE prop1.data.flat[i] with the integer
    # category k inside the loop, then the next k+1 iteration compared the
    # OVERWRITTEN integer against tk_prop[k+1] — wrong in BOTH directions
    # (tk[1]<1 and tk[1]>=1; live probe II-44; sibling gtsimk_const_prob.py
    # does it correctly). Compare every threshold against the ORIGINAL SGS
    # value, snapshot once per cell.
    for i in range(prop1.data.size):
        value = prop1.data.flat[i]
        for k in range(indicator - 1):
            if value <= tk_data[k][i]:
                prop1.data.flat[i] = k
                break
            else:
                if k == (indicator - 2):
                    prop1.data.flat[i] = k + 1

    write_property(prop1, "results/GTSIM_TRUNC_RESULT.INC", "TRUNC_RESULT_GT", -99)
    print("Done.")
    print("GTSIM: Finished.")
    return prop1
