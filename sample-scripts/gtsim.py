# GTSIM for 2 indicators (facies)
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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

# F-366: the pseudo-Gaussian transform below draws a random value for every
# hard-data cell. The E-M13 fix seeded the process-global RNG at import
# (np.random.seed(3439275)); that made the pipeline reproducible but also
# clobbered the caller's RNG state for the rest of the process. Keep the SAME
# reproducible MT19937 stream in a private generator instead — importing this
# module no longer touches np.random's global state.
_RNG = np.random.RandomState(3439275)


def _norm_cdf(x):
    """Standard normal CDF Φ(x), vectorized.

    E2-41: maps the normal-score truncation thresholds into probability
    space before the empirical-CDF inversion (mirrors the library
    gtsim_2ind F-02 fix — truncation must compare in the SAME space as
    the SGS output). F-373 also uses it on a single threshold to recover
    the cell probability pk from Φ⁻¹(pk).
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def pseudo_gaussian_transform(prop, tk_prop):
    pg_prop = _clone_prop(prop)

    for i in range(tk_prop.data.size):
        # F-373: draw the conditioning values in PROBABILITY space, exactly as
        # the library entry point does (src/geo_bsd/gtsim.py:34,37 —
        # rng.uniform(0.0, pk) for facies 0, rng.uniform(pk, 1.0) for facies
        # 1). The previous draws used normal-score bounds, (-10, tk) and
        # (tk, +10), which made the sample a DIFFERENT algorithm from
        # geo_bsd.gtsim_2ind (measured on the same input: uninformed facies-1
        # proportion 0.7154 here vs 0.626 in the library). tk_prop holds the
        # normal-score threshold Φ⁻¹(pk), so recover the cell's probability
        # with the standard normal CDF.
        pk = _norm_cdf(tk_prop.data.flat[i])
        if pg_prop.data.flat[i] == 0:
            pg_prop.data.flat[i] = _RNG.uniform(0.0, pk)
        if pg_prop.data.flat[i] == 1:
            pg_prop.data.flat[i] = _RNG.uniform(pk, 1.0)

    return pg_prop


def tk_calculation(pk_prop):
    tk_prop = _clone_prop(pk_prop)
    for i in range(tk_prop.data.size):
        value = inverse_normal_score(tk_prop.data.flat[i])
        tk_prop.data.flat[i] = value
    return tk_prop


def gtsim_2ind(grid, prop, sk_params=None, sgs_params=None, pk_prop=None):
    # prop must be continuous!

    # F-367: validate the hard-data categories up front, like both siblings
    # (gtsimk.py:103-116 and gtsimk_const_prob.py:110-123). A code outside
    # {0, 1} was left untouched by the transform and then RE-CLASSIFIED by the
    # truncation threshold (live probe: a hard value of 2 came back as facies
    # 0, no error, no warning), so the datum was silently discarded instead of
    # conditioning the simulation. Two indicators leave exactly the codes 0
    # and 1.
    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            val = int(prop.data.flat[i])
            if val != 0 and val != 1:
                raise ValueError(
                    "gtsim_2ind: hard-data category out of range — cell "
                    f"{i} has value {val}, expected 0 or 1 (2 indicators)."
                )

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
    # F-364: the artifacts below are CWD-relative results/ paths, so the
    # directory must exist before the first write — otherwise the run dies
    # with `RuntimeError: write_property failed: ... No such file or
    # directory` after the SK step has already run. The creation belongs to
    # the DRIVER, not to this function: gtsim_test.py creates results/
    # before it calls gtsim_2ind. Creating it here made the pinned smoke test
    # (tests/python/test_production_fixes_rest.py, which stubs write_property
    # for "no filesystem side effects") litter the pytest CWD with a
    # results/ directory on every default-suite run.
    write_property(pk_prop, "results/GTSIM_PKPROP.INC", "PKPROP", -99)
    tk_prop = tk_calculation(pk_prop)
    write_property(tk_prop, "results/GTSIM_TKPROP.INC", "TKPROP", -99)
    print("Done.")

    # 3. pseudo gaussian transform of initial property (prop) with pk_prop
    print("Pseudo gaussian transform of initial property (hard data)...")
    prop_pg = pseudo_gaussian_transform(prop, tk_prop)
    # E-M12: pseudo_gaussian_transform deep-copies prop via _clone_prop and
    # mutates only the clone — the caller's `prop` object is never modified.
    # The previous write_property(prop, ...) wrote the UNTRANSFORMED hard
    # data into GTSIM_TRANSFORMED_PROP.INC on every run (the sibling
    # gtsimk_const_prob.py writes its own transformed property). Write the
    # transformed clone instead.
    # G-36: under its own name — the sibling gtsimk_const_prob.py used to
    # write the same results/GTSIM_TRANSFORMED_PROP.INC path with the same
    # "TRANSPROP" title for the K-indicator transform, so either producer's
    # artifact could overwrite the other's. Each producer now owns one name:
    # this one writes GTSIM_TRANSFORMED_PROP_2IND.INC, the sibling writes
    # GTSIM_CONSTPROB_TRANSFORMED_PROP.INC.
    write_property(prop_pg, "results/GTSIM_TRANSFORMED_PROP_2IND.INC", "TRANSPROP_2IND", -99)
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
    # E2-41: the SGS output prop_sgs lives in DATA space — the C++
    # sequential_gaussian_simulation back-transforms the simulated
    # standard-normal field through the in-scope empirical CDF
    # (transform_cdf_p at sequential_gaussian_simulation.cpp:165), while
    # tk_prop holds NORMAL-SCORE thresholds Φ⁻¹(pk). Comparing across
    # spaces misclassifies every cell (realized facies proportions
    # compressed toward the marginal, 0.77 vs 0.50 — E2-41 trace). Map
    # each threshold through the SAME empirical CDF used by the
    # back-transform (port of the library gtsim_2ind F-02 fix).
    # F-373: the SIGN is mirrored here because the library's tk_calculation
    # returns mean - std_dev·Φ⁻¹(p) (src/geo_bsd/gtsim.py:101-143), i.e.
    # -Φ⁻¹(pk), while this script's tk_calculation returns +Φ⁻¹(pk)
    # (gaussian_cdf.inverse_normal_score). With the mirror the threshold is
    # F⁻¹(Φ(-tk)) = F⁻¹(1-pk), exactly the value the library compares against
    # (`prop1_flat >= tk_data -> 1`, src/geo_bsd/gtsim.py:353-355), so
    # P(facies 1) = pk here as it is there; before the mirror the two
    # implementations classified complementary outcomes (measured with a
    # constant pk: 0.634 facies 1 here vs 0.398 in the library).
    tk_data = cdf.inverse(_norm_cdf(-tk_prop.data.flat[:]))
    # II-43: two independent `if`s re-compared the OVERWRITTEN value. When
    # tk <= 0 (common: any pk <= 0.5 gives tk = -Φ⁻¹(pk) <= 0), a cell set
    # to 0 by the first if was immediately re-set to 1 by the second
    # (`0 >= tk` is True for tk <= 0) → all-facies-1 output (live probe:
    # tk=-0.5 → all facies 1; PRIOR_FIX_ATTEMPT 6b2d2cd). Use if/elif so
    # each cell is classified exactly once.
    # P-02: preserve hard-data facies through the truncation. At hard cells
    # SK is an exact interpolator → pk = 0.0/1.0 exactly → tk = ∓10 (clamp,
    # gaussian_cdf.py:17-20) → the pseudo-Gaussian transform's draw is
    # DEGENERATE (uniform(0,0) / uniform(1,1) in probability space) → SGS
    # reproduces the conditioning value exactly → prop_sgs == tk. The strict
    # `<` below maps that EQUALITY case to facies 1, so every facies-0 hard
    # cell was corrupted to 1
    # (gtsim_test.py hard-data check: 2540/3002 errors; 2540 == facies-0
    # count exactly). Cells whose ORIGINAL prop value is a facies (0 or 1 —
    # non-hard cells hold nodata -99, and the transform already assumes
    # facies are exactly 0/1) keep that facies unconditionally; only
    # non-hard cells are classified by the threshold.
    for i in range(prop_sgs.data.size):
        orig = prop.data.flat[i]
        if orig == 0 or orig == 1:
            prop_sgs.data.flat[i] = orig
        elif prop_sgs.data.flat[i] < tk_data[i]:
            prop_sgs.data.flat[i] = 0
        else:
            prop_sgs.data.flat[i] = 1
    print("Done.")
    print("GTSIM: Finished.")
    return prop_sgs
