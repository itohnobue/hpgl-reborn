import sys
import os
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property, write_property,
)
from geo_bsd.sgs import sgs_simulation
from geo_bsd.cdf import calc_cdf
from python_property import load_property_python

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location (works regardless of CWD).
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))


def mean_calc_hist(x, y, z, n, prop):
    print("Creating Grid... ")
    grid = SugarboxGrid(x, y, z)
    print("Done.\n")
    print("Loading property... ")
    # F-53: the old path "NEW_TEST_PROP.INC" did not exist anywhere in the
    # repo (deleted 6e6ae94, restored verbatim 761f9d5) and the workflow
    # failed with CriticalValidationError. The real data file is
    # NEW_TEST_PROP_01.INC under tests/python/test_data/.
    # E2-50: NEW_TEST_PROP_01.INC has only two informed unique values (0/1),
    # so calc_cdf produces a 2-point CDF and the normal-score transform
    # standardizes to variance ≈7.3 against the model sill 1.0 — realization
    # means came out high-biased (E[value]≈0.96 vs data mean 0.55). Use the
    # repo's continuous sample property (many unique values → a proper
    # multi-point empirical CDF, the library transform's intended input).
    prop = load_cont_property(
        os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_CON_160_141_20.INC"), -99, (x, y, z)
    )
    print("Done.\n")

    cov = CovarianceModel(
        type=covariance.exponential,
        ranges=(10, 10, 10),
        sill=0.4,
    )

    ntg1 = np.empty(n)
    cdf = calc_cdf(prop)
    for c in range(n):
        print("Done.\n")
        sgs_result_prop = sgs_simulation(
            prop, grid, cdf,
            radiuses=(20, 20, 20),
            seed=3141347 - 1000 * c + 500,
            max_neighbours=12,
            cov_model=cov,
        )
        write_property(sgs_result_prop, "RSGS.INC", "SGS_PROP", -99)
        values_result = load_property_python(x, y, z, "RSGS.INC", True)

        all_r = 0.0
        numb = 0.0

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    all_r = all_r + values_result[i, j, k]
                    numb = numb + 1
        ntg1[c] = all_r / numb
        print(ntg1[c])
        del sgs_result_prop
    plt.hist(ntg1, histtype='bar', orientation='vertical')
    plt.show()


if __name__ == "__main__":
    # E2-46: mean_calc_hist() was never invoked — running the script was a
    # silent no-op. Run the documented workflow: 10 SGS realizations of the
    # continuous sample property (166×141×20 = 468,120 cells) and histogram
    # the realization means. The E2-50 fix (continuous data → multi-point
    # CDF) makes this resurrected workflow statistically sound.
    mean_calc_hist(166, 141, 20, 10, None)
