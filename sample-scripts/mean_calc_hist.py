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
    prop = load_cont_property(
        os.path.join(TEST_DATA_DIR, "NEW_TEST_PROP_01.INC"), -99, (x, y, z)
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
