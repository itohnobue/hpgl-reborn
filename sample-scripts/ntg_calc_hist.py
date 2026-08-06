import sys
import os
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_ind_property, write_property,
)
from geo_bsd.sis import sis_simulation
from python_property import load_property_python

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location (works regardless of CWD).
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))


def ntg_calc_hist(x, y, z, n, sis_prop):
    print("Creating Grid... ")
    grid = SugarboxGrid(x, y, z)
    print("Done.\n")
    print("Loading property... ")
    sis_prop = load_ind_property(
        os.path.join(TEST_DATA_DIR, "NEW_TEST_PROP_01.INC"), -99, [0, 1], (x, y, z)
    )
    print("Done.\n")

    cov = CovarianceModel(
        type=covariance.spherical,
        ranges=(10, 10, 10),
        sill=0.4,
    )

    ntg = np.empty(n)
    for c in range(n):
        print("Creating SIS params... ")
        sis_data = [
            {
                "cov_model": cov,
                "radiuses": (10, 10, 10),
                "max_neighbours": 12,
                "marginal_prob": 0.5,
                "value": 0,
            },
            {
                "cov_model": cov,
                "radiuses": (10, 10, 10),
                "max_neighbours": 12,
                "marginal_prob": 0.5,
                "value": 1,
            },
        ]
        print("Done.\n")
        # F-05: sis_simulation requires the marginal_probs positional
        # argument (sis.py:67-78 signature); the old call omitted it and
        # raised TypeError (live repro F-05). The two categories here have
        # marginal_prob 0.5 each in the per-indicator data dicts, so pass
        # the equivalent marginal_probs list.
        sis_result = sis_simulation(
            sis_prop, grid, sis_data, seed=3141347 - 1000 * c + 500,
            marginal_probs=[0.5, 0.5],
        )
        # E-H4: write_property on an IndProperty takes the byte path, which
        # requires undefined_value in [0, 255] (geo.py:979-983) — -99 raised a
        # deterministic ValueError. 255 is the conventional byte sentinel and
        # does not collide with categories 0/1 (F-16, same as test_prop2array.py).
        write_property(sis_result, "RESULT.INC", "S_RESULT", 255)
        values_result = load_property_python(x, y, z, "RESULT.INC", True)

        zeros_count = 0.0
        ones = 0.0

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if values_result[i, j, k] == 1:
                        ones = ones + 1
                    if values_result[i, j, k] == 0:
                        zeros_count = zeros_count + 1
        ntg[c] = ones / (ones + zeros_count)
        print(ntg[c])
        del sis_result
    plt.hist(ntg, histtype='bar', orientation='vertical')
    plt.show()


if __name__ == "__main__":
    # E-H4: ntg_calc_hist() was never invoked — running the script was a
    # silent no-op. Run the documented workflow: 10 SIS realizations of the
    # shipped indicator property (286×10×1 = 2860 cells) and histogram the
    # resulting Net-To-Gross ratios.
    ntg_calc_hist(286, 10, 1, 10, None)
