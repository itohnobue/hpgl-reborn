import sys
import os
import numpy as np
from pylab import hist, show

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_ind_property, write_property,
)
from geo_bsd.sis import sis_simulation
from python_property import load_property_python


def ntg_calc_hist(x, y, z, n, sis_prop):
    print("Creating Grid... ")
    grid = SugarboxGrid(x, y, z)
    print("Done.\n")
    print("Loading property... ")
    sis_prop = load_ind_property("NEW_TEST_PROP_01.INC", -99, [0, 1], (x, y, z))
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
        sis_result = sis_simulation(sis_prop, grid, sis_data, seed=3141347 - 1000 * c + 500, use_vpc=False)
        write_property(sis_result, "RESULT.INC", "S_RESULT", -99)
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
    hist(ntg, histtype='bar', orientation='vertical')
    show()
