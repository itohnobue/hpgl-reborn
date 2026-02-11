import sys
import os
import numpy as np
from pylab import hist, show

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property, write_property,
)
from geo_bsd.sgs import sgs_simulation
from geo_bsd.cdf import calc_cdf
from python_property import load_property_python


def mean_calc_hist(x, y, z, n, prop):
    print("Creating Grid... ")
    grid = SugarboxGrid(x, y, z)
    print("Done.\n")
    print("Loading property... ")
    prop = load_cont_property("NEW_TEST_PROP.INC", -99, (x, y, z))
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
    hist(ntg1, histtype='bar', orientation='vertical')
    show()
