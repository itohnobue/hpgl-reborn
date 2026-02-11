import sys
import os
import numpy as np
from pylab import plot, ylabel, xlabel, show

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property, write_property, simple_kriging,
)
from python_property import load_property_python, save_property_python


def sk_calc(x, y, z, prop1):
    print("Creating Grid... ")
    grid = SugarboxGrid(x, y, z)
    print("Done.\n")
    print("Loading property... ")
    prop1 = load_cont_property("CUB.INC", -99, (x, y, z))
    print("Done.\n")

    cov = CovarianceModel(
        type=covariance.exponential,
        ranges=(10, 10, 10),
        sill=1,
    )

    i = -1
    n = 60
    while n > 10:
        i = i + 1
        prop_result = simple_kriging(
            prop1, grid,
            radiuses=(20, 20, 20),
            max_neighbours=n,
            cov_model=cov,
            mean=0.487,
        )
        write_property(prop_result, "RESULT_SK" + str(i) + ".INC", "SK_RESULT" + str(i), -99)
        values_result = load_property_python(x, y, z, "RESULT_SK" + str(i) + ".INC", True)
        save_property_python(values_result, x, y, z, "RES" + str(i) + ".INC")
        n = n - 1
    razn = np.zeros(i)
    max_n = load_property_python(x, y, z, "RES0.INC", True)
    for j in range(i):
        min_n = load_property_python(x, y, z, "RES" + str(i - j) + ".INC", True)
        for a in range(x):
            for b in range(y):
                for c in range(z):
                    razn[j] = max_n[a, b, c] - min_n[a, b, c] + razn[j]
        razn[j] = abs(razn[j])
    print(razn)
    mas = np.zeros(i)
    for f in range(i):
        n = n + 1
        mas[f] = mas[f] + n
    plot(mas, razn)
    ylabel("D")
    xlabel("max neighbours")
    show()
