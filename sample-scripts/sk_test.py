import sys
import os
import numpy as np
from matplotlib import pyplot as plt

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
    # F-54: the repo has CUBE.INC (not CUB.INC); the old name resolved to
    # nothing and the workflow failed with CriticalValidationError. Use the
    # real file in this script's directory.
    prop1 = load_cont_property(
        os.path.join(os.path.dirname(__file__), "CUBE.INC"), -99, (x, y, z)
    )
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
    # E-M16: the old metric accumulated the SIGNED difference and applied a
    # single abs() afterwards — abs(Σdiff) ≠ Σ|diff|, so cancellation made
    # the "sensitivity vs max_neighbours" curve meaningless. Accumulate the
    # per-cell absolute difference instead (Σ|diff|).
    razn = np.zeros(i)
    max_n = load_property_python(x, y, z, "RES0.INC", True)
    for j in range(i):
        min_n = load_property_python(x, y, z, "RES" + str(i - j) + ".INC", True)
        for a in range(x):
            for b in range(y):
                for c in range(z):
                    razn[j] = razn[j] + abs(max_n[a, b, c] - min_n[a, b, c])
    print(razn)
    mas = np.zeros(i)
    for f in range(i):
        n = n + 1
        mas[f] = mas[f] + n
    plt.plot(mas, razn)
    plt.ylabel("D")
    plt.xlabel("max neighbours")
    plt.show()


if __name__ == "__main__":
    # E-H3: sk_calc() was never invoked — running the script was a silent
    # no-op. Run the documented workflow: CUBE.INC in this script's
    # directory is the sample cube (286×10×1 = 2860 cells).
    sk_calc(286, 10, 1, None)
