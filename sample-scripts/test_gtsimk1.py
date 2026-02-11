import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import load_cont_property


def test_gtsim(prop1, prop2):
    n = 0
    prop_size = 0
    for i in range(prop1.data.size):
        if prop1.mask.flat[i] > 0:
            prop_size = prop_size + 1
            if prop1.data.flat[i] != prop2.data.flat[i]:
                n = n + 1
    if n > 0:
        print("error", n)
    if n == 0:
        print("ok")
    print(prop_size)


def ind_ver(prop, indicator):
    ind_count = np.zeros(indicator, dtype=float)
    ind_prob = np.zeros(indicator, dtype=float)
    prop_size = 0
    for i in range(prop.data.size):
        for j in range(indicator):
            if prop.data.flat[i] == j:
                ind_count[j] = ind_count[j] + 1.0
        if prop.mask.flat[i] > 0:
            prop_size = prop_size + 1.0
    for j in range(indicator):
        ind_prob[j] = ind_count[j] / prop_size
    print(ind_prob)


prop1 = load_cont_property("test_data/BIG_SOFT_DATA_160_141_20.INC", -99, (166, 141, 20))
prop2 = load_cont_property("results/GTSIM_TRUNC_RESULT.INC", -99, (166, 141, 20))
test_gtsim(prop1, prop2)
ind_ver(prop2, 3)
ind_ver(prop1, 3)
input("Press Enter to continue...")
