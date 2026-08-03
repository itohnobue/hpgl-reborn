import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import load_cont_property, calc_mean

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location (works regardless of CWD).
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))


def test_gtsim(prop1, prop2):
    n = 0
    prop_size = 0
    for i in range(prop1.data.size):
        if prop1.mask.flat[i] > 0:
            prop_size = prop_size + 1
            if prop1.data.flat[i] != prop2.data.flat[i]:
                n = n + 1
                print(prop1.data.flat[i] - prop2.data.flat[i])
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
            prop_size = prop_size + 1
    for j in range(indicator):
        ind_prob[j] = ind_count[j] / prop_size
    print(ind_prob[0], ind_prob[1])


prop1 = load_cont_property(
    os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_160_141_20.INC"), -99, (166, 141, 20)
)
# F-06: the producer (gtsim_test.py) writes results/GTSIM_BIG_SOFT_RESULT.INC;
# the old filename GTSIM_BIG_SOFT_DATA_RESULT.INC was produced by NO script,
# so this documented workflow always failed with CriticalValidationError.
prop2 = load_cont_property("results/GTSIM_BIG_SOFT_RESULT.INC", -99, (166, 141, 20))
test_gtsim(prop1, prop2)
ind_ver(prop2, 2)
ind_ver(prop1, 2)
input("Press Enter to continue...")
