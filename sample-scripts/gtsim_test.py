import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property, write_property, calc_mean,
)

from gtsim import gtsim_2ind

if not os.path.exists("results/"):
    os.mkdir("results/")

# gtsim test
time1 = time.time()
prop = load_cont_property("test_data/BIG_SOFT_DATA_160_141_20.INC", -99, (166, 141, 20))
grid = SugarboxGrid(166, 141, 20)

cov = CovarianceModel(type=covariance.exponential, ranges=(10, 10, 10), sill=1)

sk_params = {
    "radiuses": (20, 20, 20),
    "max_neighbours": 12,
    "cov_model": cov,
}
sgs_params = {
    "radiuses": (20, 20, 20),
    "max_neighbours": 12,
    "cov_model": cov,
}

result = gtsim_2ind(grid, prop, sk_params, sgs_params)
time2 = time.time()
print("Time:", (time2 - time1), "s.")
write_property(result, "results/GTSIM_BIG_SOFT_RESULT.INC", "GTSIM", -99)

print("Checking result...")
print("Hard data saving test...")
prop_init = load_cont_property("test_data/BIG_SOFT_DATA_160_141_20.INC", -99, (166, 141, 20))

errors_hard = 0
all_points = 0
for i in range(prop_init.data.size):
    if prop_init.mask.flat[i] > 0:
        all_points = all_points + 1
        if prop_init.data.flat[i] != result.data.flat[i]:
            errors_hard = errors_hard + 1

print("Number of points:", all_points)
print("Error points:", errors_hard)
print("Simulated property mean is:", calc_mean(result))
