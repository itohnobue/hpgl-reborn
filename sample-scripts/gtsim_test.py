import sys
import os
import time

# F-355: the package lives in src/geo_bsd (readme.md:269-274), not at the
# repository root — inserting '..' left `import geo_bsd` raising
# ModuleNotFoundError before any script logic ran.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property, write_property, calc_mean,
)

from gtsim import gtsim_2ind

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location.
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# F-362: the fixture path is absolute, so the load must be told which
# directory contains it — PathValidator's default base is the process CWD
# and rejects this path when the script is started from sample-scripts/.
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))

if not os.path.exists("results/"):
    os.mkdir("results/")

# gtsim test
time1 = time.time()
prop = load_cont_property(
    os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_160_141_20.INC"), -99, (166, 141, 20),
    basedir=REPO_ROOT,
)
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
prop_init = load_cont_property(
    os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_160_141_20.INC"), -99, (166, 141, 20),
    basedir=REPO_ROOT,
)

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
