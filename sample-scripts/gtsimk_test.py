import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property,
)

from gtsimk import gtsim_Kind

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location (works regardless of CWD).
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))

# gtsim for k indicators test
prop = load_cont_property(
    os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_160_141_20.INC"), -99, (166, 141, 20)
)
grid = SugarboxGrid(166, 141, 20)

cov = CovarianceModel(type=covariance.exponential, ranges=(10, 10, 10), sill=1)

sk_params = {
    "radiuses": (20, 20, 20),
    "max_neighbours": 12,
    "cov_model": cov,
    # E-M14: the SK mean must be a valid probability for 0/1 indicator
    # data (BIG_SOFT_DATA_160_141_20.INC holds only {-99, 0, 1}). The old
    # mean=1.6 made no-neighbour cells get pk=1.6 and Σw<1 cells pk>1 →
    # inverse_normal_score clamps at 10 → sparse regions collapsed to
    # facies 0. 0.5 is the data's marginal probability.
    "mean": 0.5,
}

indicator = 3
gtsim_Kind(grid, prop, indicator, sk_params)
input("Press Enter to continue...")
