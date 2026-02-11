import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property,
)

from gtsimk import gtsim_Kind

# gtsim for k indicators test
prop = load_cont_property("test_data/BIG_SOFT_DATA_160_141_20.INC", -99, (166, 141, 20))
grid = SugarboxGrid(166, 141, 20)

cov = CovarianceModel(type=covariance.exponential, ranges=(10, 10, 10), sill=1)

sk_params = {
    "radiuses": (20, 20, 20),
    "max_neighbours": 12,
    "cov_model": cov,
    "mean": 1.6,
}

indicator = 3
gtsim_Kind(grid, prop, indicator, sk_params)
input("Press Enter to continue...")
