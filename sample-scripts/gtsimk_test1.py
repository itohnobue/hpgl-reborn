import sys
import os

# F-355: the package lives in src/geo_bsd (readme.md:269-274), not at the
# repository root — inserting '..' left `import geo_bsd` raising
# ModuleNotFoundError before any script logic ran.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from geo_bsd import (
    SugarboxGrid, CovarianceModel, covariance,
    load_cont_property,
)

from gtsimk_const_prob import gtsim_Kind_const_prop

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

# gtsim for k indicators test (const prob variant)
prop = load_cont_property(
    os.path.join(TEST_DATA_DIR, "BIG_SOFT_DATA_160_141_20.INC"), -99, (166, 141, 20),
    basedir=REPO_ROOT,
)
grid = SugarboxGrid(166, 141, 20)

cov = CovarianceModel(type=covariance.spherical, ranges=(10, 10, 10), sill=1)

sk_params = {
    "radiuses": (20, 20, 20),
    "max_neighbours": 12,
    "cov_model": cov,
}

indicator = 3
gtsim_Kind_const_prop(grid, prop, indicator, sk_params)
# F-353: prompt only when a terminal is attached. The unconditional input()
# raised EOFError (exit status 1) on non-interactive stdin (CI, nohup, ssh
# without a pty) after the whole pipeline had already run, so a finished run
# read as a failure; on an idle terminal it blocked indefinitely.
if sys.stdin.isatty():
    input("Press Enter to continue...")
