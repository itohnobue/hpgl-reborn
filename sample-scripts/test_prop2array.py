import sys
import os

# F-355: the package lives in src/geo_bsd (readme.md:269-274), not at the
# repository root — inserting '..' left `import geo_bsd` raising
# ModuleNotFoundError before any script logic ran.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from geo_bsd import load_ind_property, write_property
from hpgl_prop_functions import prop2array, array2prop

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

prop = load_ind_property(
    os.path.join(TEST_DATA_DIR, "NEW_TEST_PROP_01.INC"), -99, [0, 1], (286, 10, 1),
    basedir=REPO_ROOT,
)

array_p = prop2array(prop, 286, 10, 1, -99)

prop = array2prop(array_p, -99)

# F-16: byte properties require an undefined_value in [0, 255]; 255 is the
# conventional byte sentinel and does not collide with categories 0/1.
# F-364: the write target is CWD-relative results/, and nothing in this
# script created it — the write failed with "open_tmp_file_checked: Can't
# open file ... No such file or directory" after the conversion work.
os.makedirs("results", exist_ok=True)
write_property(prop, "results/test_new_prop2array.inc", "test_prop2array", 255)
