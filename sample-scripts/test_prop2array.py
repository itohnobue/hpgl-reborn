import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import load_ind_property, write_property
from hpgl_prop_functions import prop2array, array2prop

# F-54: sample data lives in tests/python/test_data/ — derive the real
# path from this script's location (works regardless of CWD).
# R-06: abspath removes the literal '..' component — PathValidator
# rejects '..' before normalization (CriticalValidationError).
TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tests', 'python', 'test_data'
))

prop = load_ind_property(
    os.path.join(TEST_DATA_DIR, "NEW_TEST_PROP_01.INC"), -99, [0, 1], (286, 10, 1)
)

array_p = prop2array(prop, 286, 10, 1, -99)

prop = array2prop(array_p, -99)

# F-16: byte properties require an undefined_value in [0, 255]; 255 is the
# conventional byte sentinel and does not collide with categories 0/1.
write_property(prop, "results/test_new_prop2array.inc", "test_prop2array", 255)
