import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import load_ind_property, write_property
from hpgl_prop_functions import prop2array, array2prop

prop = load_ind_property("test_data/NEW_TEST_PROP_01.INC", -99, [0, 1], (286, 10, 1))

array_p = prop2array(prop, 286, 10, 1, -99)

prop = array2prop(array_p, -99)

write_property(prop, "results/test_new_prop2array.inc", "test_prop2array", -99)
