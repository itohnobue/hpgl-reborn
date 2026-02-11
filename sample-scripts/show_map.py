#!/usr/bin/env python3

import sys
import os
from pylab import imshow, show

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import load_cont_property, SugarboxGrid

if len(sys.argv) < 5:
    print("Usage: show_map.py <x> <y> <z> <file>")
    sys.exit()

x = int(sys.argv[1])
y = int(sys.argv[2])
z = int(sys.argv[3])
filename = sys.argv[4]

grid = SugarboxGrid(x, y, z)
prop = load_cont_property(filename, -99, (x, y, z))
prop.fix_shape(grid)

imshow(prop[0][:, :, 0])
show()
