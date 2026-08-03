import os
import sys
import time
# geo_bsd lives in the repo's src/ directory (it is not installed in the
# environment); without this the `from geo_bsd import *` fails.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from geo_bsd import *
import numpy as np
import matplotlib.pyplot as plt

size = (166, 141, 20)

print("loading image...")
data_3D = load_ind_property("BIG_SOFT_DATA_160_141_20.INC", -99, [0, 1], size)

# F-15: keep the data 3D — IndProperty rejects 2D arrays. Take the first
# layer as a 3D (166, 141, 1) cube so the written IND_data.INC matches the
# (166, 141, 1) size the reader scripts (indicator_kriging.py / sis.py)
# load with.
data = data_3D[0][:, :, :1]
mask = data_3D[1][:, :, :1]

plt.figure()
plt.imshow(data[:, :, 0], vmin=0, vmax=2)
plt.savefig("hard_data")

prop = (data, mask, 2)
# F-16: byte properties require an undefined_value in [0, 255]; 255 is the
# conventional byte sentinel and does not collide with categories 0/1. The
# readers (indicator_kriging.py / sis.py) must use the same sentinel.
write_property(prop, "IND_data.INC", "Ind_data", 255)
