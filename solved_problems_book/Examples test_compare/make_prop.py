import os
import time
from geo_bsd import *
import numpy as np
import matplotlib.pyplot as plt

size = (166, 141, 20)

print("loading image...")
data_3D = load_ind_property("BIG_SOFT_DATA_160_141_20.INC", -99, [0, 1], size)

data = data_3D[0][:, :, 0]
mask = data_3D[1][:, :, 0]

plt.figure()
plt.imshow(data[:, :], vmin=0, vmax=2)
plt.savefig("hard_data")

prop = (data, mask, 2)
write_property(prop, "IND_data.INC", "Ind_data", -99)
