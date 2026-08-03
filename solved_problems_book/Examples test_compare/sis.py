import os
import sys
import time
# geo_bsd lives in the repo's src/ directory (it is not installed in the
# environment); without this the `from geo_bsd import *` fails.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from geo_bsd import *
from geo_bsd.geo import covariance
import numpy as np
import matplotlib.pyplot as plt

size = (166, 141, 1)
grid = SugarboxGrid(166, 141, 1)

# F-16 chain: IND_data.INC is written by make_prop.py with the byte
# sentinel 255 — the reader must use the same undefined_value.
data = load_ind_property("IND_data.INC", 255, [0, 1], size)

cov1 = CovarianceModel(type=1, ranges=(20, 20, 1), sill=1)

ik_data = [{
			"cov_model": cov1,
            "radiuses": (40, 40, 1),
            "max_neighbours": 12,
            },
            {
			"cov_model": cov1,
            "radiuses": (40, 40, 1),
            "max_neighbours": 12,
            }]

sis_result = sis_simulation(prop=data, grid=grid, data=ik_data, marginal_probs=(0.8, 0.2), seed=3241347)

plt.figure()
plt.imshow(data[0][:, :, 0], vmin=0, vmax=2)
plt.savefig("hard_data")

plt.figure()
plt.hist(data[0].compress((data[0] != -99).flat), bins=20)
plt.title("Histogram of Harddata")

plt.figure()
plt.imshow(sis_result[0][:, :, 0], vmin=0, vmax=2)
plt.savefig("SIS_result")

plt.figure()
plt.hist(sis_result[0].compress((sis_result[0] != -99).flat), bins=20)
plt.title("Histogram of SIS Result")
plt.show()
