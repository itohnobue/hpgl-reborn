import os
import time
from geo_bsd import *
from geo_bsd.geo import covariance
import numpy as np
import matplotlib.pyplot as plt

size = (166, 141, 1)
grid = SugarboxGrid(166, 141, 1)

data = load_ind_property("IND_data.INC", -99, [0, 1], size)

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

ik_result = indicator_kriging(prop=data, grid=grid, data=ik_data, marginal_probs=(0.8, 0.2))

plt.figure()
plt.imshow(data[0][:, :, 0], vmin=0, vmax=2)
plt.savefig("hard_data")

plt.figure()
plt.hist(data[0].compress((data[1] != 0).flat), bins=20)
plt.title("Histogram of Harddata")

plt.figure()
plt.imshow(ik_result[0][:, :, 0], vmin=0, vmax=2)
plt.savefig("IK_result")

plt.figure()
plt.hist(ik_result[0].compress((ik_result[0] != -99).flat), bins=20)
plt.title("Histogram of Indicator Kriging Result")
plt.show()
