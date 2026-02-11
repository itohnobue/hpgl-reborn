import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import matplotlib.pyplot as plt
from grid_3d import *

# NOTE: simple_kriging_weights is not available in the new geo_bsd API.
# The w_kriging function below will not work without it.
# Consider using geo_bsd.simple_kriging() instead for kriging estimates.

# Inverse distance weighting calculation
def w_idw(Grid, PointSet, c, nx, ny, nz):
	widw = np.zeros(len(PointSet[0]), dtype=float)
	for i in range(Grid.i_max):
		for j in range(Grid.j_max):
			for k in range(Grid.k_max):
				x_center, y_center, z_center = get_center_points(i, j, k, nx, ny, nz, min(PointSet[0]), min(PointSet[1]), min(PointSet[2]))
				ww_idw = get_weights_idw(Grid, x_center, y_center, z_center, PointSet, c)
				for q in range(len(PointSet[0])):
					widw[q] = widw[q] + ww_idw[q]
	return widw

# Kriging weights calculation (requires simple_kriging_weights - not available in geo_bsd)
# def w_kriging(Grid, PointSet):
# 	...

#Drawing bar
def bar_show(w_cell, wsk, widw, x):
	ind = np.arange(x)
	for i in range(len(wsk)):
		p1 = plt.bar(i, widw[i], color='y', width=0.3)
		p2 = plt.bar(i+0.3, w_cell[i], width=0.3)
		p3 = plt.bar(i+0.6, wsk[i], color='r', width=0.3)
		plt.bar(i+0.8, 0.0, color='w', width=0.2)
	plt.legend((p1[0], p2[0], p3[0]), ('IDW', 'Cell', 'Kriging'), loc='upper left')
	plt.xlabel("Number of data")
	plt.ylabel("Standardized weights")
	plt.title("Comparison of Declustering Methods")
	plt.xticks(ind+0.4, ('1','2','3','4','5','6','7','8','9','10','11','12','13','14'))
	plt.show()
