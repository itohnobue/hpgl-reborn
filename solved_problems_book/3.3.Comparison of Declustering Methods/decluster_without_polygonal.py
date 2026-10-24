import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
# F-47 pattern: geo_bsd lives in the repo's src/ directory (it is not
# installed in the environment); without this the
# `from geo_bsd import simple_kriging_weights` below fails with
# ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from grid_3d import *

# E-M27: simple_kriging_weights IS available in geo_bsd (geo.py:2394,
# exported from geo_bsd.__init__:150) — the old comment claiming it is
# "not available in the new geo_bsd API" was false.
from geo_bsd import simple_kriging_weights

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

# F-094: the kriging-declustering covariance range is a MODELING CHOICE, not a
# measured property: with the model held fixed, per-datum weights change 2-4x
# between ranges 100 and 1000 (datum 2: 0.536 -> 0.192 -> 0.147; datum 3:
# 1.025 -> 0.397 -> 0.261) and the heaviest datum changes identity. 300 is not
# derived from the data (the nearest-neighbour spacings are 25-124, median 49)
# and no variogram fitted in the lesson is consulted — it is stated here as a
# named constant so the printed weights are reproducible and the choice is
# visible at the point of use (the same constant decluster.py declares).
KRIGING_RANGE = 300.0

# Kriging declustering: for every grid-cell centre, compute the simple kriging
# weights of all data points at that centre (geo_bsd.simple_kriging_weights) and
# accumulate them. A data point that kriges many cells (or kriges cells with
# large weights) receives a large total weight — its "influence" over the
# domain. The covariance model is geo_bsd's default (exponential, sill 1,
# nugget 0) with every range set to KRIGING_RANGE — see the constant above for
# how strongly that choice moves the published weights. Weights are
# standardized by the caller (stand_weight).
def w_kriging(Grid, PointSet, ranges=None):
	if ranges is None:
		ranges = (KRIGING_RANGE, KRIGING_RANGE, KRIGING_RANGE)
	wsk = np.zeros(len(PointSet[0]), dtype=float)
	for i in range(Grid.i_max):
		for j in range(Grid.j_max):
			for k in range(Grid.k_max):
				x_center, y_center, z_center = get_center_points(i, j, k, Grid.nx, Grid.ny, Grid.nz, min(PointSet[0]), min(PointSet[1]), min(PointSet[2]))
				ww = simple_kriging_weights((x_center, y_center, z_center), PointSet[0], PointSet[1], PointSet[2], ranges=ranges)
				wsk = wsk + ww
	return wsk

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
