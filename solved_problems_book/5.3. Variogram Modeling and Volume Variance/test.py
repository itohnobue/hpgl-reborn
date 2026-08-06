import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from gslib import *
from variogram_routines import *
from grid_3d import *

# Loading sample data from file
data_dict = load_gslib_file("allwelldata.txt")

# x, y, z size(m)
nx = [454, 555, 909, 1000, 1666]

# property
value = "Por"

# number of cells
i_max = [22, 18, 11, 10, 6]
j_max = [22, 18, 11, 10, 6]
k_max = [2, 3, 8, 10, 27]

# II-48: choose each Z cell size so the grid covers the full data Z range.
# The old fixed nz values covered only ~80 m while the data reach ~395 m —
# 85% of the points were silently outside the grid.
nz = [(max(data_dict['Z']) - min(data_dict['Z'])) / k for k in k_max]

# E-M36: same fix for the Y axis (the II-48 Z fix missed Y). The old
# fixed ny values left y_max in [10068, 10085] m while the data reach
# 10102 m — 55 points (all at Y=10102) were silently dropped in every
# configuration. ceil() guarantees each grid strictly covers max(Y).
ny = [np.ceil((max(data_dict['Y']) - min(data_dict['Y'])) / j) for j in j_max]

for q in range(5):
	# Lets define 3D grid
	array_grid = Grid(min(data_dict['X']), min(data_dict['Y']), min(data_dict['Z']), i_max[q], j_max[q], k_max[q], nx[q], ny[q], nz[q])
	print("X,Y size =", nx[q], ny[q], "Z size =", nz[q])
	print("nx, ny =", i_max[q], "nz =", k_max[q])
	prop_ijk = np.array([])
	i_coord = np.array([])
	j_coord = np.array([])
	k_coord = np.array([])

	for i in range(i_max[q]):
		for j in range(j_max[q]):
			for k in range(k_max[q]):
				arithmetic_mean = get_sum_cell_value(array_grid, data_dict['X'], data_dict['Y'], data_dict['Z'], i, j, k, data_dict[value])
				if (arithmetic_mean > 0):
					i_coord = np.append(i_coord, i)
					j_coord = np.append(j_coord, j)
					k_coord = np.append(k_coord, k)
					prop_ijk = np.append(prop_ijk, arithmetic_mean)

	IndicatorData = []
	IndicatorData.append(prop_ijk)

	# Lets make a PointSet
	PointSet = {}
	PointSet['X'] = i_coord
	PointSet['Y'] = j_coord
	PointSet['Z'] = k_coord
	PointSet['Property'] = prop_ijk

	Params = {'HardData': IndicatorData}
	Function = CalcVariogramFunction

	#Suggested Parameters for Variogram:

	#Azimuth = 0 (Azimut)
	#Dip = 0 (Dip)
	#Lag Distance = (i_max[q]/2) m (LagWidth, LagSeparation)
	#Horizontal Bandwith = (i_max[q]/2) m (R2)
	#Vertical Bandwith = (k_max[q]/2) m (R3)
	#Number of Lags = 6 (NumLags)

	XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
		LagWidth=(i_max[q] / 2), LagSeparation=(i_max[q] / 2), TolDistance=10, NumLags=6,
		Ellipsoid=TVEllipsoid(R1=1, R2=(i_max[q] / 2), R3=(k_max[q] / 2), Azimut=0, Dip=0, Rotation=0)
	), PointSet, Function, Params)

	Variogram_ver = XVariogram[:, 0]
	print("Variogram:")
	print(Variogram_ver)

	# E-H6: GSLIB gammabar — average semivariogram over all pairs of
	# points discretizing the block volume (1/N^2 sum_i sum_j gamma),
	# giving the volume variance for the simple case.  The previous code
	# printed the raw sum of the lag values, which is not a variance at
	# all.  Modeled variogram: reference-faithful 5.3 vertical model
	# (exponential, sill 11, range 35 m — see 3d_variogram.py).
	sill = 11.0
	var_range = 35.0
	disc = 5
	xs = (np.arange(disc) + 0.5) * nx[q] / disc
	ys = (np.arange(disc) + 0.5) * ny[q] / disc
	zs = (np.arange(disc) + 0.5) * nz[q] / disc
	pts = np.array(np.meshgrid(xs, ys, zs)).reshape(3, disc**3).T
	N = len(pts)
	gammab = 0.0
	for i in range(N):
		h = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
		gammab += np.sum(sill * (1.0 - np.exp(-3.0 * h / var_range)))
	gammab /= (N * N)
	print("Gammab = ", gammab)
