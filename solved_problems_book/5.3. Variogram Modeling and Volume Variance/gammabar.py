#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 5.3
#	"Variogram Modeling and Volume Variance"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import matplotlib.pyplot as plt
from gslib import *
from variogram_routines import *
from grid_3d import *

#---------------------------------------------------
#	Problem:
#
#	Using classical volume variance relations, determine a reasonable block size for geostatistical modeling of this field for a fixed number of 1000 blocks.
#
# ----------------------------------------------------

# Loading sample data from file
data_dict = load_gslib_file("allwelldata.txt")

# x, y, z size(m)
nx = 454

# property
value = "Por"

# number of cells
i_max = 22
j_max = 22
k_max = 2

# II-48: choose the Z cell size so the grid covers the full data Z range.
# The old fixed nz=40 m with k_max=2 covered only 80 m while the data
# reach ~395 m — 85% of the points were silently outside the grid.
nz = (max(data_dict['Z']) - min(data_dict['Z'])) / k_max

# E-M36: same fix for the Y axis (the II-48 Z fix missed Y). The old
# fixed ny=454 with j_max=22 covered only 85 + 22*454 = 10073 m while
# the data reach 10102 m — 55 points (all at Y=10102) were silently
# dropped from the block averaging, the PointSet and the variogram.
# ceil() guarantees the grid strictly covers max(Y) so the points at
# exactly max(Y) are not dropped by the inclusive-boundary round-trip.
ny = np.ceil((max(data_dict['Y']) - min(data_dict['Y'])) / j_max)

# Lets define 3D grid
array_grid = Grid(min(data_dict['X']), min(data_dict['Y']), min(data_dict['Z']), i_max, j_max, k_max, nx, ny, nz)

prop_ijk = np.array([])
i_coord = np.array([])
j_coord = np.array([])
k_coord = np.array([])

for i in range(i_max):
	for j in range(j_max):
		for k in range(k_max):
			arithmetic_mean = get_sum_cell_value(array_grid, data_dict['X'], data_dict['Y'], data_dict['Z'], i, j, k, data_dict[value])
			if (arithmetic_mean > 0):
				i_coord = np.append(i_coord, i)
				j_coord = np.append(j_coord, j)
				k_coord = np.append(k_coord, k)
				prop_ijk = np.append(prop_ijk, arithmetic_mean)

# Lets make a PointSet
PointSet = {}
PointSet['X'] = i_coord
PointSet['Y'] = j_coord
PointSet['Z'] = k_coord
PointSet['Property'] = prop_ijk

IndicatorData = []
IndicatorData.append(prop_ijk)

Params = {'HardData': IndicatorData}
Function = CalcVariogramFunction

#Suggested Parameters for Variogram
#Azimuth = 0 (Azimut)
#Dip = 0 (Dip)
#Lag Distance = 2 (LagWidth, LagSeparation)
#Horizontal Bandwith = 10 (R2)
#Vertical Bandwith = 3 (R3)
#Number of Lags = 5 (NumLags)

XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=2, LagSeparation=2, TolDistance=4, NumLags=5,
    Ellipsoid=TVEllipsoid(R1=10, R2=10, R3=2, Azimut=0, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_ver = XVariogram[:, 0]
print("XVariogram:")
print(Variogram_ver)

# E-H6: GSLIB gammabar — the average semivariogram value over all pairs
# of points discretizing the block volume V:
#     gammabar = (1 / N^2) * sum_i sum_j gamma(|u_i - u_j|)
# where N is the number of discretization points of the block and
# gamma(h) is the modeled semivariogram.  For the simple case (point
# support, no nugget) the volume variance of the block is exactly this
# average:  sigma^2_V = gammabar.
#
# The previous formula summed the 5 experimental lag values and divided
# by (nx*ny*nz)^2 — a volume squared (m^6), not a pair count — which is
# dimensionally invalid and printed ~1.6e-14 instead of an O(sill)
# variance (~10 for this porosity data).
#
# The modeled variogram below is the reference-faithful 5.3 vertical
# model from 3d_variogram.py (Result/3d_variogram.txt): exponential,
# sill 11, range 35 m.
sill = 11.0
var_range = 35.0

def exp_variogram(h):
	return sill * (1.0 - np.exp(-3.0 * h / var_range))

def gammabar_volume(block_nx, block_ny, block_nz, disc=5):
	"""GSLIB gammabar: average gamma over all pairs of points of the
	discretized block volume.  disc points per axis -> N = disc^3."""
	xs = (np.arange(disc) + 0.5) * block_nx / disc
	ys = (np.arange(disc) + 0.5) * block_ny / disc
	zs = (np.arange(disc) + 0.5) * block_nz / disc
	pts = np.array(np.meshgrid(xs, ys, zs)).reshape(3, disc**3).T
	N = len(pts)
	gb = 0.0
	for i in range(N):
		h = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
		gb += np.sum(exp_variogram(h))
	return gb / (N * N)

gammab = gammabar_volume(nx, ny, nz)
print("Gammab (average gamma over block pairs): ", gammab)
print("Volume variance for this block size (simple case): ", gammab)

#Variogram modeling results for the vertical direction
plt.figure()
plt.plot(XLagDistance, Variogram_ver, 'bo')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Variogram modeling results for the vertical direction")
plt.show()
