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
ny = 454
nz = 40

# property
value = "Por"

# number of cells
i_max = 22
j_max = 22
k_max = 2

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

#Calculate Gammabar
gammab = 0
for i in range(len(Variogram_ver)):
	gammab = gammab + Variogram_ver[i]
print("Gammab: ", (gammab / ((nx * ny * nz)**2)))

#Variogram modeling results for the vertical direction
plt.figure()
plt.plot(XLagDistance, Variogram_ver, 'bo')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Variogram modeling results for the vertical direction")
plt.show()
