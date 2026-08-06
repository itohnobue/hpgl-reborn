#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 3.3
#	"Comparison of Declustering Methods"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
# F-47 pattern: geo_bsd lives in the repo's src/ directory (it is not
# installed in the environment); decluster.py (imported below) imports
# geo_bsd at module level, so without this the transitive import fails
# with ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from grid_3d import *
from gslib import *
from decluster import *

# ---------------------------------------------------
#	Problem:
#
#	Compute declustering weights using the four methods: Polygonal declustering, Cell declustering, Kriging declustering, Inverse distance weighting.
#
# ----------------------------------------------------

# Number of cells
dx = 5
dy = 5
dz = 1

# for IDW
c = 2

# Loading sample data from file
data_dict = load_gslib_file("decluster.txt")

# Lets make a PointSet tuple
array3 = np.zeros(len(data_dict['Northing']), dtype=float)
PointSet = (data_dict['Northing'], data_dict['Elev.'], array3)

# nx, ny, nz - cells length (extended space along nx, ny)
nx = 10 + (max(data_dict['Northing']) - min(data_dict['Northing']))/dx
ny = 10 + (max(data_dict['Elev.']) - min(data_dict['Elev.']))/dy
nz = (max(array3) - min(array3))/dz

print("Cells Length.")
print("nx:", nx, "ny:", ny, "nz:", nz)

# Lets define 3D grid with dx*dy*dz cells and nx, ny, nz cells length
array_grid = Grid(min(PointSet[0]), min(PointSet[1]), min(PointSet[2]), dx, dy, dz, nx, ny, nz)

#Cell declustering calculation
w_cell = get_weights_cell(array_grid, PointSet)
w_cell = stand_weight(w_cell, len(PointSet[0]))
print("Cell declustering")
print(w_cell)

#Inverse distance weighting calculation
widw = w_idw(array_grid, PointSet, c, nx, ny, nz)
widw = stand_weight(widw, len(PointSet[0]))
print("Inverse Distance Weighting")
print(widw)

# E-M27: kriging declustering — simple_kriging_weights IS available in geo_bsd
# (geo.py:2394, exported from geo_bsd.__init__:150); the old comment claiming
# it was not available was false.
wsk = w_kriging(array_grid, PointSet)
wsk = stand_weight(wsk, len(PointSet[0]))
print("Kriging Weights")
print(wsk)

# E-M27: polygonal declustering — Voronoi cells of the data points, weight =
# cell area (clipped to a finite region around the data).
wp = w_polygonal(PointSet)
wp = stand_weight(wp, len(PointSet[0]))
print("Polygonal declustering")
print(wp)

# Drawing bar (four methods: polygonal, IDW, cell, kriging)
bar_show(w_cell, wsk, widw, len(PointSet[0]), wp)
