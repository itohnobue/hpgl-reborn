#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 4.2
#	"Bootstrap & Spatial Bootstrap"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from statistics import *
from grid_3d import *
from gslib import *

print("----------------------------------------------------")
print("Loading data & initializing...")

# Loading sample data from file
# F-090: the cell size is a modelling parameter of the cell declustering, not
# a property of welldata.txt. Lesson 4.2 uses a different cell size in each
# of its scripts: this first step uses dx=dy=10 (shipped answer
# Result/1/1_cell_declusterisation.txt: declustered mean 8.20568), while
# 2_bootstrap.py uses dx=7/dy=8 (shipped answer Result/2/2_bootstrap.txt:
# weighted correlation 0.57062 — 0.59209 at dx=dy=10). Each recorded answer
# is reproducible only at its own cell size, so the two steps report
# statistics of two different declusterings of the same 62 wells; changing
# the cell size here requires the same change there and a regeneration of
# both Result files.
dx = 10
dy = 10
dz = 1

data_dict = load_gslib_file("welldata.txt")

print("Done.")
print("----------------------------------------------------")

array3 = np.zeros(len(data_dict['X']), dtype=float)
PointSet = (data_dict['X'], data_dict['Y'], array3)

nx = 10 + (max(data_dict['X']) - min(data_dict['X']))/dx
ny = 10 + (max(data_dict['Y']) - min(data_dict['Y']))/dy
nz = (max(array3) - min(array3))/dz

# Lets define 3D grid with dx*dy*dz cells and nx, ny, nz cells length
array_grid = Grid(min(PointSet[0]), min(PointSet[1]), min(PointSet[2]), dx, dy, dz, nx, ny, nz)

array4 = data_dict['Por']

#Cell declustering calculation
w_cell = get_weights_cell(array_grid, PointSet)

# Weights standardization
w_cell = stand_weight(w_cell, len(data_dict['X']))

#Calculate porosity mean
por_mean = array4.mean()
print("Porosity mean =", por_mean)

# Calculate porosity standard deviation
por_quadr_var = calc_quadr_var(array4, por_mean)

#Calculate porosity mean with cell declustering
por_cell_mean = w_mean(w_cell, array4)
print("Porosity mean with cell declustering =", por_cell_mean)

# Calculate porosity variance with cell declustering
por_cell_var = w_var(w_cell, array4, por_cell_mean)
print("Porosity variance with cell declustering =", por_cell_var)

print("Difference between means = ", por_mean-por_cell_mean)
