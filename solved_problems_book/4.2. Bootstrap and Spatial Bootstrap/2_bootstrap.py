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
import matplotlib.pyplot as plt
from grid_3d import *
from statistics import *
from gslib import *

print("----------------------------------------------------")
print("Loading data & initializing...")

# Loading sample data from file
n = 62
l = 1000
dx = 7
dy = 8
dz = 1

data_dict = load_gslib_file("welldata.txt")

poro_values = data_dict['Por']
seismic_values = data_dict['Seis']

print("Done.")
print("----------------------------------------------------")

array3 = np.zeros(len(data_dict['X']), dtype=float)
PointSet = (data_dict['X'], data_dict['Y'], array3)

nx = 10 + (max(data_dict['X']) - min(data_dict['X']))/dx
ny = 10 + (max(data_dict['Y']) - min(data_dict['Y']))/dy
nz = (max(array3) - min(array3))/dz

# Lets define 3D grid with dx*dy*dz cells and nx, ny, nz cells length
array_grid = Grid(min(PointSet[0]), min(PointSet[1]), min(PointSet[2]), dx, dy, dz, nx, ny, nz)

# Correlation coefficient calculation
coef = corr_coef(poro_values, seismic_values)
print("Correlation between porosity and seismic: ", coef)

#Cell declustering calculation
w_cell = get_weights_cell(array_grid, PointSet)

# Weights standardization
w_cell = stand_weight(w_cell, len(data_dict['X']))

# Weighted correlation coefficient calculation
w_coef = w_corr_coef(poro_values, seismic_values, w_cell)
print("Weighted correlation between porosity and seismic: ", w_coef)
print("----------------------------------------------------")

print("Doing boostrap for poro mean and correlation with seismic data... ")
print("Number of random realizations: ", l)

# Doing bootstrap for mean and correlation
mean_poro = []
coef_rand = []
w_coef_rand = []

mean_seis = w_mean(w_cell, seismic_values)
mean_poro_value = w_mean(w_cell, poro_values)

for i in range(l):
	# Sampling arrays _pairwise_ for correlation bootstrap estimation
	[poro_rand, seismic_rand] = rand_arrays(poro_values, seismic_values, n)

	mean_poro_value = poro_rand.mean()
	mean_poro.append(mean_poro_value)

	# Correlation
	coef_rand.append(corr_coef(poro_rand, seismic_rand))
	# Weighted correlation
	w_coef_rand.append(w_corr_coef(poro_rand, seismic_rand, w_cell))

print("Bootstrap calculation completed.")
print("----------------------------------------------------")

print("Drawing histograms...")

#Draw histogram "Distribution of bootstrapped mean for porosity"
plt.figure()
plt.hist(mean_poro, 50, density=True)
plt.xlabel("Porosity (%)")
plt.ylabel("Frequency")
plt.title("Distribution of bootstrapped mean for porosity")

#Draw histogram "Bootstrapped correlation between weighted porosity and collocated seismic"
plt.figure()
plt.hist(w_coef_rand, 50, density=True)
plt.xlabel("Correlation coefficient")
plt.ylabel("Frequency")
plt.title("Bootstrapped correlation between porosity and collocated seismic")

#Draw histogram "Bootstrapped correlation between porosity and collocated seismic"
plt.figure()
plt.hist(coef_rand, 50, density=True)
plt.xlabel("Correlation coefficient")
plt.ylabel("Frequency")
plt.title("Bootstrapped correlation between weighted porosity and collocated seismic")

#Draw cross plot with seismic data
plt.figure()
plt.plot(poro_values, seismic_values, 'bo')
plt.xlabel("Porosity (%)")
plt.ylabel("Seismic")
plt.title("Cross plot porosity vs seismic")
plt.show()
