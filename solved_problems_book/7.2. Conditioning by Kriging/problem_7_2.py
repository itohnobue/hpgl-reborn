#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 7.2
#	"Conditioning by Kriging"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from geo_bsd import *
from geo_bsd.geo import covariance
import matplotlib.pyplot as plt
from gslib import *
from grid_3d import *
from gaussian_cdf import *

#---------------------------------------------------
#	Problem:
#
#	By using the 1D dataset, transect.txt, generate a conditional simulation using the conditioning by kriging method and plot the results
#
# ----------------------------------------------------

# number of cells
i_max = 20
j_max = 1
k_max = 1

# Loading sample data from file
data_dict = load_gslib_file("transect.txt")

x_coord = data_dict['X']
value = data_dict['val']

# Define 3D grid
grid = SugarboxGrid(i_max, j_max, k_max)

# x, y, z size(m)
nx = 1 + (max(x_coord) - min(x_coord)) / i_max
ny = 1
nz = 1

array_grid = Grid(min(x_coord) - 5, 0, 0, i_max, j_max, k_max, nx, ny, nz)

array_val = np.array(np.zeros((i_max)), order='F', dtype='float32')
prop_val = np.array(np.zeros((i_max)), order='F', dtype='float32')
array_for_sk = np.array(np.zeros((i_max)), order='F', dtype='float32')
array_val[0:i_max] = -99
prop_val[0:i_max] = -99
array_for_sk[0:i_max] = -99

array_defined = np.array(np.zeros((i_max)), order='F', dtype='uint8')
prop_defined = np.array(np.zeros((i_max)), order='F', dtype='uint8')

x_val = np.array([])
for i in range(i_max):
	x, y, z = get_center_points(i, 0, 0, nx, 1, 1, min(x_coord) - 5, 0, 0)
	x_val = np.append(x_val, x)

for q in range(len(x_coord)):
	i, j, k = array_grid.get_ijk(x_coord[q], 0, 0)
	array_val[i] = value[q]
	array_defined[i] = 1
	x_val[i] = x_coord[q]

# Generate a kriged field of estimates by performing SK on the Gaussian-transformed dataset in transect.txt
prop1 = (np.float32(array_val), array_defined)
variogram1 = CovarianceModel(type=covariance.exponential, ranges=(5, 1, 1), sill=1)

prop_result1 = simple_kriging(prop=prop1, grid=grid, radiuses=(5, 1, 1), max_neighbours=2, cov_model=variogram1)
print("SK result:", prop_result1[0])

# Generate an unconditional Gaussian simulation, retaining the values at the data locations
prop = (np.float32(prop_val), prop_defined)
variogram2 = CovarianceModel(type=covariance.exponential, ranges=(5, 1, 1), sill=1)

sgs_params = {"cov_model": variogram2, "cdf_data": prop1, "force_single_thread": True}
sgs_result = sgs_simulation(prop, grid, radiuses=(2, 1, 1), max_neighbours=2, seed=3244759, **sgs_params)
print("SGS result:", sgs_result[0])

sgsed_harddata = np.array([])
for i in range(i_max):
	if (array_defined[i] == 1):
		array_for_sk[i] = sgs_result[0][i]
		sgsed_harddata = np.append(sgsed_harddata, sgs_result[0][i])

# Perform a second simple kriging using these values
prop2 = (np.float32(array_for_sk), array_defined)

prop_result2 = simple_kriging(prop=prop2, grid=grid, radiuses=(5, 1, 1), max_neighbours=2, cov_model=variogram1)
print("Simple Kriging result:", prop_result2[0])

# Take the difference between the fields of SGS and second SK and the result to kriged field from first SK
diff = np.array([])
final_diff = np.array([])
diff = prop_result2[0] - sgs_result[0]
final_diff = diff + prop_result1[0]
print("Error Field:", diff)
print("Cond.Sim:", final_diff)

# SK of the dataset and an unconditional SGS
plt.figure()
plt.plot(x_coord, value, 'bo')
p1 = plt.plot(x_val, prop_result1[0])
p2 = plt.plot(x_val, sgs_result[0], '--')
plt.legend((p1[0], p2[0]), ('z*(u)', 'U.C.Sim'), loc='upper left')
plt.xlabel("Distance (m)")
plt.ylabel("Value")
plt.title("SK of the dataset and an unconditional SGS")
plt.axis([0.0, 30, -3.0, 2.0])

# Simulated error field obtained by taking the difference between a SK of the values retained from the unconditional simulation at the locations of the data
plt.figure()
plt.plot(x_coord, sgsed_harddata, 'bo')
p1 = plt.plot(x_val, prop_result2[0])
p2 = plt.plot(x_val, sgs_result[0], '--')
p3 = plt.plot(x_val, diff, '-.')
plt.legend((p1[0], p2[0], p3[0]), ('zs*(u)', 'U.C.Sim', 'Error Field'), loc='upper left')
plt.xlabel("Distance (m)")
plt.ylabel("Value")
plt.axis([0.0, 30, -3.0, 2.0])

# Final resultant conditional simulation obtained by adding the realization of the error field to the original kriging of the data
plt.figure()
plt.plot(x_coord, value, 'bo')
p1 = plt.plot(x_val, prop_result1[0])
p2 = plt.plot(x_val, sgs_result[0], '--')
p3 = plt.plot(x_val, diff, '-.')
p4 = plt.plot(x_val, final_diff, 'k')
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('z*(u)', 'U.C.Sim', 'Error Field', 'Cond.Sim'), loc='upper left')
plt.xlabel("Distance (m)")
plt.ylabel("Value")
plt.axis([0.0, 30, -3.0, 2.0])
plt.show()
