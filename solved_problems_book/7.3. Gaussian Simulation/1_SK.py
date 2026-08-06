#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 7.3
#	"Gaussian Simulation"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
# F-47 pattern: geo_bsd lives in the repo's src/ directory (it is not
# installed in the environment); without this the `from geo_bsd import *`
# below fails with ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from geo_bsd import *
from geo_bsd.geo import covariance
import matplotlib.pyplot as plt
from gslib import *
from grid_3d import *
from gaussian_cdf import *
from variogram_routines import *
from geo_routines import *

#---------------------------------------------------
#	Problem:
#
#	Perform SK of the data over a domain of size 15,000 * 15,000 m using a discretization of 50 * 50 m blocks using the supplied variogram model. Ensure that the search radius used in the kriging is commensurate with the variogram ranges; a good rule of thumb is to use a search radius that is at least the maximum range of the variogram model. Note that the variogram model given is the variogram of the normal score transformed data, so the kriging should be performed on the transformed data. Back transform the kriged estimates in order to compare the histograms, but retain the normal scores kriged estimates to compare the variograms.
#
# ----------------------------------------------------

# number of cells
# E-M38: the problem statement (line 30) specifies a domain of
# 15,000 * 15,000 m discretized at 50 * 50 m -> 300 * 300 cells.
# The previous 200 * 200 grid covered only 10,000 m (2/3 of the
# stated domain), while the sibling 2_var.py upscaled grid covers
# the full 15,000 m — the point-scale model was being compared
# against a larger domain.
i_max = 300
j_max = 300
k_max = 1

# Loading sample data from file
data_dict = load_gslib_file("DBHdata.txt")

x_coord = data_dict['X']
y_coord = data_dict['Y']
z_coord = np.zeros((len(x_coord)), order='F', dtype='uint8')

# property
value = "Por"

# Define 3D grid
grid = SugarboxGrid(i_max, j_max, k_max)

# x, y, z size(m)
nx = 50
ny = 50
nz = 1

# Lets define 3D grid
array_grid = Grid(0, 0, 0, i_max, j_max, k_max, nx, ny, nz)

prop_ijk = np.zeros((i_max, j_max, k_max))
prop_ijk = np.require(prop_ijk, dtype=np.float32, requirements=['F'])

array_defined = np.zeros((i_max, j_max, k_max))
array_defined = np.require(array_defined, dtype=np.uint8, requirements=['F'])

for i in range(i_max):
	for j in range(j_max):
		for k in range(k_max):
			arithmetic_mean = get_sum_cell_value(array_grid, x_coord, y_coord, z_coord, i, j, k, data_dict[value])
			if (arithmetic_mean > 0):
				prop_ijk[i, j, k] = arithmetic_mean
				array_defined[i, j, k] = 1
			else:
				prop_ijk[i, j, k] = -99
				array_defined[i, j, k] = 0

initial_data = np.copy(prop_ijk)

# Transform initial data with cdf_transform function
props, values = cdf_transform(prop_ijk, -99)
transformed_data = prop_ijk

# Generate a property of transformed data for SK and SGS
prop_transformed = (transformed_data, array_defined)

# Generate a property of initial data for SGS
prop_initial = (initial_data, array_defined)

# Show location map of the data
plt.figure()
plt.imshow(prop_transformed[0][:, :, 0], vmin=-3, vmax=3, extent=[0, 15000, 0, 15000])
plt.title("Location map of the data")

# Histogram of sample data from the data set
plt.figure()
plt.hist(initial_data.compress((initial_data != -99).flat))
plt.title("Histogram of sample data from the data set")
plt.xlabel("Sampled value")

# Generate a kriged field by performing SK with 315 azimuth direction
# E-M37: the same variogram model must be used for SK and SGS — the
# problem statement (lines 30, 169) calls for "the supplied variogram
# model" for both, so the kriged/simulated comparison is meaningful.
# The previous SK used an exponential model with z-range 11 while the
# SGS below used spherical (80, 20, 1) — different functional form AND
# different z-range for the same problem.  Aligned to the spherical
# (80, 20, 1) model used by the SGS and by the sibling 2_var.py.
variogram1 = CovarianceModel(type=covariance.spherical, ranges=(80, 20, 1), sill=1, angles=(315, 0, 0))

krigged_transformed_data = simple_kriging(prop=prop_transformed, grid=grid, radiuses=(160, 40, 1), max_neighbours=20, cov_model=variogram1)
print("SK_transformed_data result:", krigged_transformed_data[0])

#---------------------------------------------------
#	Problem:
#
#	Compute the normal scores semivariograms of the gridded kriged estimates and the simulated values and compare these experimental variograms to the input (Gaussian) variogram model.
#
# --------------------------------------------------

# X, Y, Z, Property = Cube2PointSet(krigged_transformed_data[0], array_defined)

# PointSet = (X, Y, np.zeros(len(X)))

# IndicatorData = []
# IndicatorData.append(Property)

# Params = {'HardData':IndicatorData}
# Function = CalcVariogramFunction

# XVariogram, XLagDistance1 = PointSetScanGridStyle(TVVariogramSearchTemplate(
    # LagWidth = 20, LagSeparation = 20, TolDistance = 20, NumLags = 12,
    # Ellipsoid = TVEllipsoid(R1 = 1, R2 = 20, R3 = 1, Azimut = 315, Dip = 0, Rotation = 0)
# ), PointSet, Function, Params)

# Variogram1 = XVariogram[:, 0]
# print("XVariogram:")
# print(Variogram1)

# F-10: np.copy of a ContProperty produces a 0-d object array and
# back_krigged_transformed_data[0] raises IndexError. Copy the underlying
# .data/.mask arrays instead and rebuild the property.
back_krigged_transformed_data = ContProperty(np.copy(krigged_transformed_data[0]), np.copy(krigged_transformed_data[1]))
# Back transform to original data space
back_cdf_transform(back_krigged_transformed_data[0], props, values, -99)

# X, Y, Z, Property = Cube2PointSet(back_krigged_transformed_data[0], array_defined)

# PointSet = (X, Y, np.zeros(len(X)))

# IndicatorData = []
# IndicatorData.append(Property)

# Params = {'HardData':IndicatorData}
# Function = CalcVariogramFunction

# XVariogram, XLagDistance2 = PointSetScanGridStyle(TVVariogramSearchTemplate(
    # LagWidth = 20, LagSeparation = 20, TolDistance = 20, NumLags = 12,
    # Ellipsoid = TVEllipsoid(R1 = 1, R2 = 20, R3 = 1, Azimut = 315, Dip = 0, Rotation = 0)
# ), PointSet, Function, Params)

# Variogram2 = XVariogram[:, 0]
# print("XVariogram:")
# print(Variogram2)

# plt.figure()
# plt.plot(XLagDistance1, Variogram1, 'bo', color = 'blue')
# plt.plot(XLagDistance2, Variogram2, 'bo', color = 'green')
# plt.xlabel("Distance")
# plt.ylabel("Gamma")
# plt.title("Directional semivariogram of kriged points")


#---------------------------------------------------
#	Problem:
#
#	Run a sequential Gaussian simulation program on the same data and domain to generate a single realization. Here too, run one simulation on the normal score transformed data without a back-transform after the simulation to compare the variograms, and run a second simulation on the original data including a back transform to original data space to compare the histograms. Whatever software is used, ensure that the random path between the two realizations in the same (i.e., by using the same random number seed) so that the impact of the path is removed from the comparison. Alternatively, run one simulation using the normal score transformed data, and back transform the entire realization to original units. In this case. a straight back transform using only the transformation table between the original data units and the normal-scores values is correct.
#
# --------------------------------------------------

# Generate a simulated field by performing SGS on transformed property with 315 azimuth direction
variogram2 = CovarianceModel(type=covariance.spherical, ranges=(80, 20, 1), sill=1, angles=(315, 0, 0))
sgs_params = {"cov_model": variogram2, "radiuses": (160, 40, 1), "max_neighbours": 20}
# F-12: sgs_simulation requires cdf_data. The transformed property is
# already in normal-score space, so pass None (no CDF transform).
sgs_transformed_data = sgs_simulation(prop_transformed, grid, seed=542783, cdf_data=None, **sgs_params)
print("SGS_transformed_data result:", sgs_transformed_data[0])

# Generate a simulated field by performing SGS on initial property with 315 azimuth direction
# F-12: the initial (raw) property needs a CdfData so the simulation can
# map raw values <-> normal scores internally (II-01 makes calc_cdf valid).
sgs_initial_data = sgs_simulation(prop_initial, grid, seed=542783, cdf_data=calc_cdf(ContProperty(initial_data, array_defined)), **sgs_params)
print("SGS_initial_data result:", sgs_initial_data[0])

# F-10: same np.copy fix as above — rebuild a ContProperty from copies of
# the .data/.mask arrays before the in-place back transform.
sgs_res = ContProperty(np.copy(sgs_transformed_data[0]), np.copy(sgs_transformed_data[1]))
# Back transform to original data space
back_cdf_transform(sgs_res[0], props, values, -99)

# Simulated conditional realization in normal-score space
plt.figure()
plt.imshow(sgs_transformed_data[0][:, :, 0], vmin=-3, vmax=3, extent=[0, 15000, 0, 15000])
plt.title("Simulated conditional realization in normal-score space")

# Gridded kriged estimates in normal-score transformed space
plt.figure()
plt.imshow(krigged_transformed_data[0][:, :, 0], vmin=-3, vmax=3, extent=[0, 15000, 0, 15000])
plt.title("Gridded kriged estimates in normal-score transformed space")

#---------------------------------------------------
#	Problem:
#
#	Plot the histograms of the back transformed kriged estimates and the simulated values and compare these to the histogram of the data. Note the reduction in the variance of the kriged estimates. Also note the shape changes in the histograms.
#
# ----------------------------------------------------

# Gridded kriged estimates
plt.figure()
plt.hist(back_krigged_transformed_data[0].flat)
plt.xlabel("Kriged estimate")
plt.title("Gridded kriged estimates")

# Simulated grid of back transformed values
plt.figure()
plt.hist(sgs_res[0].flat)
plt.title("Simulated grid of back transformed values")

# Simulated conditional realization in initial
plt.figure()
plt.imshow(sgs_initial_data[0][:, :, 0], vmin=sgs_initial_data[0].min(), vmax=sgs_initial_data[0].max(), extent=[0, 15000, 0, 15000])
plt.title("Simulated conditional realization in initial")

# Simulated grid of initial values
plt.figure()
plt.hist(sgs_initial_data[0].flat)
plt.title("Simulated grid of initial values")

plt.show()
