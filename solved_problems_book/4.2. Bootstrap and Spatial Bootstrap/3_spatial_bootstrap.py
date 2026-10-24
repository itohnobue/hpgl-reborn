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
from numpy import sqrt
from statistics import *
from gslib import *
from gaussian_cdf import *
from scipy.linalg import cholesky
from scipy.interpolate import interp1d

def cov(h):
	if (h == 0):
		return 1.0
	elif (h > 1):
		return 0.0
	else:
		return (0.9*(1.0 - 1.5*(h) + 0.5*(h**3)))

def get_distance(x1, y1, x2, y2, a_x, a_y):
	return sqrt(((x1 - x2)/a_x)**2 + ((y1 - y2)/a_y)**2)

def spatial_bootstrap(x_num, y_num, poro_values, l):
	A = np.zeros(shape=(x_num, x_num))

	# Set covariance values
	for i in range(x_num):
		for j in range(x_num):
			h = get_distance(x_coord[i], y_coord[i], x_coord[j], y_coord[j], a_x, a_y)
			A[i, j] = cov(h)

	#Calculate lower triangular matrix by a Cholesky LU decomposition
	L = cholesky(A, lower=True, overwrite_a=False)

	#Vector of independent Gaussian values
	w = np.random.randn(x_num)

	#Matrix multiplication
	y = np.dot(L, w)

	to_delete = []
	for i in range(len(y)):
		if (i > l):
			to_delete.append(i)
	y = np.delete(y, to_delete)

	[x, y1] = np.histogram(poro_values, bins=15)

	for i in range(len(x)-1):
		x[i+1] = x[i+1] + x[i]
	x_float = np.zeros(len(x), dtype=float)

	for i in range(len(x)):
		x_float[i] = float(x[i]) / x_num

	y1 = np.delete(y1, [0])

	x_min = x_float.min()

	# Calculate p (n by 1 vector of probability values [0,1]) with standard
	# normal distribution. normal_score returns the standard-normal CDF, which
	# lies in [0, 1] — the old abs() and the old `p[i] > 1.0` guard were dead
	# code. E-M30: draws with p < x_min (the smallest cumulative frequency,
	# 1/62 on welldata.txt) were DELETED, which biased the bootstrapped means
	# upward on every run. Clamp them onto x_min instead: every draw is kept
	# and the interpolation below stays within [x_float.min(), x_float.max()].
	p = np.zeros(l, dtype=float)
	for i in range(l):
		p[i] = normal_score(y[i])
		if (p[i] < x_min):
			p[i] = x_min

	z = np.zeros(len(p), dtype=float)

	#Interpolation
	int_obj = interp1d(x_float, y1)
	for i in range(len(p)):
		z[i] = int_obj(p[i])
	return z

# -------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------
print("----------------------------------------------------")
print("Loading data & initializing...")

# Loading sample data from file

x_num = 62
y_num = 6

#The range of correlation in the x and y directions
a_x = 8500
a_y = 6500

# Number of realizations
n = 1000
# Number of data to bootstrap
l = 62

data_dict = load_gslib_file("welldata.txt")

x_coord = data_dict['X']
y_coord = data_dict['Y']
poro_values = data_dict['Por']
seismic_values = data_dict['Seis']

print("Done.")
print("----------------------------------------------------")

print("Number of bootstrapped realizations: ", n)
print("Number of points to bootstrap: ", l)
print("----------------------------------------------------")
print("Running bootstrap...")

# E-M28: fixed seed so the random draws (and therefore the bootstrapped
# distribution) are reproducible run to run.
np.random.seed(12345)

z_mean = []
for i in range(n):
	z_mas = spatial_bootstrap(x_num, y_num, poro_values, l)
	z_mean.append(calc_mean_array(z_mas))

# F-104: report the statistics the lesson is about. 4.2's only published
# number is the effective number of independent data n_eff (Eq. 4.2), which
# the script never computed or printed (the histogram was its only output),
# so the spread of the bootstrap mean could not be read off a run either.
# Printed below: the mean and standard deviation of the spatial bootstrap
# mean, and the variance-ratio equivalent sample size
#     n_eff = Var(porosity) / Var(spatial bootstrap mean)
# (equivalently n * Var(classical bootstrap mean) / Var(spatial bootstrap
# mean), since the mean of n independent draws has variance
# Var(porosity)/n). That is the standard definition of an effective sample
# size — the number of independent data whose mean would have the observed
# variance; the book's own derivation is not in the repo, so the value
# printed is this definition's answer, computed from the run's own draws.
z_mean_arr = np.asarray(z_mean, dtype=float)
z_mean_mean = z_mean_arr.mean()
z_mean_std = np.sqrt(calc_quadr_var(z_mean_arr, z_mean_mean))
poro_var = calc_quadr_var(poro_values, poro_values.mean())
print("Mean of the spatial bootstrap mean: ", z_mean_mean)
print("Standard deviation of the spatial bootstrap mean: ", z_mean_std)
print("Effective number of independent data "
	"(Var(porosity) / Var(spatial bootstrap mean)): ", poro_var / z_mean_std ** 2)

print("Done!")
print("Drawing histogram...")

#Draw histogram "Distribution of mean porosity considering spatial correlation between the data"
plt.figure()
plt.hist(z_mean, 15, density=True)
plt.ylabel("Frequency")
plt.show()
