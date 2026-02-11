#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 4.3
#	"Transfer of Uncertainty"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import matplotlib.pyplot as plt
from statistics import *
from scipy.interpolate import interp1d

# Mean and variance of annual revenue, R.
mean_R = 10
var_R = 2

# Mean and variance of annual costs, C.
mean_C = 8
var_C = 2

# Constant discount factor, r.
r = 0.1

# Number of years.
n = 5

# Number of realizations.
l = 1000

# Function for calculating NPV:
def npv_calculate(R, C, r, n):
	npv = 0.0
	for i in range(n):
		npv = npv + ((R[i] - C[i]) / (1 + r)**i)
	return npv

npv_array = []

# Repeat l times to assemble the distribution for NPV.
for i in range(l):
	#Simulate random R
	R = np.random.normal(mean_R, np.sqrt(var_R), n)

	#Simulate random C
	C = np.random.normal(mean_C, np.sqrt(var_C), n)

	npv_array.append(npv_calculate(R, C, r, n))
print("NPV", npv_array)

[array_bins, array_hist] = np.histogram(npv_array, l)

for i in range(l-1):
	array_bins[i+1] = array_bins[i+1] + array_bins[i]

array_bin = np.zeros(l, dtype=float)

for i in range(l):
	array_bin[i] = float(array_bins[i]) / l

array_hist = np.delete(array_hist, [0])
z = []

# Interpolation
int_obj = interp1d(array_bin, array_hist, kind='linear')
z.append(int_obj(0.1))
z.append(int_obj(0.5))
z.append(int_obj(0.9))
print("P10 = ", z[0])
print("P50 = ", z[1])
print("P90 = ", z[2])

# Calculate the probability of a negative NPV.

negative_npv = 0.0
for i in range(l):
	if (npv_array[i] < 0):
		negative_npv = negative_npv + 1.0

prob_npv = negative_npv / l
print("Probability of negative NPV = ", prob_npv)

plt.figure()
plt.hist(npv_array, 50, density=True)
plt.xlabel("NPV")
plt.ylabel("Frequency")
plt.title("Histogram of simulated NPV")

plt.figure()
plt.plot(array_hist, array_bin)
plt.xlabel("NPV")
plt.ylabel("Cumulative Frequency")
plt.title("Cumulative histogram of simulated NPV")
plt.show()
