#
#	Solved problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 3.1
#	"Basic Declustering"
# ------------------------------------------------

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from statistics import *
from gslib import *

# Weight calculation function
def calculate_weights(points, x, n):
	weights = np.zeros(x, dtype=float)
	for i in range(x):
		for j in range(n):
			weights[i] = weights[i]+points[i, j]
	return weights

print("---------------------------------------------------")
print("Loading data and initializing...")

x = 15
y = 3

# Loading data set
data_dict = load_gslib_file("dataset.txt")
dataset_east = data_dict["East"]
dataset_north = data_dict["North"]

# Number of closest data to take in account
n = 4

print("Done.")
print("---------------------------------------------------")

# Creating values vector
values = data_dict["Grade"]

# Original mean and variance
original_mean = values.mean()
original_var = calc_quadr_var(values, original_mean)

print("Original data mean is: ", original_mean)
print("Original data variance is: ", original_var)
print("---------------------------------------------------")

# Calculating distances between points
distances_v = np.zeros((x, x), dtype=float)

for i in range(x):
	for j in range(x):
		x1 = dataset_east[i]
		y1 = dataset_north[i]

		x2 = dataset_east[j]
		y2 = dataset_north[j]

		distances_v[i, j] = calc_distance(x1, y1, x2, y2)

# Sort distances
distances_v.sort()

# Calculate weights using n closest neighbourhoods
weights = calculate_weights(distances_v, x, n)

print("Weights:")
print(weights)
print("---------------------------------------------------")

# Weighted mean and variance
weighted_mean = w_mean(weights, values)
weighted_var = w_var(weights, values, weighted_mean)

print("Weighted data mean is: ", weighted_mean)
print("Weighted data variance is: ", weighted_var)
print("---------------------------------------------------")
