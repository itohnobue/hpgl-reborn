#
#	Solved problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 4.1
#	"Impact of the central limit theorem"
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from statistics import *

# Function to generate random uniform values
def get_random_uniform_values(n):
	x = np.zeros(n, dtype=float)
	for i in range(n):
		x[i] = np.random.uniform(0, 1)
	return x

# Since RVs are independent we can calculate variance simplier
def calc_var_sum(std_dev, n):
	return std_dev**2 / n


# Number of random variables
n = 10
# Number of S realizations
S_num = 100

# Generate one realization of random variables and check mean/var
random_variables = get_random_uniform_values(n)

print("---------------------------------------------------")
rand_mean = random_variables.mean()
rand_var = random_variables.var()

print("One random realization mean is: ", rand_mean)
print("One random realization var is: ", rand_var)
print("---------------------------------------------------")

# Calculating summary mean/var for S realizations

# Vector with S realizations
summary_vec = np.zeros(S_num, dtype=float)

for j in range(S_num):
	random_variables = get_random_uniform_values(n)
	for i in range(n):
		summary_vec[j] = summary_vec[j] + random_variables[i]

sum_mean = summary_vec.mean()
sum_var = calc_var_sum(calc_quadr_var(summary_vec, sum_mean), n)

print("Summary mean is: ", sum_mean)
print("Summary variance is:", sum_var)
print("---------------------------------------------------")

print("plotting histograms...")
# Histogram of random variables
plt.figure()
plt.hist(random_variables)
plt.xlabel("Random variables")
plt.ylabel("Number")
plt.title("Histogram of random variables")

# Histogram of summary random variables statistics
plt.figure()
plt.hist(summary_vec)
plt.xlabel("Summary random variables")
plt.ylabel("Number")
plt.title("Histogram of summary random variables statistics")
plt.show()
