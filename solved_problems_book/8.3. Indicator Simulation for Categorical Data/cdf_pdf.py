#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 8.3
#	"Indicator Simulation for Categorical Data"
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------
#	Problem:
#
#	Drawing 100 realizations from the cdf yields 100 realizations of the rock type category. A distribution of uncertainty constructed for these realizations should look similar to the distributions of the conditional pdf.
#
# ----------------------------------------------------

# rock type
rock_type = np.array([2, 2, 3, 3, 1])

# number of realizations
n = 100

array_hist = np.zeros((n), order='F', dtype=int)

for i in range(n):
	index = np.random.randint(0, len(rock_type), 1)
	array_hist[i] = rock_type[index]

prob = []
value = []

for i in range(min(rock_type), max(rock_type) + 1):
	val = array_hist.compress((array_hist == i).flat)
	prob.append((float(len(val)) / (len(array_hist))))
	value.append(i)

# Histogram of the conditional pdf
plt.figure()
for i in range(3):
	plt.bar(value[i], prob[i], width=0.33)
plt.title("pdf")
plt.xlabel("Rock Type")
plt.ylabel("f")

# Histogram of the cdf derived from indicator kriging of the rock type data
plt.figure()
plt.hist(array_hist, cumulative=True, density=True)
plt.title("cdf")
plt.xlabel("Rock Type")
plt.ylabel("F")

plt.show()
