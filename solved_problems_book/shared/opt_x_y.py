import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from decl_grid import *
from statistics import *


def optimal_dx_dy(array1, array2, array3, d_x, d_y, min_max, x):
	w_m = []
	dx_ar = []
	dy_ar = []
	for dx in range(1, d_x):
		for dy in range(1, d_y):
			l1 = (min_max[2] - min_max[0])/dx
			l2 = (min_max[3] - min_max[1])/dy

			array_grid = Grid(min_max[0], min_max[1], dx, dy, l1, l2)

			for i in range(x):
				array_grid.add_point(array1[i], array2[i])

#cell_declustering

			w_cell = array_grid.get_weights_cell()
			w_cell = stand_weight(w_cell, x)
			w_m.append(w_mean(w_cell, array3))
			dx_ar.append(dx)
			dy_ar.append(dy)

	w_min = min(w_m)
	for i in range(len(w_m)):
		if (w_m[i] == w_min):
			i_min = i
	return dx_ar[i_min], dy_ar[i_min]
