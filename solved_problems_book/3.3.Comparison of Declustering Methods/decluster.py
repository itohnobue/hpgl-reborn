import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
# F-47 pattern: geo_bsd lives in the repo's src/ directory (it is not
# installed in the environment); without this the
# `from geo_bsd import simple_kriging_weights` below fails with
# ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from grid_3d import *
from scipy.spatial import Voronoi

# E-M27: simple_kriging_weights IS available in geo_bsd (geo.py:2394,
# exported from geo_bsd.__init__:150) — the old comment claiming it is
# "not available in the new geo_bsd API" was false.
from geo_bsd import simple_kriging_weights

# Inverse distance weighting calculation
def w_idw(Grid, PointSet, c, nx, ny, nz):
	widw = np.zeros(len(PointSet[0]), dtype=float)
	for i in range(Grid.i_max):
		for j in range(Grid.j_max):
			for k in range(Grid.k_max):
				x_center, y_center, z_center = get_center_points(i, j, k, nx, ny, nz, min(PointSet[0]), min(PointSet[1]), min(PointSet[2]))
				ww_idw = get_weights_idw(Grid, x_center, y_center, z_center, PointSet, c)
				for q in range(len(PointSet[0])):
					widw[q] = widw[q] + ww_idw[q]
	return widw

# Kriging declustering: for every grid-cell centre, compute the simple kriging
# weights of all data points at that centre (geo_bsd.simple_kriging_weights) and
# accumulate them. A data point that kriges many cells (or kriges cells with
# large weights) receives a large total weight — its "influence" over the
# domain. The covariance model defaults to a spherical-scale range covering
# the data spacing; weights are standardized by the caller (stand_weight).
def w_kriging(Grid, PointSet, ranges=(300, 300, 300)):
	wsk = np.zeros(len(PointSet[0]), dtype=float)
	for i in range(Grid.i_max):
		for j in range(Grid.j_max):
			for k in range(Grid.k_max):
				x_center, y_center, z_center = get_center_points(i, j, k, Grid.nx, Grid.ny, Grid.nz, min(PointSet[0]), min(PointSet[1]), min(PointSet[2]))
				ww = simple_kriging_weights((x_center, y_center, z_center), PointSet[0], PointSet[1], PointSet[2], ranges=ranges)
				wsk = wsk + ww
	return wsk

# Polygonal declustering: partition the plane into Voronoi cells (one per data
# point); a point's declustering weight is the area of its cell, clipped to a
# finite region around the data (unbounded hull cells are closed with a large
# radius box). Weights are standardized by the caller (stand_weight).
def w_polygonal(PointSet, radius=None):
	points = np.column_stack((PointSet[0], PointSet[1]))
	vor = Voronoi(points)
	regions, vertices = voronoi_finite_polygons_2d(vor, radius)
	wp = np.zeros(len(points), dtype=float)
	for i in range(len(points)):
		poly = vertices[regions[i]]
		wp[i] = polygon_area(poly)
	return wp

# Reconstruct infinite Voronoi regions in a 2D diagram to finite regions
# (scipy recipe — see scipy.spatial.Voronoi docs).
def voronoi_finite_polygons_2d(vor, radius=None):
	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")
	new_regions = []
	new_vertices = vor.vertices.tolist()
	center = vor.points.mean(axis=0)
	if radius is None:
		# np.ptp (not the removed ndarray.ptp method) — numpy >= 2.0
		radius = np.ptp(vor.points).max() * 2
	# Construct a map containing all ridges for a given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))
	# Reconstruct infinite regions
	for p1, region_idx in enumerate(vor.point_region):
		vertices = vor.regions[region_idx]
		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]
		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				continue
			t = vor.points[p2] - vor.points[p1]
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])
			midpoint = vor.points[[p1, p2]].mean(axis=0)
			direction = np.sign(np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius
			new_vertices.append(far_point.tolist())
			new_region.append(len(new_vertices) - 1)
		new_regions.append(new_region)
	return new_regions, np.asarray(new_vertices)

# Polygon area by the shoelace formula
def polygon_area(poly):
	x = poly[:, 0]
	y = poly[:, 1]
	return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

#Drawing bar (polygonal is optional — the "without polygonal" variant omits it)
def bar_show(w_cell, wsk, widw, x, wp=None):
	ind = np.arange(x)
	for i in range(len(wsk)):
		if wp is not None:
			p1 = plt.bar(i, wp[i], color='g', width=0.3)
			p2 = plt.bar(i+0.3, widw[i], color='y', width=0.3)
			p3 = plt.bar(i+0.6, w_cell[i], width=0.3)
			p4 = plt.bar(i+0.9, wsk[i], color='r', width=0.3)
		else:
			p1 = plt.bar(i, widw[i], color='y', width=0.3)
			p2 = plt.bar(i+0.3, w_cell[i], width=0.3)
			p3 = plt.bar(i+0.6, wsk[i], color='r', width=0.3)
			plt.bar(i+0.8, 0.0, color='w', width=0.2)
	if wp is not None:
		plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Polygonal', 'IDW', 'Cell', 'Kriging'), loc='upper left')
	else:
		plt.legend((p1[0], p2[0], p3[0]), ('IDW', 'Cell', 'Kriging'), loc='upper left')
	plt.xlabel("Number of data")
	plt.ylabel("Standardized weights")
	plt.title("Comparison of Declustering Methods")
	plt.xticks(ind+0.4, ('1','2','3','4','5','6','7','8','9','10','11','12','13','14'))
	plt.show()
