#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 5.3
#	"Variogram Modeling and Volume Variance"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import matplotlib.pyplot as plt
from gslib import *
from variogram_routines import *

#---------------------------------------------------
#	Problem:
#
#	Model the experimental semivariograms from Part 2 of the previous Problem 5.2 using maximum of two nested structures. All directions must be modeled using the same structures and variance contributions for each structure, but each structure may have different range parameters.
#
# ----------------------------------------------------

nugget = 0

sill_hor1 = 15
var_range_hor1 = 4000

sill_hor2 = 20
var_range_hor2 = 5000

sill_ver = 11
var_range_ver = 35

def exp_var(sill, nugget, var_range, h_vect):
	Gamma = np.zeros((len(h_vect)), dtype=np.float32)
	for i in range(len(h_vect)):
		Gamma[i] = (sill - nugget) * (1 - np.exp(float(-h_vect[i]) * 3 / (var_range)))
	return Gamma

# Loading sample data from file
data_dict = load_gslib_file("allwelldata.txt")

x_coord = data_dict['X']
y_coord = data_dict['Y']
z_coord = data_dict['Z']
poro_values = data_dict['Por']

# Lets make a PointSet
PointSet = {}
PointSet['X'] = x_coord
PointSet['Y'] = y_coord
PointSet['Z'] = z_coord
PointSet['Property'] = poro_values

IndicatorData = []
IndicatorData.append(poro_values)

Params = {'HardData': IndicatorData}
Function = CalcVariogramFunction

#Suggested Parameters for Horizontal Variogram 1:

#Azimuth = 320 (Azimut)
#Dip = 0 (Dip)
#Lag Distance = 550 m (LagWidth, LagSeparation)
#Horizontal Bandwith = 500 m (R2)
#Vertical Bandwith = 5 m (R3)
#Number of Lags = 11 (NumLags)

XVariogram, XLagDistance1 = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=550, LagSeparation=550, TolDistance=450, NumLags=12,
    Ellipsoid=TVEllipsoid(R1=1, R2=500, R3=5, Azimut=320, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_hor1 = XVariogram[:, 0]
print("Horizontal XVariogram 1:")
print(Variogram_hor1)

#Suggested Parameters for Horizontal Variogram 2:

#Azimuth = 230 (Azimut)
#Dip = 0 (Dip)
#Lag Distance = 550 m (LagWidth, LagSeparation)
#Horizontal Bandwith = 500 m (R2)
#Vertical Bandwith = 5 m (R3)
#Number of Lags = 11 (NumLags)

XVariogram, XLagDistance2 = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=550, LagSeparation=550, TolDistance=450, NumLags=12,
    Ellipsoid=TVEllipsoid(R1=1, R2=500, R3=5, Azimut=230, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_hor2 = XVariogram[:, 0]
print("Horizontal XVariogram 2:")
print(Variogram_hor2)

#Calculate Gamma for horizontal semivariogram 1 and 2
Gamma1 = exp_var(sill_hor1, nugget, var_range_hor1, range(int(min(XLagDistance1)), int(max(XLagDistance1)), 1))
print("Gamma for horizontal semivariogram 1: ", Gamma1)

Gamma2 = exp_var(sill_hor2, nugget, var_range_hor2, range(int(min(XLagDistance2)), int(max(XLagDistance2)), 1))
print("Gamma for horizontal semivariogram 2: ", Gamma2)

#Experimental horizontal semivariogram 1 and 2
plt.figure()
plt.plot(XLagDistance1, Variogram_hor1, 'bo', color='blue')
plt.plot(range(int(min(XLagDistance1)), int(max(XLagDistance1)), 1), Gamma1, color='blue')
plt.plot(XLagDistance2, Variogram_hor2, 'bo', color='green')
plt.plot(range(int(min(XLagDistance2)), int(max(XLagDistance2)), 1), Gamma2, color='green')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Experimental horizontal semivariogram")

#Suggested Parameters for Vertical Variogram:

#Azimuth = 0 (Azimut)
#Dip = 90 (Dip)
#Lag Distance = 4 m (LagWidth, LagSeparation)
#Horizontal Bandwith = 0.0 m (R2)
#Vertical Bandwith = 10 m (R3)
#Number of Lags = 10 (NumLags)

XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=4, LagSeparation=4, TolDistance=4, NumLags=11,
    Ellipsoid=TVEllipsoid(R1=1, R2=0.1, R3=10, Azimut=0, Dip=90, Rotation=0)
), PointSet, Function, Params)

Variogram_ver = XVariogram[:, 0]
print("Vertical Variogram:")
print(Variogram_ver)

#Calculate Gamma for vertical semivariogram
Gamma = exp_var(sill_ver, nugget, var_range_ver, range(int(min(XLagDistance)), int(max(XLagDistance)), 1))
print("Gamma for vertical semivariogram: ", Gamma)

#Variogram modeling results for the vertical direction
plt.figure()
plt.plot(XLagDistance, Variogram_ver, 'bo')
plt.plot(range(int(min(XLagDistance)), int(max(XLagDistance)), 1), Gamma)
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Variogram modeling results for the vertical direction")
plt.show()
