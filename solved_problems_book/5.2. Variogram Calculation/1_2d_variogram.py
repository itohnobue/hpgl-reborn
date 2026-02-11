#
#	Solved Problems in Geostatistics
#
# ------------------------------------------------
#	Script for lesson 5.2
#	"Variogram Calculation"
# ------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import matplotlib.pyplot as plt
from gslib import *
from variogram_routines import *

# Loading sample data from file
data_dict = load_gslib_file("3welldata.txt")

x_coord = data_dict['X(Easting)(m)']
z_coord = data_dict['Z(m)']
poro_values = data_dict['Por']
y_coord = np.zeros(len(x_coord), dtype=np.float32)

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

#Suggested Parameters for Vertical Variogram:
XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=4, LagSeparation=4, TolDistance=4, NumLags=8,
    Ellipsoid=TVEllipsoid(R1=1, R2=0.1, R3=0.1, Azimut=0, Dip=90, Rotation=0)
), PointSet, Function, Params)

Variogram_x = XVariogram[:, 0]
print("Vertical semivariogram:", Variogram_x)

#Experimental vertical semivariogram
plt.figure()
plt.plot(XLagDistance, Variogram_x, 'bo')
plt.plot(XLagDistance, Variogram_x, 'b--')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Experimental vertical semivariogram")

#Suggested Parameters for Horizontal Variogram:
XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=1000, LagSeparation=1000, TolDistance=1000, NumLags=3,
    Ellipsoid=TVEllipsoid(R1=1, R2=0.1, R3=4.5, Azimut=0, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_z = XVariogram[:, 0]
print("Horizontal semovariogram:", Variogram_z)

#Experimental horizontal semivariogram
plt.figure()
plt.plot(XLagDistance, Variogram_z, 'bo')
plt.plot(XLagDistance, Variogram_z, 'b--')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Experimental horizontal semivariogram")
plt.show()
