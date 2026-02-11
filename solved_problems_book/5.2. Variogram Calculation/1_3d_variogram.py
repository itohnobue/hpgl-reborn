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

#Horizontal Variogram 1:
XVariogram, XLagDistance1 = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=550, LagSeparation=550, TolDistance=800, NumLags=12,
    Ellipsoid=TVEllipsoid(R1=1, R2=500, R3=5, Azimut=320, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_hor1 = XVariogram[:, 0]
print("Horizontal Variogram 1:", Variogram_hor1)

#Horizontal Variogram 2:
XVariogram, XLagDistance2 = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=550, LagSeparation=550, TolDistance=800, NumLags=12,
    Ellipsoid=TVEllipsoid(R1=1, R2=500, R3=5, Azimut=230, Dip=0, Rotation=0)
), PointSet, Function, Params)

Variogram_hor2 = XVariogram[:, 0]
print("Horizontal Variogram 2:", Variogram_hor2)

#Experimental horizontal semivariogram 1 and 2
plt.figure()
plt.plot(XLagDistance1, Variogram_hor1, 'bo')
plt.plot(XLagDistance1, Variogram_hor1, 'b-.')
plt.plot(XLagDistance2, Variogram_hor2, 'bo', color='green')
plt.plot(XLagDistance2, Variogram_hor2, 'b--', color='green')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Experimental horizontal semivariogram")

#Vertical Variogram:
XVariogram, XLagDistance = PointSetScanContStyle(TVVariogramSearchTemplate(
    LagWidth=4, LagSeparation=4, TolDistance=1.5, NumLags=11,
    Ellipsoid=TVEllipsoid(R1=1, R2=0.1, R3=10, Azimut=0, Dip=90, Rotation=0)
), PointSet, Function, Params)

Variogram_ver = XVariogram[:, 0]
print("Vertical Variogram:", Variogram_ver)

#Experimental vertical semivariogram
plt.figure()
plt.plot(XLagDistance, Variogram_ver, 'bo')
plt.plot(XLagDistance, Variogram_ver, 'b--')
plt.xlabel("Distance")
plt.ylabel("Gamma")
plt.title("Experimental vertical semivariogram")
plt.show()
