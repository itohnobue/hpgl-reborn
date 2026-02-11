import sys
import os
import numpy as np
import numpy.ma as ma
from numpy import savetxt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from geo_bsd import *


def CalcMean(Cube, Mask):
    CubeMasked = ma.masked_array(Cube, Mask == False)
    return CubeMasked.mean()


def CalcMarginalProbsIndicator(Cube, Mask, Indicators):
    Result = np.zeros(len(Indicators))
    for i in range(len(Indicators)):
        Result.flat[i] = CalcMean(Cube == Indicators[i], Mask)
    return Result


def CalcVPC(Cube, Mask, MarginalMean):
    NZ = Cube.shape[2]
    MaskSum = Mask.sum(0).sum(0)
    CubeMasked = Cube
    CubeMasked[Mask == False] = 0

    CubeSum = Cube.sum(0).sum(0)
    Result = np.ones(NZ) * MarginalMean
    Filter = MaskSum > 0
    Result[Filter] = np.float32(CubeSum[Filter]) / np.float32(MaskSum[Filter])

    return Result


def CalcVPCsIndicator(Cube, Mask, Indicators, MarginalProbs):
    Result = []
    for i in range(len(Indicators)):
        VPC = CalcVPC(Cube == Indicators[i], Mask, MarginalProbs[i])
        Result.append(VPC)

    return np.array(Result)


def CubeFromVPC(VPC, NX, NY):
    NZ = len(VPC)
    VPC = np.reshape(VPC, (1, 1, NZ))
    Cube = np.repeat(np.repeat(VPC, NX, axis=0), NY, axis=1)
    return np.float32(Cube)


def CubesFromVPCs(VPCs, NX, NY):
    Cubes = []
    for i in range(len(VPCs)):
        Cube = CubeFromVPC(VPCs[i], NX, NY)
        Cubes.append(Cube)
    return Cubes


def Cubes2PointSet(CubesDictionary, Mask):
    NX, NY, NZ = list(CubesDictionary.values())[0].shape
    I, J = np.mgrid[0:NX, 0:NY]
    PointSet = {'X': np.zeros(0, dtype=np.int32), 'Y': np.zeros(0, dtype=np.int32), 'Z': np.zeros(0, dtype=np.int32)}
    for Key in CubesDictionary.keys():
        PointSet[Key] = np.zeros(0, dtype=np.int32)

    for k in range(NZ):
        Slice = Mask[:, :, k]
        PointSet['X'] = np.append(PointSet['X'], I[Slice])
        PointSet['Y'] = np.append(PointSet['Y'], J[Slice])
        PointSet['Z'] = np.append(PointSet['Z'], k * np.ones(Slice.sum(0).sum(0), dtype=np.int32))
        for Key in CubesDictionary.keys():
            DataSlice = CubesDictionary[Key][:, :, k]
            PointSet[Key] = np.append(PointSet[Key], DataSlice[Slice])

    return PointSet


def Cube2PointSet(Cube, Mask):
    NX, NY, NZ = Cube.shape
    I, J = np.mgrid[0:NX, 0:NY]
    X = np.zeros(0, dtype=np.int32)
    Y = np.zeros(0, dtype=np.int32)
    Z = np.zeros(0, dtype=np.int32)
    Property = np.zeros(0, dtype=np.int32)
    for k in range(NZ):
        Slice = Mask[:, :, k]
        X = np.append(X, I[Slice])
        Y = np.append(Y, J[Slice])
        Z = np.append(Z, k * np.ones(Slice.sum(0).sum(0), dtype=np.int32))
        DataSlice = Cube[:, :, k]
        Property = np.append(Property, DataSlice[Slice])
    return X, Y, Z, Property


def PointSet2Cube(X, Y, Z, Property, Cube):
    NX, NY, NZ = Cube.shape
    for Ind in range(len(X.flat)):
        if (0 <= X[Ind]) & (X[Ind] < NX) & (0 <= Y[Ind]) & (Y[Ind] < NY) & (0 <= Z[Ind]) & (Z[Ind] < NZ):
            Cube[X[Ind], Y[Ind], Z[Ind]] = Property[Ind]
    return Cube


def SaveGSLIBPointSet(PointSet, FileName, Caption):
    with open(FileName, "w") as f:
        # 1. Caption
        f.write(Caption + '\n')

        # 2. Number of properties in file
        f.write(str(len(PointSet)) + '\n')

        # 3. Properties names
        lens = np.array([])

        for Key in PointSet.keys():
            f.write(Key + '\n')
            lens = np.append(lens, len(PointSet[Key].flat))

        # Check that all properties have the same length
        if np.sum(lens - lens[0]) == 0:
            MegaPointSet = np.zeros((int(lens[0]), 0))
            for Key in PointSet.keys():
                MegaPointSet = np.column_stack((MegaPointSet, PointSet[Key]))
            savetxt(f, MegaPointSet)
        else:
            print("ERROR! All properties in GSLIB dictionary must have equal size")


def SaveGSLIBCubes(CubesDictionary, FileName, Caption, Format="%d"):
    with open(FileName, "w") as f:
        # 1. Caption
        f.write(Caption + '\n')

        # 2. Number of properties in file
        f.write(str(len(CubesDictionary)) + '\n')

        # 3. Properties names
        lens = np.array([])

        for Key in CubesDictionary.keys():
            f.write(Key + '\n')
            lens = np.append(lens, len(CubesDictionary[Key].flat))

        # Check that all properties have the same length
        if np.sum(lens - lens[0]) == 0:
            MegaCube = np.zeros((int(lens[0]), 0))
            for Key in CubesDictionary.keys():
                MegaCube = np.column_stack((MegaCube, CubesDictionary[Key].copy().swapaxes(0, 2).swapaxes(1, 2).flat))
            savetxt(f, MegaCube, Format)
        else:
            print("ERROR! All properties in GSLIB dictionary must have equal size")
