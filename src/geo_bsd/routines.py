import numpy
import numpy.ma as ma
from numpy import (
    append,
    column_stack,
    copy,
    float32,
    float64,
    int32,
    mgrid,
    nonzero,
    ones,
    repeat,
    require,
    reshape,
    savetxt,
    sum,
    uint8,
    zeros,
)

from .validation import PathValidator


def CalcMean(Cube, Mask):
    CubeMasked = ma.masked_array(Cube, Mask == 0)
    return CubeMasked.mean()

def CalcMarginalProbsIndicator(Cube, Mask, Indicators):
    Result = zeros(len(Indicators))
    for i in range(len(Indicators)):
        Result.flat[i] = CalcMean(Cube == Indicators[i], Mask)
    return Result

def CalcVPC(Cube, Mask, MarginalMean):
    NZ = Cube.shape[2]
    MaskSum = Mask.sum(0).sum(0)
    CubeMasked = copy(Cube)
    CubeMasked[Mask == 0] = 0

    CubeSum = Cube.sum(0).sum(0)
    Result = ones(NZ) * MarginalMean
    Filter = MaskSum > 0
    Result[Filter] = float32(CubeSum[Filter]) / float32(MaskSum[Filter])

    return Result

def CalcVPCsIndicator(Cube, Mask, Indicators, MarginalProbs):
    Result = []
    for i in range(len(Indicators)):
        VPC = CalcVPC(Cube == Indicators[i], Mask, MarginalProbs[i])
        Result.append(VPC)

    return Result

def CubeFromVPC(VPC, NX, NY):
    NZ = len(VPC)
    VPC = reshape(VPC, (1, 1, NZ))
    Cube = repeat(repeat(VPC, NX, axis = 0), NY, axis = 1)
    return float32(Cube)

def CubesFromVPCs(VPCs, NX, NY):
    Cubes = []
    for i in range(len(VPCs)):
        Cube = CubeFromVPC(VPCs[i], NX, NY)
        Cubes.append(Cube)
    return Cubes

def Cubes2PointSet(CubesDictionary, Mask):
    NX, NY, NZ = list(CubesDictionary.values())[0].shape
    I, J = mgrid[0:NX, 0:NY]
    PointSet = {'X':zeros(0, dtype=int32), 'Y':zeros(0, dtype=int32), 'Z':zeros(0, dtype=int32)}
    for Key in CubesDictionary.keys():
        PointSet[Key] = zeros(0, dtype=int32)

    for k in range(NZ):
        Slice = Mask[:, :, k]
        PointSet['X'] = append(PointSet['X'], I[Slice])
        PointSet['Y'] = append(PointSet['Y'], J[Slice])
        PointSet['Z'] = append(PointSet['Z'], k * ones(Slice.sum(0).sum(0), dtype=int32))
        for Key in CubesDictionary.keys():
            DataSlice = CubesDictionary[Key][:, :, k]
            PointSet[Key] = append(PointSet[Key], DataSlice[Slice])

    return PointSet

def Cube2PointSet(Cube, Mask):
    NX, NY, NZ = Cube.shape
    I, J = mgrid[0:NX, 0:NY]
    X = zeros(0, dtype=int32)
    Y = zeros(0, dtype=int32)
    Z = zeros(0, dtype=int32)
    Property = zeros(0, dtype=int32)
    for k in range(NZ):
        Slice = Mask[:, :, k]
        X = append(X, I[Slice])
        Y = append(Y, J[Slice])
        Z = append(Z, k * ones(Slice.sum(0).sum(0), dtype=int32))
        DataSlice = Cube[:, :, k]
        Property = append(Property, DataSlice[Slice])
    return X, Y, Z, Property

def PointSet2Cube(X, Y, Z, Property, Cube):
    NX, NY, NZ = Cube.shape
    Mask = zeros(Cube.shape)
    for Ind in range(len(X.flat)):
        if (0 <= X[Ind]) & (X[Ind] < NX) & (0 <= Y[Ind]) & (Y[Ind] < NY) & (0 <= Z[Ind]) & (Z[Ind] < NZ):
            Cube[X[Ind], Y[Ind], Z[Ind]] = Property[Ind]
            Mask[X[Ind], Y[Ind], Z[Ind]] = 1
    return Cube, Mask == 1

def SaveGSLIBPointSet(PointSet, FileName, Caption):
    if not FileName or not isinstance(FileName, str):
        raise ValueError("SaveGSLIBPointSet: FileName must be a non-empty string")
    PathValidator.validate_filepath(FileName)

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in PointSet.keys():
        lens = numpy.append(lens, len(PointSet[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError("SaveGSLIBPointSet: All properties in GSLIB dictionary must have equal size")

    with open(FileName, "w", encoding='utf-8') as f:
        # 1. Caption
        f.write(Caption + '\n')

        # 2. Number of properties in file
        f.write(str(len(PointSet)) + '\n')

        # 3. Properties names
        for Key in PointSet.keys():
            f.write(Key + '\n')

        MegaPointSet = zeros((int(lens[0]), 0))
        for Key in PointSet.keys():
            MegaPointSet = column_stack((MegaPointSet, PointSet[Key]))
        savetxt(f, MegaPointSet)

def SaveGSLIBCubes(CubesDictionary, FileName, Caption, Format = "%d"):
    if not FileName or not isinstance(FileName, str):
        raise ValueError("SaveGSLIBCubes: FileName must be a non-empty string")
    PathValidator.validate_filepath(FileName)

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in CubesDictionary.keys():
        lens = numpy.append(lens, len(CubesDictionary[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError("SaveGSLIBCubes: All properties in GSLIB dictionary must have equal size")

    with open(FileName, "w", encoding='utf-8') as f:
        # 1. Caption
        f.write(Caption + '\n')

        # 2. Number of properties in file
        f.write(str(len(CubesDictionary)) + '\n')

        # 3. Properties names
        for Key in CubesDictionary.keys():
            f.write(Key + '\n')

        MegaCube = zeros((int(lens[0]), 0))
        for Key in CubesDictionary.keys():
            MegaCube = column_stack((MegaCube, CubesDictionary[Key].copy().swapaxes(0, 2).swapaxes(1, 2).flat))
        savetxt(f, MegaCube, Format)

def GetCubicalMask(Radiuses):
    MeanMask = ones( (Radiuses[0]*2, Radiuses[1]*2, Radiuses[2]*2), dtype = uint8)
    MeanMask = require(MeanMask, requirements = 'F')
    return MeanMask

def GetEllipseMask(Radiuses):
    MeanMask = zeros( (Radiuses[0]*2, Radiuses[1]*2, Radiuses[2]*2), dtype = uint8)
    MeanMask = require(MeanMask, requirements = 'F')

    x0 = Radiuses[0]
    y0 = Radiuses[1]
    z0 = Radiuses[2]

    for a in range(Radiuses[0]*2):
        for b in range(Radiuses[1]*2):
            for c in range(Radiuses[2]*2):
                if ( ((a-x0)**2 / float32(Radiuses[0]**2)) + ((b-y0)**2 / float32(Radiuses[1]**2)) + ((c-z0)**2 / float32(Radiuses[2]**2)) <= 1):
                    MeanMask[a,b,c] = 1
    return MeanMask

def MeanCalc(Cube, Mask, Radiuses, MeanMask, coords, undefined_value):
    i, j, k = coords
    imin = i-Radiuses[0]
    imax = i+Radiuses[0]

    jmin = j-Radiuses[1]
    jmax = j+Radiuses[1]

    kmin = k-Radiuses[2]
    kmax = k+Radiuses[2]

    if imin < 0:
        imin = 0
    if jmin < 0:
        jmin = 0
    if kmin < 0:
        kmin = 0

    if imax > Cube.shape[0]:
        imax = Cube.shape[0]
    if jmax > Cube.shape[1]:
        jmax = Cube.shape[1]
    if kmax > Cube.shape[2]:
        kmax = Cube.shape[2]

    if (sum ( (Mask[imin:imax, jmin:jmax, kmin:kmax]==1) & (MeanMask[0:(imax-imin), 0:(jmax-jmin), 0:(kmax-kmin)]==1) ) > 0):
        return Cube[imin:imax, jmin:jmax, kmin:kmax][nonzero((Mask[imin:imax, jmin:jmax, kmin:kmax]==1) & (MeanMask[0:(imax-imin), 0:(jmax-jmin), 0:(kmax-kmin)]==1))].mean()
    else:
        return undefined_value


def MovingAverage3D(cube_mask, Radiuses, undefined_value, MaskCalcFunction):
    Cube, Mask = cube_mask
    MACube = copy(Cube)
    MeanMask = MaskCalcFunction(Radiuses)

    for i in range(Cube.shape[0]):
        for j in range(Cube.shape[1]):
            for k in range(Cube.shape[2]):
                MACube[i,j,k] = MeanCalc(Cube, Mask, Radiuses, MeanMask, (i,j,k), undefined_value)

    return MACube

def LoadGslibFile(filename, property_size):
    if not filename or not isinstance(filename, str):
        raise ValueError("LoadGslibFile: filename must be a non-empty string")

    # Validate filepath for security (path traversal prevention, exists check)
    safe_path = PathValidator.validate_filepath(filename, must_exist=True)

    result = {}
    list_prop = []
    points = []

    with open(safe_path, encoding='utf-8') as f:
        f.readline()  # Skip caption line
        num_p = int(f.readline())

        for _ in range(num_p):
            list_prop.append(str(f.readline().strip()))

        for i in range(len(list_prop)):
            result[list_prop[i]] = zeros(property_size[0]*property_size[1]*property_size[2])

        index = zeros(len(list_prop), dtype=int)

        for line in f:
            points = line.split()
            for j in range(len(points)):
                if index[j] >= len(result[list_prop[j]]):
                    raise RuntimeError(
                        f"LoadGslibFile: too many values for property '{list_prop[j]}'. "
                        f"Expected {property_size[0]*property_size[1]*property_size[2]} elements "
                        f"(grid {property_size}), got more."
                    )
                result[list_prop[j]][index[j]] = float64(points[j])
                index[j] += 1

    for dkey in result.keys():
        result[dkey] = result[dkey].reshape(property_size)

    return result
