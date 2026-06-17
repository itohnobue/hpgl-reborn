# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import os

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

from .validation import PathValidator, ValidationConstants


def CalcMean(Cube, Mask):
    """Calculate the arithmetic mean of unmasked (informed) cells.

    Parameters
    ----------
    Cube : numpy.ndarray
        3D array of property values.
    Mask : numpy.ndarray
        3D array where non-zero indicates an informed cell, zero
        indicates an uninformed (masked) cell.

    Returns
    -------
    float
        Mean of values in cells where ``Mask != 0``.

    Notes
    -----
    Uses ``numpy.ma.masked_array`` to exclude masked cells from the
    mean computation.
    """
    CubeMasked = ma.masked_array(Cube, Mask == 0)
    return CubeMasked.mean()

def CalcMarginalProbsIndicator(Cube, Mask, Indicators):
    Result = zeros(len(Indicators))
    for i in range(len(Indicators)):
        Result.flat[i] = CalcMean(Cube == Indicators[i], Mask)
    return Result

def CalcVPC(Cube, Mask, MarginalMean):
    """Compute the Vertical Proportion Curve (VPC) for a 3D property.

    For each layer ``k`` along the Z-axis, calculates the mean value
    of unmasked cells. Layers with no informed cells default to the
    marginal mean.

    Parameters
    ----------
    Cube : numpy.ndarray
        3D array of property values.
    Mask : numpy.ndarray
        3D array where non-zero indicates an informed cell.
    MarginalMean : float
        Default value for layers with no informed cells.

    Returns
    -------
    numpy.ndarray
        1D array of length ``Cube.shape[2]`` with per-layer means.
    """
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
    """Write scattered point data to a file in GSLIB format.

    Produces a text file with a header (caption, property count,
    property names) followed by data columns.

    Parameters
    ----------
    PointSet : dict
        Dictionary mapping property names to 1D arrays. All arrays
        must have the same length.
    FileName : str
        Output file path (validated for security).
    Caption : str
        One-line description written to the file header.

    Raises
    ------
    ValueError
        If ``FileName`` is empty or not a string.
    RuntimeError
        If properties in ``PointSet`` have unequal lengths.

    See Also
    --------
    SaveGSLIBCubes : Write 3D grid data in GSLIB format.
    """
    if not FileName or not isinstance(FileName, str):
        raise ValueError("SaveGSLIBPointSet: FileName must be a non-empty string")
    safe_path = PathValidator.validate_filepath_in_basedir(
        FileName, basedir=os.path.dirname(os.path.abspath(FileName)))

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in PointSet.keys():
        lens = numpy.append(lens, len(PointSet[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError("SaveGSLIBPointSet: All properties in GSLIB dictionary must have equal size")

    with open(safe_path, "w", encoding='utf-8') as f:
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
    """Write 3D grid properties to a file in GSLIB format.

    Same header structure as ``SaveGSLIBPointSet``, but flattens 3D
    cubes into columns of data lines.

    Parameters
    ----------
    CubesDictionary : dict
        Dictionary mapping property names to 3D numpy arrays. All
        arrays must have the same shape.
    FileName : str
        Output file path (validated for security).
    Caption : str
        One-line description written to the file header.
    Format : str, optional
        NumPy ``savetxt`` format string (default ``"%d"``).

    Raises
    ------
    ValueError
        If ``FileName`` is empty or not a string.
    RuntimeError
        If properties have unequal sizes.

    See Also
    --------
    SaveGSLIBPointSet : Write scattered point data in GSLIB format.
    """
    if not FileName or not isinstance(FileName, str):
        raise ValueError("SaveGSLIBCubes: FileName must be a non-empty string")
    safe_path = PathValidator.validate_filepath_in_basedir(
        FileName, basedir=os.path.dirname(os.path.abspath(FileName)))

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in CubesDictionary.keys():
        lens = numpy.append(lens, len(CubesDictionary[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError("SaveGSLIBCubes: All properties in GSLIB dictionary must have equal size")

    with open(safe_path, "w", encoding='utf-8') as f:
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
    """Load a GSLIB-format file into a dictionary of 3D property arrays.

    Reads the GSLIB header (caption, property count, property names)
    and then parses data lines into separately-named arrays reshaped
    to ``property_size``.

    Parameters
    ----------
    filename : str
        Path to the GSLIB file (must exist).
    property_size : tuple of int
        Grid dimensions ``(nx, ny, nz)`` for reshaping loaded data.

    Returns
    -------
    dict
        Dictionary mapping property names to 3D numpy arrays with
        shape ``property_size``.

    Raises
    ------
    ValueError
        If ``filename`` is empty/not a string, ``num_p`` is invalid,
        or the property count exceeds the maximum.
    RuntimeError
        If the file contains more values than expected for a given
        property.

    See Also
    --------
    SaveGSLIBPointSet : Write point data in GSLIB format.
    SaveGSLIBCubes : Write grid data in GSLIB format.
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("LoadGslibFile: filename must be a non-empty string")

    # Validate filepath for security (path traversal prevention, exists check)
    safe_path = PathValidator.validate_filepath_in_basedir(
        filename, basedir=os.path.dirname(os.path.abspath(filename)), must_exist=True)

    result = {}
    list_prop = []
    points = []

    with open(safe_path, encoding='utf-8') as f:
        f.readline()  # Skip caption line
        num_p = int(f.readline())

        # Validate num_p against a reasonable upper bound to prevent
        # memory exhaustion from malicious GSLIB file headers.
        if num_p < 1:
            raise ValueError(f"LoadGslibFile: num_p must be at least 1, got {num_p}")
        max_props = max(ValidationConstants.MAX_INDICATORS * 4, 1024)
        if num_p > max_props:
            raise ValueError(
                f"LoadGslibFile: num_p {num_p} exceeds maximum allowed "
                f"properties ({max_props}). File may be corrupted or malicious."
            )

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
