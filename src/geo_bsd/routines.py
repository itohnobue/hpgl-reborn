# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import os

import numpy
import numpy.ma as ma
from numpy import (
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

from .validation import GridValidator, PathValidator, ValidationConstants


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

    CubeSum = CubeMasked.sum(0).sum(0)
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
    Cube = repeat(repeat(VPC, NX, axis=0), NY, axis=1)
    return float32(Cube)


def CubesFromVPCs(VPCs, NX, NY):
    Cubes = []
    for i in range(len(VPCs)):
        Cube = CubeFromVPC(VPCs[i], NX, NY)
        Cubes.append(Cube)
    return Cubes


def Cubes2PointSet(CubesDictionary, Mask):
    if not CubesDictionary:
        raise ValueError("Cubes2PointSet: CubesDictionary must not be empty")

    NX, NY, NZ = list(CubesDictionary.values())[0].shape
    grid_i, grid_j = mgrid[0:NX, 0:NY]

    total = int(Mask.sum())

    PointSet = {
        "X": zeros(total, dtype=int32),
        "Y": zeros(total, dtype=int32),
        "Z": zeros(total, dtype=int32),
    }
    for Key in CubesDictionary.keys():
        PointSet[Key] = zeros(total, dtype=CubesDictionary[Key].dtype)

    offset = 0
    for k in range(NZ):
        Slice = Mask[:, :, k].astype(bool)
        n = Slice.sum()
        PointSet["X"][offset : offset + n] = grid_i[Slice]
        PointSet["Y"][offset : offset + n] = grid_j[Slice]
        PointSet["Z"][offset : offset + n] = k * ones(n, dtype=int32)
        for Key in CubesDictionary.keys():
            DataSlice = CubesDictionary[Key][:, :, k]
            PointSet[Key][offset : offset + n] = DataSlice[Slice]
        offset += n

    return PointSet


def Cube2PointSet(Cube, Mask):
    NX, NY, NZ = Cube.shape
    grid_i, grid_j = mgrid[0:NX, 0:NY]

    total = int(Mask.sum())
    X = zeros(total, dtype=int32)
    Y = zeros(total, dtype=int32)
    Z = zeros(total, dtype=int32)
    Property = zeros(total, dtype=Cube.dtype)

    offset = 0
    for k in range(NZ):
        Slice = Mask[:, :, k].astype(bool)
        n = Slice.sum()
        X[offset : offset + n] = grid_i[Slice]
        Y[offset : offset + n] = grid_j[Slice]
        Z[offset : offset + n] = k * ones(n, dtype=int32)
        DataSlice = Cube[:, :, k]
        Property[offset : offset + n] = DataSlice[Slice]
        offset += n
    return X, Y, Z, Property


def PointSet2Cube(X, Y, Z, Property, Cube):
    NX, NY, NZ = Cube.shape
    Mask = zeros(Cube.shape)

    # Vectorized: filter all valid indices at once, then assign in bulk.
    Xf = numpy.array(X).ravel()
    Yf = numpy.array(Y).ravel()
    Zf = numpy.array(Z).ravel()
    Pf = numpy.array(Property).ravel()

    valid = (Xf >= 0) & (Xf < NX) & (Yf >= 0) & (Yf < NY) & (Zf >= 0) & (Zf < NZ)

    if numpy.any(valid):
        vX = Xf[valid]
        vY = Yf[valid]
        vZ = Zf[valid]
        vP = Pf[valid]
        Cube[vX, vY, vZ] = vP
        Mask[vX, vY, vZ] = 1

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

    if not PointSet:
        raise ValueError("SaveGSLIBPointSet: PointSet must not be empty")

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in PointSet.keys():
        lens = numpy.append(lens, len(PointSet[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError(
            "SaveGSLIBPointSet: All properties in GSLIB dictionary must have equal size"
        )

    # Security: safe_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks.
    with PathValidator.safe_open_write(FileName, basedir=os.path.dirname(os.path.abspath(FileName))) as f:
        # 1. Caption
        f.write(Caption + "\n")

        # 2. Number of properties in file
        f.write(str(len(PointSet)) + "\n")

        # 3. Properties names
        for Key in PointSet.keys():
            f.write(Key + "\n")

        MegaPointSet = zeros((int(lens[0]), 0))
        for Key in PointSet.keys():
            MegaPointSet = column_stack((MegaPointSet, PointSet[Key]))
        savetxt(f, MegaPointSet)


def SaveGSLIBCubes(CubesDictionary, FileName, Caption, Format="%g"):
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
        NumPy ``savetxt`` format string (default ``"%g"``).

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

    if not CubesDictionary:
        raise ValueError("SaveGSLIBCubes: CubesDictionary must not be empty")

    # Validate all properties have the same length before writing any data
    lens = numpy.array([])
    for Key in CubesDictionary.keys():
        lens = numpy.append(lens, len(CubesDictionary[Key].flat))

    if sum(lens - lens[0]) != 0:
        raise RuntimeError(
            "SaveGSLIBCubes: All properties in GSLIB dictionary must have equal size"
        )

    # Security: safe_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks.
    with PathValidator.safe_open_write(FileName, basedir=os.path.dirname(os.path.abspath(FileName))) as f:
        # 1. Caption
        f.write(Caption + "\n")

        # 2. Number of properties in file
        f.write(str(len(CubesDictionary)) + "\n")

        # 3. Properties names
        for Key in CubesDictionary.keys():
            f.write(Key + "\n")

        MegaCube = zeros((int(lens[0]), 0))
        for Key in CubesDictionary.keys():
            MegaCube = column_stack((MegaCube, CubesDictionary[Key].copy().swapaxes(0, 2).flat))
        savetxt(f, MegaCube, Format)


def GetCubicalMask(Radiuses):
    MeanMask = ones((Radiuses[0] * 2, Radiuses[1] * 2, Radiuses[2] * 2), dtype=uint8)
    MeanMask = require(MeanMask, requirements="F")
    return MeanMask


def GetEllipseMask(Radiuses):
    rx, ry, rz = Radiuses
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError(
            f"GetEllipseMask: radius components must be positive, "
            f"got ({rx}, {ry}, {rz})"
        )
    x0, y0, z0 = rx, ry, rz

    a, b, c = mgrid[0 : rx * 2, 0 : ry * 2, 0 : rz * 2]
    ellipsoid_eq = (
        (a - x0) ** 2 / (rx**2) + (b - y0) ** 2 / (ry**2) + (c - z0) ** 2 / (rz**2)
    ) <= 1

    MeanMask = zeros((rx * 2, ry * 2, rz * 2), dtype=uint8)
    MeanMask = require(MeanMask, requirements="F")
    MeanMask[ellipsoid_eq] = 1
    return MeanMask


def MeanCalc(Cube, Mask, Radiuses, MeanMask, coords, undefined_value):
    i, j, k = coords
    imin = i - Radiuses[0]
    imax = i + Radiuses[0]

    jmin = j - Radiuses[1]
    jmax = j + Radiuses[1]

    kmin = k - Radiuses[2]
    kmax = k + Radiuses[2]

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

    # Compute MeanMask offsets to align with the clamped data window.
    # When the window is clamped near the left/bottom/front boundary,
    # the MeanMask must be shifted by the same amount so that the mask
    # center stays aligned with the cell position (i, j, k).
    i_offset = imin - i + Radiuses[0]
    j_offset = jmin - j + Radiuses[1]
    k_offset = kmin - k + Radiuses[2]

    if (
        sum(
            (Mask[imin:imax, jmin:jmax, kmin:kmax] == 1)
            & (
                MeanMask[
                    i_offset : (i_offset + imax - imin),
                    j_offset : (j_offset + jmax - jmin),
                    k_offset : (k_offset + kmax - kmin),
                ]
                == 1
            )
        )
        > 0
    ):
        return Cube[imin:imax, jmin:jmax, kmin:kmax][
            nonzero(
                (Mask[imin:imax, jmin:jmax, kmin:kmax] == 1)
                & (
                    MeanMask[
                        i_offset : (i_offset + imax - imin),
                        j_offset : (j_offset + jmax - jmin),
                        k_offset : (k_offset + kmax - kmin),
                    ]
                    == 1
                )
            )
        ].mean()
    else:
        return undefined_value


def MovingAverage3D(cube_mask, Radiuses, undefined_value, MaskCalcFunction):
    Cube, Mask = cube_mask
    MACube = copy(Cube)
    MeanMask = MaskCalcFunction(Radiuses)

    for i in range(Cube.shape[0]):
        for j in range(Cube.shape[1]):
            for k in range(Cube.shape[2]):
                MACube[i, j, k] = MeanCalc(
                    Cube, Mask, Radiuses, MeanMask, (i, j, k), undefined_value
                )

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

    # Validate property_size dimensions
    if not isinstance(property_size, tuple) or len(property_size) != 3:
        raise ValueError(
            f"LoadGslibFile: property_size must be a tuple of 3 ints, "
            f"got {type(property_size).__name__} with len={len(property_size) if isinstance(property_size, tuple) else 'N/A'}"
        )
    nx, ny, nz = property_size
    if not all(isinstance(d, (int, numpy.integer)) and d > 0 for d in (nx, ny, nz)):
        raise ValueError(
            f"LoadGslibFile: property_size dimensions must be positive integers, "
            f"got ({nx}, {ny}, {nz})"
        )

    # Validate grid dimensions against MAX_GRID_SIZE for consistency with
    # all other grid-creating paths (SugarboxGrid, sgs_simulation, sis_simulation, etc.)
    GridValidator.validate_grid_dimensions(nx, ny, nz)

    result = {}
    list_prop = []
    points = []

    # Security: safe_open_read validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks.
    with PathValidator.safe_open_read(filename, basedir=os.path.dirname(os.path.abspath(filename))) as f:
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
            result[list_prop[i]] = zeros(property_size[0] * property_size[1] * property_size[2])

        index = zeros(len(list_prop), dtype=int)

        grid_size = nx * ny * nz
        for line in f:
            # Skip blank lines (whitespace-only)
            if not line.strip():
                continue
            points = line.split()
            # Validate token count matches expected property count per line
            if len(points) != num_p:
                raise RuntimeError(
                    f"LoadGslibFile: expected {num_p} values per data line, "
                    f"got {len(points)} tokens in line: {line.strip()!r}"
                )
            for j in range(num_p):
                if index[j] >= len(result[list_prop[j]]):
                    raise RuntimeError(
                        f"LoadGslibFile: too many values for property '{list_prop[j]}'. "
                        f"Expected {grid_size} elements "
                        f"(grid {property_size}), got more."
                    )
                result[list_prop[j]][index[j]] = float64(points[j])
                index[j] += 1

    # Validate that all properties received the expected number of values
    for j, key in enumerate(list_prop):
        if index[j] != grid_size:
            raise RuntimeError(
                f"LoadGslibFile: property '{key}' has {index[j]} values, "
                f"expected {grid_size} (grid {property_size}). "
                f"File may be truncated or corrupted."
            )

    for dkey in result.keys():
        if not numpy.all(numpy.isfinite(result[dkey])):
            raise ValueError(
                f"LoadGslibFile: property '{dkey}' contains non-finite "
                f"values (NaN or Inf). GSLIB format does not support non-finite values."
            )
        result[dkey] = result[dkey].reshape(property_size, order="F")

    return result
