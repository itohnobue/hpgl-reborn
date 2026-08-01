# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
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
    pad,
    repeat,
    require,
    reshape,
    savetxt,
    sum,
    uint8,
    where,
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

    # Reject property names that collide with the coordinate names. A
    # property named "X"/"Y"/"Z" would silently overwrite the coordinate
    # arrays below, producing a point set with garbage coordinates.
    for Key in CubesDictionary.keys():
        if Key in ("X", "Y", "Z"):
            raise ValueError(
                f"Cubes2PointSet: property name '{Key}' collides with the "
                f"coordinate names X/Y/Z. Rename the property."
            )

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


def SaveGSLIBPointSet(PointSet, FileName, Caption, basedir=None):
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
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

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

    # F-32: reject non-finite values before writing, matching the C++ writer
    # contract (property_writer.cpp write_value). GSLIB has no NaN/Inf
    # representation; writing them produces a file the loader refuses to read.
    for Key in PointSet.keys():
        if not numpy.all(numpy.isfinite(PointSet[Key])):
            raise ValueError(
                f"SaveGSLIBPointSet: property '{Key}' contains non-finite values "
                f"(NaN or Inf). GSLIB format does not support non-finite values."
            )

    # Security: safe_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks. The basedir is
    # the trusted base (DEFAULT_BASE_DIR unless overridden) — NOT derived
    # from the filename's own directory, which would defeat containment.
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_open_write(FileName, basedir=basedir) as f:
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


def SaveGSLIBCubes(CubesDictionary, FileName, Caption, Format="%g", basedir=None):
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
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

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

    # F-32: reject non-finite values before writing, matching the C++ writer
    # contract (property_writer.cpp write_value). GSLIB has no NaN/Inf
    # representation; writing them produces a file the loader refuses to read.
    for Key in CubesDictionary.keys():
        if not numpy.all(numpy.isfinite(CubesDictionary[Key])):
            raise ValueError(
                f"SaveGSLIBCubes: property '{Key}' contains non-finite values "
                f"(NaN or Inf). GSLIB format does not support non-finite values."
            )

    # Security: safe_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks. The basedir is
    # the trusted base (DEFAULT_BASE_DIR unless overridden) — NOT derived
    # from the filename's own directory, which would defeat containment.
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_open_write(FileName, basedir=basedir) as f:
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

    # I2-06: the cubical-mask path is vectorized with 3D cumulative sums.
    # The per-cell Python loop below is O(N) with ~10 numpy ops per cell
    # (minutes-to-hours for a 100^3 grid); the integral-image equivalent
    # computes the same box means in O(N) vectorized numpy operations.
    # The vectorization applies whenever the mean mask is uniform (the
    # GetCubicalMask case); non-uniform (ellipse) masks fall back to the
    # loop, bounded by MAX_MOVING_AVERAGE_VOLUME to prevent a hang.
    if bool(numpy.all(MeanMask == 1)):
        return _moving_average_cubical(Cube, Mask, Radiuses, undefined_value)

    volume = int(Cube.shape[0]) * int(Cube.shape[1]) * int(Cube.shape[2])
    if volume > ValidationConstants.MAX_MOVING_AVERAGE_VOLUME:
        raise ValueError(
            f"MovingAverage3D: grid volume {volume} exceeds the maximum "
            f"{ValidationConstants.MAX_MOVING_AVERAGE_VOLUME} supported by "
            f"the non-cubical (ellipse-mask) pure-Python path. Use a cubical "
            f"mask or reduce the grid size."
        )

    for i in range(Cube.shape[0]):
        for j in range(Cube.shape[1]):
            for k in range(Cube.shape[2]):
                MACube[i, j, k] = MeanCalc(
                    Cube, Mask, Radiuses, MeanMask, (i, j, k), undefined_value
                )

    return MACube


def _moving_average_cubical(Cube, Mask, Radiuses, undefined_value):
    """Vectorized moving average for uniform (cubical) mean masks.

    Computes, for every cell, the mean of ``Cube`` over the radius box
    restricted to ``Mask == 1`` cells, falling back to
    ``undefined_value`` when the box contains no informed cell. This is
    exactly the ``MeanCalc`` result for an all-ones ``MeanMask`` (the
    ``GetCubicalMask`` case), evaluated with 3D integral images so the
    whole grid is processed in vectorized numpy calls.
    """
    rx, ry, rz = Radiuses
    nx, ny, nz = Cube.shape
    informed = (Mask == 1).astype(float64)
    cube_masked = where(informed > 0, Cube, 0.0).astype(float64)

    cube_pad = pad(cube_masked, ((rx, rx), (ry, ry), (rz, rz)), mode="constant")
    inf_pad = pad(informed, ((rx, rx), (ry, ry), (rz, rz)), mode="constant")

    # Cumulative sums with a leading-zero boundary so C[x,y,z] equals the
    # sum over [0,x) x [0,y) x [0,z) of the padded array.
    cc = numpy.zeros((nx + 2 * rx + 1, ny + 2 * ry + 1, nz + 2 * rz + 1))
    cc[1:, 1:, 1:] = cube_pad
    cc = cc.cumsum(0).cumsum(1).cumsum(2)
    ci = numpy.zeros_like(cc)
    ci[1:, 1:, 1:] = inf_pad
    ci = ci.cumsum(0).cumsum(1).cumsum(2)

    # 3D box sum via 8-corner inclusion-exclusion: box [a,b)x[c,d)x[e,f).
    def _box(cs, a, b, c, d, e, f):
        return (
            cs[b, d, f] - cs[a, d, f] - cs[b, c, f] - cs[b, d, e]
            + cs[a, c, f] + cs[a, d, e] + cs[b, c, e] - cs[a, c, e]
        )

    i, j, k = mgrid[0:nx, 0:ny, 0:nz]
    cnt = _box(ci, i, i + 2 * rx, j, j + 2 * ry, k, k + 2 * rz)
    sm = _box(cc, i, i + 2 * rx, j, j + 2 * ry, k, k + 2 * rz)

    out = numpy.full((nx, ny, nz), undefined_value, dtype=float64)
    out = where(cnt > 0, sm / where(cnt > 0, cnt, 1), undefined_value)
    return out.astype(Cube.dtype, copy=False)


def LoadGslibFile(filename, property_size, basedir=None):
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
    basedir : str or pathlib.Path, optional
        Trusted base directory for path containment. Defaults to
        ``PathValidator.DEFAULT_BASE_DIR`` (the process working
        directory at import time).

    Returns
    -------
    dict
        Dictionary mapping property names to 3D numpy arrays with
        shape ``property_size``.

    Raises
    ------
    ValueError
        If ``filename`` is empty/not a string, ``num_p`` is invalid,
        the property count exceeds the maximum, or property names
        are duplicated.
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

    # Security: safe_open_read validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks. The basedir is
    # the trusted base (DEFAULT_BASE_DIR unless overridden) — NOT derived
    # from the filename's own directory, which would defeat containment.
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_open_read(filename, basedir=basedir) as f:
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

        # F-31: reject duplicate property names. Duplicate names cause
        # ``result[name]`` to be overwritten, silently corrupting data
        # (both columns accumulate into the same array). Mirrors the
        # seen-set check in geo.py _load_prop_ind_slow.
        seen = set()
        for _ in range(num_p):
            name = str(f.readline().strip())
            if name in seen:
                raise ValueError(
                    f"LoadGslibFile: duplicate property name '{name}' in GSLIB header. "
                    f"Each property name must be unique."
                )
            seen.add(name)
            list_prop.append(name)

        grid_size = nx * ny * nz

        # I2-05: bound the total number of values parsed by this pure-Python
        # loader. grid_size alone can reach MAX_GRID_SIZE (1e9); multiplying
        # by num_p (up to 1024) previously allowed a 1e12-value file to
        # consume ~8 TB. The cap keeps the worst case at the same footprint
        # a single MAX_GRID_SIZE property already permits.
        if grid_size * num_p > ValidationConstants.MAX_GSLIB_VALUES:
            raise ValueError(
                f"LoadGslibFile: file would contain {grid_size * num_p} values "
                f"(grid {property_size} x {num_p} properties), exceeding the "
                f"maximum allowed {ValidationConstants.MAX_GSLIB_VALUES}. "
                f"File may be corrupted or malicious."
            )

        # I2-05: collect validated rows, then vectorize the string->float64
        # conversion with a single numpy call instead of a per-token Python
        # loop (the previous implementation ran one float64() per token).
        rows: list[list[str]] = []
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
            if len(rows) >= grid_size:
                raise RuntimeError(
                    f"LoadGslibFile: too many values for property '{list_prop[0]}'. "
                    f"Expected {grid_size} elements "
                    f"(grid {property_size}), got more."
                )
            rows.append(points)

        if len(rows) != grid_size:
            raise RuntimeError(
                f"LoadGslibFile: property '{list_prop[0]}' has {len(rows)} values, "
                f"expected {grid_size} (grid {property_size}). "
                f"File may be truncated or corrupted."
            )

        data = numpy.array(rows, dtype=float64)  # shape (grid_size, num_p)
        for j, key in enumerate(list_prop):
            result[key] = data[:, j].reshape(property_size, order="F")

    for dkey in result.keys():
        if not numpy.all(numpy.isfinite(result[dkey])):
            raise ValueError(
                f"LoadGslibFile: property '{dkey}' contains non-finite "
                f"values (NaN or Inf). GSLIB format does not support non-finite values."
            )

    return result
