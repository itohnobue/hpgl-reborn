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

from .gslib_ref import (
    GSLIB_SENTINEL_WINDOW,
    is_gslib_missing_sentinel,
)
from .validation import (
    GridValidator,
    ParameterValidator,
    PathValidator,
    ValidationConstants,
    validate_property_name,
)

# II-42: maximum accepted byte length of a single GSLIB data line, checked
# BEFORE any `line.split()` token materialization. The slow parser in geo.py
# uses the same 1 MB bound (_MAX_SLOW_PARSER_LINE_BYTES) to defend against
# crafted newline-free lines; without a bound, a multi-MB line with hundreds
# of thousands of tokens allocated the full token list (500k CPython strs,
# ~28 MB RSS on a 3.4 MB line — memory-exhaustion DoS) before the
# token-count check below could fire. A legal GSLIB line holds exactly
# num_p (≤ 1024) short tokens, so 1 MB is far above any legitimate line.
MAX_GSLIB_LINE_BYTES = 1_000_000


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
    # III-18: isfinite gate — NaN in an informed cell must raise, not
    # silently propagate. Mirrors the C++ calc_mean.cpp:18-22 guard
    # (`if (!std::isfinite(val)) { *success = false; return NAN; }`).
    # NaN is reachable here via LoadGslibFile's ±1.0e21 sentinel→NaN
    # conversion; without this gate the caller receives a silent NaN mean
    # with no signal (live probe: CalcMean with NaN in an informed cell →
    # nan, zero warnings). Only informed cells matter — masked (zero)
    # cells are excluded by definition.
    if not numpy.all(numpy.isfinite(Cube[(Mask != 0)])):
        raise ValueError(
            "CalcMean: Cube contains NaN or Inf values in informed (Mask != 0) cells"
        )
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
    # III-18: isfinite gate — NaN in an informed cell must raise, not
    # silently propagate (mirror C++ calc_mean.cpp:18-22). Without it a
    # layer containing a NaN informed cell silently returns NaN (live
    # probe: CalcVPC with NaN in an informed cell → nan, zero warnings).
    if not numpy.all(numpy.isfinite(Cube[(Mask != 0)])):
        raise ValueError(
            "CalcVPC: Cube contains NaN or Inf values in informed (Mask != 0) cells"
        )
    # II-41: standardize mask semantics — "non-zero = informed" (the
    # documented contract). A raw `Mask.sum()` denominator treats a mask
    # value of 2 as TWO informed cells, halving the layer mean (live probe:
    # CalcVPC with mask=2 halves every layer). bool-convert before summing.
    MaskSum = (Mask != 0).sum(0).sum(0)
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

    # III-17: equal-shape validation across all cubes (mirror
    # SaveGSLIBCubes' equal-size check). Pre-fix the first cube's shape was
    # taken as canonical with no validation — a (2,2,2) cube alongside a
    # (2,2,3) cube silently truncated layer 2 (probe: layer-2 data lost),
    # and the reverse mismatch raised an opaque IndexError. Every cube must
    # share the same (NX, NY, NZ).
    first_key = next(iter(CubesDictionary.keys()))
    first_shape = CubesDictionary[first_key].shape
    for Key in CubesDictionary.keys():
        if CubesDictionary[Key].shape != first_shape:
            raise ValueError(
                f"Cubes2PointSet: property '{Key}' has shape "
                f"{CubesDictionary[Key].shape}, expected {first_shape} "
                f"(all cubes must have identical shape, mirroring "
                f"SaveGSLIBCubes). Extra Z-layers would otherwise be "
                f"silently truncated."
            )

    NX, NY, NZ = first_shape
    grid_i, grid_j = mgrid[0:NX, 0:NY]

    # II-41: non-zero = informed. A raw `Mask.sum()` would allocate
    # int(Mask.sum()) rows for a mask containing values > 1 (e.g. mask=2 →
    # 16 rows for 8 informed cells) while the fill loop below writes only
    # (Mask != 0).sum() rows, leaving silent trailing zeros in every
    # output array (live probe: 8+8 trailing zeros). bool-convert first.
    total = int((Mask != 0).sum())

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

    # II-41: non-zero = informed (see Cubes2PointSet comment).
    total = int((Mask != 0).sum())
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

    # F-29: ALSO reject FINITE values outside the ±1.0e21 sentinel window
    # (strict inequality per the GSLIB convention — see gslib_ref.py
    # reference-fact table, got-20260802092630). The reader (LoadGslibFile,
    # read_inc_file.cpp) converts those to NaN, so accepting them at write
    # time would silently corrupt the round-trip (probe: SaveGSLIBCubes
    # [2.0e21] → LoadGslibFile [NaN]).
    for Key in PointSet.keys():
        arr = PointSet[Key]
        if not numpy.all(numpy.isfinite(arr)) or numpy.any(is_gslib_missing_sentinel(arr)):
            raise ValueError(
                f"SaveGSLIBPointSet: property '{Key}' contains non-finite values "
                f"(NaN or Inf) or values outside the GSLIB ±{GSLIB_SENTINEL_WINDOW:g} "
                f"sentinel window (|v| > {GSLIB_SENTINEL_WINDOW:g}). GSLIB format "
                f"cannot represent them; the reader would convert them to NaN."
            )

    # F-N17: validate the Caption and every property name against the shared
    # property-name contract (validate_property_name, shared with the C++
    # writers). A name/caption containing '\n' or other control characters
    # would inject phantom header lines that LoadGslibFile (or C++ readers)
    # mis-parses; a '--' prefix is a comment marker that readers skip, and
    # leading/trailing whitespace shifts the data off-by-one. Reject before
    # any file is created.
    validate_property_name(Caption, "SaveGSLIBPointSet")
    for Key in PointSet.keys():
        validate_property_name(Key, "SaveGSLIBPointSet")

    # Security: safe_atomic_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks. The basedir is
    # the trusted base (DEFAULT_BASE_DIR unless overridden) — NOT derived
    # from the filename's own directory, which would defeat containment.
    # Content is written to a uniquely-named temp file and atomically renamed
    # into place on clean exit, so a crash mid-write never truncates the
    # target (F-N18, matching the C++ writers' temp+rename pattern).
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_atomic_open_write(FileName, basedir=basedir) as f:
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

    # F-29: ALSO reject FINITE values outside the ±1.0e21 sentinel window
    # (strict inequality per the GSLIB convention — see gslib_ref.py
    # reference-fact table, got-20260802092630). The reader (LoadGslibFile,
    # read_inc_file.cpp) converts those to NaN, so accepting them at write
    # time would silently corrupt the round-trip.
    for Key in CubesDictionary.keys():
        arr = CubesDictionary[Key]
        if not numpy.all(numpy.isfinite(arr)) or numpy.any(is_gslib_missing_sentinel(arr)):
            raise ValueError(
                f"SaveGSLIBCubes: property '{Key}' contains non-finite values "
                f"(NaN or Inf) or values outside the GSLIB ±{GSLIB_SENTINEL_WINDOW:g} "
                f"sentinel window (|v| > {GSLIB_SENTINEL_WINDOW:g}). GSLIB format "
                f"cannot represent them; the reader would convert them to NaN."
            )

    # F-N17: validate the Caption and every property name against the shared
    # property-name contract (validate_property_name, shared with the C++
    # writers). A name/caption containing '\n' or other control characters
    # would inject phantom header lines that LoadGslibFile (or C++ readers)
    # mis-parses; a '--' prefix is a comment marker that readers skip, and
    # leading/trailing whitespace shifts the data off-by-one. Reject before
    # any file is created.
    validate_property_name(Caption, "SaveGSLIBCubes")
    for Key in CubesDictionary.keys():
        validate_property_name(Key, "SaveGSLIBCubes")

    # Security: safe_atomic_open_write validates the path and opens atomically
    # with O_NOFOLLOW to prevent TOCTOU symlink attacks. The basedir is
    # the trusted base (DEFAULT_BASE_DIR unless overridden) — NOT derived
    # from the filename's own directory, which would defeat containment.
    # Content is written to a uniquely-named temp file and atomically renamed
    # into place on clean exit, so a crash mid-write never truncates the
    # target (F-N18, matching the C++ writers' temp+rename pattern).
    if basedir is None:
        basedir = PathValidator.DEFAULT_BASE_DIR
    with PathValidator.safe_atomic_open_write(FileName, basedir=basedir) as f:
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
    rx, ry, rz = Radiuses
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError(
            f"GetCubicalMask: radius components must be positive, "
            f"got ({rx}, {ry}, {rz})"
        )
    MeanMask = ones((rx * 2, ry * 2, rz * 2), dtype=uint8)
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

    # II-41: non-zero = informed. The `Mask == 1` tests treated a legal
    # mask value > 1 (e.g. mask=2) as UNINFORMED, returning
    # undefined_value for every cell in a mask=2 grid (live probe:
    # MovingAverage3D with mask=2 → every cell undefined). The MeanMask
    # window itself is binary by construction (GetCubicalMask/GetEllipseMask
    # emit 0/1), so only the data Mask needs the non-zero test.
    informed_window = (Mask[imin:imax, jmin:jmax, kmin:kmax] != 0) & (
        MeanMask[
            i_offset : (i_offset + imax - imin),
            j_offset : (j_offset + jmax - jmin),
            k_offset : (k_offset + kmax - kmin),
        ]
        == 1
    )

    if sum(informed_window) > 0:
        return Cube[imin:imax, jmin:jmax, kmin:kmax][nonzero(informed_window)].mean()
    else:
        return undefined_value


def MovingAverage3D(cube_mask, Radiuses, undefined_value, MaskCalcFunction):
    Cube, Mask = cube_mask
    MACube = copy(Cube)

    # F-M16: validate radiuses BEFORE any allocation. GetCubicalMask had no
    # validation at all (0/negative radiuses produced empty-dim masks or raw
    # numpy errors), and an unbounded radius (r=1e6 is within MAX_RADIUS)
    # previously attempted a (2e6)^3 = 8e18-byte mask allocation before the
    # volume cap could fire. The mask is (2rx·2ry·2rz) cells, so a radius
    # larger than the grid in any axis would only allocate unused memory —
    # reject it here with a clear error.
    Radiuses = ParameterValidator.validate_radius(Radiuses, "Radiuses")
    rx, ry, rz = Radiuses
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError(
            f"MovingAverage3D: radius components must be positive, "
            f"got ({rx}, {ry}, {rz})"
        )
    nx, ny, nz = Cube.shape
    if rx > nx or ry > ny or rz > nz:
        raise ValueError(
            f"MovingAverage3D: radius ({rx}, {ry}, {rz}) exceeds the grid "
            f"({nx}, {ny}, {nz}) in at least one axis. The mean-mask window "
            f"is clamped to the grid, so a larger radius is never used."
        )

    # F-M16: grid-volume cap BEFORE the mask allocation. The cubical
    # (vectorized) path was previously unbounded because it returned before
    # this check; the mask allocation itself is radius-driven, so both paths
    # now share the bound and an over-cap grid is rejected before the mask
    # (and the cubical integral images) are ever allocated.
    volume = int(nx) * int(ny) * int(nz)
    if volume > ValidationConstants.MAX_MOVING_AVERAGE_VOLUME:
        raise ValueError(
            f"MovingAverage3D: grid volume {volume} exceeds the maximum "
            f"{ValidationConstants.MAX_MOVING_AVERAGE_VOLUME} supported by "
            f"the pure-Python moving-average path. Reduce the grid size."
        )

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

    # F-30: work-based cap on the per-cell ellipse path. The grid-volume
    # cap above bounds MEMORY only; this loop is O(N·V) — every one of the
    # N grid cells runs MeanCalc over a (2rx·2ry·2rz) mean-mask window,
    # ~10 numpy ops per window cell. With the volume cap at 1e6 and a
    # radius reaching the full grid, the worst case is ~8e12 numpy ops
    # (multi-hour hang; probe: r=2 → 12.5 s at cap, r=50 extrapolates to
    # ~54 h). The C++ kernel has no Python-scale cost problem (compiled);
    # this pure-Python path needs an explicit work bound. Reject the
    # product N × window_volume before the loop starts.
    window_volume = (2 * rx) * (2 * ry) * (2 * rz)
    moving_average_work = volume * window_volume
    if moving_average_work > ValidationConstants.MAX_MOVING_AVERAGE_WORK:
        raise ValueError(
            f"MovingAverage3D: estimated per-cell work {moving_average_work} "
            f"(grid volume {volume} × mean-mask volume {window_volume}) exceeds "
            f"the maximum {ValidationConstants.MAX_MOVING_AVERAGE_WORK} for the "
            f"per-cell (ellipse-mask) path. Reduce the grid size or Radiuses."
        )

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
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
    # II-41: non-zero = informed (the documented mask contract). The old
    # `Mask == 1` treated mask=2 cells as uninformed, so a legal non-binary
    # mask produced an undefined_value grid even though every cell was
    # informed (live probe: MovingAverage3D mask=2 → every cell undefined).
    informed = (Mask != 0).astype(float64)
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

        # I2-05/2-M-10: parse the data lines INCREMENTALLY into a preallocated
        # float64 array instead of collecting the whole file into a list of
        # lists of Python strings first. The old implementation materialized
        # every token as a CPython str (~50-65 B each) plus per-row list
        # overhead before a single numpy conversion — roughly 10-20x the
        # memory of the final array, so a legal large file (e.g. 1e8 values)
        # OOM'd. Row-by-row assignment keeps intermediate Python-object
        # memory bounded to a single line while numpy still performs the
        # string->float64 conversion.
        #
        # R-12: count the data rows BEFORE allocating the output array. The
        # pre-fix streaming fix allocated numpy.empty((grid_size, num_p))
        # up front, so a TRUNCATED file with a large declared grid allocated
        # the full array (1.6-8 GB for a 1e8-cell grid) and only then raised
        # the row-count error below. Count non-blank data lines first and
        # raise the same row-count errors the parse loop raises, but before
        # the allocation — the truncation error path then fails cleanly at
        # KBs of memory. The success path is unchanged (the count pass is a
        # cheap line scan; the parse then streams into the preallocated
        # array exactly as before).
        data_start = f.tell()
        row_count = 0
        for line in f:
            if line.strip():
                row_count += 1
        if row_count > grid_size:
            raise RuntimeError(
                f"LoadGslibFile: too many values for property '{list_prop[0]}'. "
                f"Expected {grid_size} elements "
                f"(grid {property_size}), got more."
            )
        if row_count != grid_size:
            raise RuntimeError(
                f"LoadGslibFile: property '{list_prop[0]}' has {row_count} values, "
                f"expected {grid_size} (grid {property_size}). "
                f"File may be truncated or corrupted."
            )
        f.seek(data_start)

        data = numpy.empty((grid_size, num_p), dtype=float64)
        row_idx = 0
        for line in f:
            # Skip blank lines (whitespace-only)
            if not line.strip():
                continue
            # II-42: bound the line length and count tokens BEFORE
            # `line.split()` materializes the token list. A crafted
            # newline-free line (probe: 3.4 MB → 500k tokens → 28.3 MB RSS)
            # previously allocated every token as a CPython str before the
            # count check could fire. A legal GSLIB line has exactly num_p
            # (≤ 1024) short tokens, so both bounds are far above legitimate
            # data while catching the DoS input before any allocation.
            if len(line) > MAX_GSLIB_LINE_BYTES:
                raise RuntimeError(
                    f"LoadGslibFile: data line exceeds {MAX_GSLIB_LINE_BYTES} bytes. "
                    f"File may be corrupted or malicious."
                )
            token_count = 0
            in_token = False
            for ch in line:
                if ch.isspace():
                    in_token = False
                elif not in_token:
                    token_count += 1
                    if token_count > num_p:
                        break
                    in_token = True
            if token_count != num_p:
                raise RuntimeError(
                    f"LoadGslibFile: expected {num_p} values per data line, "
                    f"got {token_count} tokens in line: {line.strip()!r}"
                )
            points = line.split()
            # Validate token count matches expected property count per line
            # (defense in depth — the pre-split count above already checked,
            # but split() is the authoritative tokenization).
            if len(points) != num_p:
                raise RuntimeError(
                    f"LoadGslibFile: expected {num_p} values per data line, "
                    f"got {len(points)} tokens in line: {line.strip()!r}"
                )
            if row_idx >= grid_size:
                raise RuntimeError(
                    f"LoadGslibFile: too many values for property '{list_prop[0]}'. "
                    f"Expected {grid_size} elements "
                    f"(grid {property_size}), got more."
                )
            data[row_idx] = points
            row_idx += 1

        # Reject genuine non-finite source values (NaN or Inf) BEFORE the
        # F-M18 sentinel trim — GSLIB cannot represent them and HPGL-written
        # files never contain them (the writer rejects non-finite at write
        # time). The trim below converts only FINITE out-of-window values.
        for j, key in enumerate(list_prop):
            if not numpy.all(numpy.isfinite(data[:, j])):
                raise ValueError(
                    f"LoadGslibFile: property '{key}' contains non-finite "
                    f"values (NaN or Inf). GSLIB format does not support non-finite values."
                )

        # F-M18: GSLIB missing-value trimming — finite values outside the
        # ±1.0e21 window (strict inequality per the GSLIB convention "less
        # than -1.0e21 or greater than 1.0e21") are missing sentinels, not
        # real data. Convert them to NaN (the numpy missing marker) so
        # downstream mean/variogram/kriging do not silently compute with
        # third-party sentinel magnitudes. Matches the C++ fast reader
        # (read_inc_file.cpp:287-305) and get_gslib_property.
        out_of_window = is_gslib_missing_sentinel(data)
        data[out_of_window] = numpy.nan

        for j, key in enumerate(list_prop):
            result[key] = data[:, j].reshape(property_size, order="F")

    return result
