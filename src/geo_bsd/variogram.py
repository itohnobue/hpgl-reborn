# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import warnings

import numpy
from numpy import (
    array,
    bitwise_and,
    ceil,
    column_stack,
    cos,
    dot,
    float32,
    float64,
    floor,
    mgrid,
    ones,
    power,
    prod,
    radians,
    ravel_multi_index,
    repeat,
    reshape,
    sin,
    sum,
    vstack,
    zeros,
)

MAX_NUM_LAGS = 10000


class TVEllipsoid:
    """Three-axis ellipsoid defined by ranges and rotation angles.

    Defines the anisotropy directions for variogram search. The three
    axes are computed by applying azimuth, dip, and rotation angles
    to the coordinate system.

    Parameters
    ----------
    R1 : float
        Range along the first (major) axis.
    R2 : float
        Range along the second axis.
    R3 : float
        Range along the third (minor) axis.
    Azimut : float, optional
        Azimuth angle in degrees (default 0).
    Dip : float, optional
        Dip angle in degrees (default 0).
    Rotation : float, optional
        Rotation angle in degrees (default 0).

    Attributes
    ----------
    Direction1 : numpy.ndarray
        Direction vector of the first axis.
    Direction2 : numpy.ndarray
        Direction vector of the second axis.
    Direction3 : numpy.ndarray
        Direction vector of the third axis.
    R1, R2, R3 : float
        Range values stored as instance attributes.
    """

    Direction1 = [1, 0, 0]
    Direction2 = [0, 1, 0]
    Direction3 = [0, 0, 1]
    R1 = 1
    R2 = 1
    R3 = 1

    def __init__(self, R1, R2, R3, Azimut=0, Dip=0, Rotation=0):
        if R1 < 0 or R2 < 0 or R3 < 0:
            raise ValueError(
                f"TVEllipsoid: ranges must not be negative, got R1={R1!r}, R2={R2!r}, R3={R3!r}"
            )
        Azimut = radians(Azimut)
        Dip = radians(Dip)
        Rotation = radians(Rotation)

        A = array([[cos(Azimut), -sin(Azimut), 0], [sin(Azimut), cos(Azimut), 0], [0, 0, 1]])

        B = array([[cos(Dip), 0, -sin(Dip)], [0, 1, 0], [sin(Dip), 0, cos(Dip)]])

        C = array(
            [[1, 0, 0], [0, cos(Rotation), -sin(Rotation)], [0, sin(Rotation), cos(Rotation)]]
        )

        ABC = A @ B @ C

        self.Direction1 = ABC[:, 0]
        self.Direction2 = ABC[:, 1]
        self.Direction3 = ABC[:, 2]

        self.R1 = R1
        self.R2 = R2
        self.R3 = R3


class TVVariogramSearchTemplate:
    """Parameters controlling experimental variogram computation.

    Defines the lag geometry (width, separation, tolerance) and the
    ellipsoid search volume for scanning point pairs.

    Parameters
    ----------
    LagWidth : float
        Width of each lag band.
    LagSeparation : float
        Distance between consecutive lag centers.
    TolDistance : float
        Angular tolerance multiplier for the tunnel filter.
    NumLags : int
        Number of lags to compute.
    Ellipsoid : TVEllipsoid
        Ellipsoid defining anisotropy directions and ranges.
    FirstLagDistance : float, optional
        Distance to the start of the first lag (default 0).

    Attributes
    ----------
    LagWidth : float
    LagSeparation : float
    TolDistance : float
    NumLags : int
    Ellipsoid : TVEllipsoid
    FirstLagDistance : float
    """

    LagSeparation = 1
    TolDistance = 1
    NumLags = 10
    Ellipsoid = TVEllipsoid(1, 1, 1)

    def __init__(
        self, LagWidth, LagSeparation, TolDistance, NumLags, Ellipsoid, FirstLagDistance=0
    ):
        if NumLags <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: NumLags must be positive, got {NumLags}"
            )
        if NumLags > MAX_NUM_LAGS:
            raise ValueError(
                f"TVVariogramSearchTemplate: NumLags {NumLags} exceeds maximum {MAX_NUM_LAGS}"
            )
        if LagWidth <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: LagWidth must be positive, got {LagWidth}"
            )
        if LagSeparation <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: LagSeparation must be positive, got {LagSeparation}"
            )
        if TolDistance <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: TolDistance must be positive, got {TolDistance}"
            )
        if FirstLagDistance < 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: FirstLagDistance must be non-negative, got {FirstLagDistance}"
            )
        self.LagWidth = LagWidth
        self.LagSeparation = LagSeparation
        self.TolDistance = TolDistance
        self.NumLags = NumLags
        self.Ellipsoid = Ellipsoid
        self.FirstLagDistance = FirstLagDistance


def _IsInTunnel(VariogramSearchTemplate, V):
    # Compute projections of each vector onto ellipsoid axes via dot product.
    # V is expected as (N, 3) where each row is a displacement vector (dx, dy, dz).
    # Returns a 1D boolean array of length N indicating which vectors are inside the tunnel.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    D2 = VariogramSearchTemplate.Ellipsoid.Direction2
    D3 = VariogramSearchTemplate.Ellipsoid.Direction3

    # Ensure V is 2D for consistent dot product behavior
    V = array(V, ndmin=2)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"_IsInTunnel: V must have shape (N, 3), got {V.shape}")

    SS1 = abs(dot(V, D1))
    SS2 = abs(dot(V, D2))
    SS3 = abs(dot(V, D3))

    if (
        VariogramSearchTemplate.Ellipsoid.R1 == 0
        or VariogramSearchTemplate.Ellipsoid.R2 == 0
        or VariogramSearchTemplate.Ellipsoid.R3 == 0
    ):
        warnings.warn(
            f"_IsInTunnel: R1={VariogramSearchTemplate.Ellipsoid.R1!r}, "
            f"R2={VariogramSearchTemplate.Ellipsoid.R2!r}, "
            f"R3={VariogramSearchTemplate.Ellipsoid.R3!r} — at least one range is zero. "
            f"Returning all-False result (no vectors in tunnel).",
            stacklevel=2,
        )
        return zeros(V.shape[0], dtype=bool)

    S1 = SS1 / VariogramSearchTemplate.Ellipsoid.R1
    S2 = SS2 / VariogramSearchTemplate.Ellipsoid.R2
    S3 = SS3 / VariogramSearchTemplate.Ellipsoid.R3

    Dist = power(power(S2, 2) + power(S3, 2), 0.5)
    Result = array(bitwise_and(Dist <= 1, VariogramSearchTemplate.TolDistance * Dist <= S1))

    return Result.ravel()


def _CalcSearchTemplateWindow(VariogramSearchTemplate):
    Max = 1e10
    MinI = Max
    MaxI = -Max
    MinJ = Max
    MaxJ = -Max
    MinK = Max
    MaxK = -Max
    for i in range(0, 2):
        for j in range(-1, 2, 2):
            for k in range(-1, 2, 2):
                DI = (
                    VariogramSearchTemplate.Ellipsoid.Direction1
                    * (
                        VariogramSearchTemplate.LagSeparation
                        * VariogramSearchTemplate.NumLags
                        + VariogramSearchTemplate.FirstLagDistance
                        + VariogramSearchTemplate.LagWidth / 2
                    )
                    * i
                )
                DJ = (
                    VariogramSearchTemplate.Ellipsoid.Direction2
                    * VariogramSearchTemplate.Ellipsoid.R2
                    * j
                )
                DK = (
                    VariogramSearchTemplate.Ellipsoid.Direction3
                    * VariogramSearchTemplate.Ellipsoid.R3
                    * k
                )
                V = DI + DJ + DK

                MinI = float(min(MinI, V[0]))
                MaxI = float(max(MaxI, V[0]))
                MinJ = float(min(MinJ, V[1]))
                MaxJ = float(max(MaxJ, V[1]))
                MinK = float(min(MinK, V[2]))
                MaxK = float(max(MaxK, V[2]))
    return MinI, MinJ, MinK, MaxI, MaxJ, MaxK


def _CalcLagDistances(VariogramSearchTemplate):
    LagIndexes = range(0, VariogramSearchTemplate.NumLags)
    LagDistance = (
        array(LagIndexes) * VariogramSearchTemplate.LagSeparation
        + VariogramSearchTemplate.FirstLagDistance
    )
    LagWidth = VariogramSearchTemplate.LagWidth
    LagStart = LagDistance - float(LagWidth) / 2
    LagEnd = LagDistance + float(LagWidth) / 2

    return LagIndexes, LagDistance, LagStart, LagEnd


def _CalcLagsAreas(VariogramSearchTemplate):
    (MinI, MinJ, MinK, MaxI, MaxJ, MaxK) = _CalcSearchTemplateWindow(VariogramSearchTemplate)
    MinI = int(floor(MinI))
    MinJ = int(floor(MinJ))
    MinK = int(floor(MinK))
    MaxI = int(ceil(MaxI))
    MaxJ = int(ceil(MaxJ))
    MaxK = int(ceil(MaxK))

    idx_i = zeros([])
    idx_j = zeros([])
    idx_k = zeros([])
    lag_indexes = zeros([])

    (Index, LagDistance, LagStart, LagEnd) = _CalcLagDistances(VariogramSearchTemplate)

    GI, GJ, GK = mgrid[MinI : MaxI + 1, MinJ : MaxJ + 1, MinK : MaxK + 1]

    GI = GI.reshape(prod(GI.shape), 1)
    GJ = GJ.reshape(prod(GJ.shape), 1)
    GK = GK.reshape(prod(GK.shape), 1)

    ActivePoints = _IsInTunnel(VariogramSearchTemplate, column_stack((GI, GJ, GK)))

    GI = GI[ActivePoints]
    GJ = GJ[ActivePoints]
    GK = GK[ActivePoints]

    Dist = power(power(GI, 2) + power(GJ, 2) + power(GK, 2), 0.5)

    # Accumulate per-lag results in Python lists to avoid O(L²) vstack copies
    idx_i_parts = []
    idx_j_parts = []
    idx_k_parts = []
    lag_indexes_parts = []

    for i in Index:
        Filter = bitwise_and(LagStart[i] <= Dist, Dist < LagEnd[i])
        NumPoints = sum(Filter)
        if NumPoints == 0:
            continue
        idx_i_parts.append(GI[Filter].reshape(NumPoints, 1))
        idx_j_parts.append(GJ[Filter].reshape(NumPoints, 1))
        idx_k_parts.append(GK[Filter].reshape(NumPoints, 1))
        lag_indexes_parts.append(ones((NumPoints, 1), dtype=float32) * i)

    if idx_i_parts:
        idx_i = vstack(idx_i_parts).astype(int).ravel()
        idx_j = vstack(idx_j_parts).astype(int).ravel()
        idx_k = vstack(idx_k_parts).astype(int).ravel()
        lag_indexes = vstack(lag_indexes_parts).astype(int).ravel()
    else:
        idx_i = zeros(0, dtype=int)
        idx_j = zeros(0, dtype=int)
        idx_k = zeros(0, dtype=int)
        lag_indexes = zeros(0, dtype=int)
    return idx_i, idx_j, idx_k, lag_indexes, LagDistance


def _verify_dict_keys(d, required_keys, name="dict"):
    """Validate that a dictionary contains all required keys.

    Args:
        d: The dictionary to check.
        required_keys: List of required key names.
        name: Name of the dict for error messages.

    Raises:
        TypeError: If d is not a dict.
        KeyError: If any required key is missing, with an explicit message
            listing which keys were not found.
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected {name} to be a dict, got {type(d).__name__}")
    missing = [repr(k) for k in required_keys if k not in d]
    if missing:
        raise KeyError(f"{name} is missing required key(s): {', '.join(missing)}")


def _verify_shape(obj, ndim, name="array"):
    """Validate that an object has a .shape attribute with expected dimensions.

    Args:
        obj: Object to validate (expects numpy array or compatible).
        ndim: Expected number of dimensions.
        name: Name of the object for error messages.

    Raises:
        AttributeError: If obj has no .shape attribute.
        ValueError: If obj.shape has wrong number of dimensions.
    """
    if not hasattr(obj, "shape"):
        raise AttributeError(
            f"Expected {name} to have a .shape attribute, got {type(obj).__name__}"
        )
    shp = obj.shape
    if len(shp) != ndim:
        raise ValueError(f"Expected {name}.shape to have {ndim} dimensions, got {len(shp)}: {shp}")


def PointSetScanContStyle(VariogramSearchTemplate, PointSet, Function, Params):
    """Compute experimental variogram/covariance from scattered point data.

    Scans all point pairs within the search template, accumulating
    pairwise statistics through the provided Function callback.

    Parameters
    ----------
    VariogramSearchTemplate : TVVariogramSearchTemplate
        Template defining search geometry and lag configuration.
    PointSet : dict
        Dictionary with keys 'X', 'Y', 'Z', each mapping to 1D arrays
        of point coordinates.
    Function : callable or None
        Callback ``f(point1_idx, point2_idx, result_accum, params)``
        that accumulates statistics for each point pair. If None,
        initializes and returns the result array without populating it.
    Params : dict or None
        User-supplied parameters forwarded to Function.

    Returns
    -------
    Result : numpy.ndarray
        Accumulated result array of shape ``(NumLags, NumValues)``.
    LagDistance : numpy.ndarray
        Distance values for each lag center.

    Notes
    -----
    Uses a spatial indexing approach: for each point, candidate
    neighbors are filtered first by bounding box, then by distance
    range, and finally by the ellipsoid tunnel test (``_IsInTunnel``).
    """
    _verify_dict_keys(PointSet, ["X", "Y", "Z"], "PointSet")
    PX = PointSet["X"]
    PY = PointSet["Y"]
    PZ = PointSet["Z"]

    MinX, MinY, MinZ, MaxX, MaxY, MaxZ = _CalcSearchTemplateWindow(VariogramSearchTemplate)

    LagIndex, LagDistance, LagStart, LagEnd = _CalcLagDistances(VariogramSearchTemplate)
    MinDistance2 = max(0, LagStart.min()) ** 2
    MaxDistance2 = LagEnd.max() ** 2

    if Function is not None:
        Result = Function(0, 0, None, Params)
        Result = reshape(Result, (1, len(Result)))
        Result = repeat(Result, VariogramSearchTemplate.NumLags, 0)
    else:
        return zeros((VariogramSearchTemplate.NumLags, 1)), LagDistance

    Index = array(range(0, len(PX)))
    for i in range(len(PX)):
        X1, Y1, Z1 = PX[i], PY[i], PZ[i]
        DX, DY, DZ = PX - X1, PY - Y1, PZ - Z1

        Filter = MinX <= DX
        Filter = bitwise_and(Filter, DX <= MaxX)
        Filter = bitwise_and(Filter, MinY <= DY)
        Filter = bitwise_and(Filter, DY <= MaxY)
        Filter = bitwise_and(Filter, MinZ <= DZ)
        Filter = bitwise_and(Filter, DZ <= MaxZ)

        FDX, FDY, FDZ = DX[Filter], DY[Filter], DZ[Filter]
        FIndex = Index[Filter]

        FDistance2 = FDX**2 + FDY**2 + FDZ**2
        Filter = MinDistance2 <= FDistance2
        Filter = bitwise_and(Filter, FDistance2 <= MaxDistance2)

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance2 = FDistance2[Filter]

        Filter = _IsInTunnel(VariogramSearchTemplate, column_stack((FDX, FDY, FDZ)))

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance2 = FDistance2[Filter]

        FDistance = FDistance2**0.5

        for Lag in LagIndex:
            Filter = bitwise_and(LagStart[Lag] <= FDistance, FDistance < LagEnd[Lag])
            for j in FIndex[Filter]:
                Result[Lag, :] = Function([i], [j], Result[Lag, :], Params)

    return Result, LagDistance


def PointSetScanGridStyle(VariogramSearchTemplate, PointSetXYZ, Function, Params):
    """Compute variogram from grid-aligned point data using lag index lookups.

    Unlike ``PointSetScanContStyle``, this function uses precomputed
    lag area indices (via ``_CalcLagsAreas``) to match point pairs
    by exact grid-cell offsets rather than continuous distance.

    Parameters
    ----------
    VariogramSearchTemplate : TVVariogramSearchTemplate
        Template defining search geometry and lag configuration.
    PointSetXYZ : tuple of numpy.ndarray
        Tuple ``(X, Y, Z)`` of 1D coordinate arrays.
    Function : callable or None
        Callback accumulating pairwise statistics. See
        ``PointSetScanContStyle`` for signature details.
    Params : dict or None
        User-supplied parameters forwarded to Function.

    Returns
    -------
    Result : numpy.ndarray
        Accumulated result array of shape ``(NumLags, NumValues)``.
    LagDistance : numpy.ndarray
        Distance values for each lag center.
    """
    if isinstance(PointSetXYZ, dict):
        _verify_dict_keys(PointSetXYZ, [0, 1, 2], "PointSetXYZ")
    elif not hasattr(PointSetXYZ, "__len__") or len(PointSetXYZ) < 3:
        # PointSetXYZ is typically a tuple (X, Y, Z) of coordinate arrays
        raise KeyError(
            f"PointSetXYZ must have at least 3 elements (X, Y, Z), "
            f"got {type(PointSetXYZ).__name__}"
            f"{' with length ' + str(len(PointSetXYZ)) if hasattr(PointSetXYZ, '__len__') else ''}"
        )
    LI, LJ, LK, LagIndexes, LagDistance = _CalcLagsAreas(VariogramSearchTemplate)
    IMin, IMax = LI.min(), LI.max()
    JMin, JMax = LJ.min(), LJ.max()
    KMin, KMax = LK.min(), LK.max()

    PI = PointSetXYZ[0]
    PJ = PointSetXYZ[1]
    PK = PointSetXYZ[2]

    if Function is not None:
        Result = Function(0, 0, None, Params)
        Result = reshape(Result, (1, len(Result)))
        Result = repeat(Result, VariogramSearchTemplate.NumLags, 0)
    else:
        return zeros((VariogramSearchTemplate.NumLags, 1)), LagDistance

    Index = array(range(0, len(PI)))
    for i in range(len(PI)):
        I1, J1, K1 = PI[i], PJ[i], PK[i]
        DI, DJ, DK = PI - I1, PJ - J1, PK - K1

        Filter = IMin <= DI
        Filter = bitwise_and(Filter, DI <= IMax)
        Filter = bitwise_and(Filter, JMin <= DJ)
        Filter = bitwise_and(Filter, DJ <= JMax)
        Filter = bitwise_and(Filter, KMin <= DK)
        Filter = bitwise_and(Filter, DK <= KMax)

        FPI, FPJ, FPK = PI[Filter], PJ[Filter], PK[Filter]
        FDI, FDJ, FDK = FPI - I1, FPJ - J1, FPK - K1
        FIndex = Index[Filter]

        for j in range(len(FDI)):
            LFilter = FDI[j] == LI
            LFilter = bitwise_and(LFilter, FDJ[j] == LJ)
            LFilter = bitwise_and(LFilter, FDK[j] == LK)

            ActiveLags = LagIndexes[LFilter]

            if Function is not None:
                for Lag in ActiveLags:
                    Result[Lag, :] = Function(i, FIndex[j], Result[Lag, :], Params)

    return Result, LagDistance


def CubeScan(VariogramSearchTemplate, Mask, Function, Params):
    """Compute variogram from a dense 3D grid using precomputed lag offsets.

    Iterates over precomputed lag offset pairs ``(DI, DJ, DK)`` and
    computes the intersection of the shifted mask to identify valid
    cell pairs.

    Parameters
    ----------
    VariogramSearchTemplate : TVVariogramSearchTemplate
        Template defining search geometry and lag configuration.
    Mask : numpy.ndarray
        3D boolean or integer array where non-zero indicates an
        informed (valid) cell.
    Function : callable or None
        Callback accumulating pairwise statistics. Receives two tuples
        of (I, J, K) index arrays for matched cells.
    Params : dict or None
        User-supplied parameters forwarded to Function.

    Returns
    -------
    Result : numpy.ndarray
        Accumulated result array of shape ``(NumLags, NumValues)``.
    LagDistance : numpy.ndarray
        Distance values for each lag center.
    """
    _verify_shape(Mask, 3, "Mask")
    NI, NJ, NK = Mask.shape

    LI, LJ, LK, LagIndexes, LagDistance = _CalcLagsAreas(VariogramSearchTemplate)

    if len(LagIndexes) == 0:
        raise ValueError(
            "CubeScan: _CalcLagsAreas returned empty LagIndexes — "
            "no lag offsets found. Check search template parameters "
            "(NumLags, LagSeparation, LagWidth, Ellipsoid ranges)."
        )

    if Function is not None:
        Result = Function(0, 0, None, Params)
        Result = reshape(Result, (1, len(Result)))
        Result = repeat(Result, VariogramSearchTemplate.NumLags, 0)
    else:
        return zeros((VariogramSearchTemplate.NumLags, 1)), LagDistance

    GI, GJ, GK = mgrid[0:NI, 0:NJ, 0:NK]

    for i in range(len(LagIndexes)):
        DI = LI[i]
        DJ = LJ[i]
        DK = LK[i]
        Lag = LagIndexes[i]

        Mask1 = Mask[
            max(0, 0 + DI) : min(NI, NI + DI),
            max(0, 0 + DJ) : min(NJ, NJ + DJ),
            max(0, 0 + DK) : min(NK, NK + DK),
        ]
        Mask2 = Mask[
            max(0 - DI, 0) : min(NI - DI, NI),
            max(0 - DJ, 0) : min(NJ - DJ, NJ),
            max(0 - DK, 0) : min(NK - DK, NK),
        ]

        Intersection = Mask1 & Mask2

        GI1 = GI[
            max(0, 0 + DI) : min(NI, NI + DI),
            max(0, 0 + DJ) : min(NJ, NJ + DJ),
            max(0, 0 + DK) : min(NK, NK + DK),
        ]
        GJ1 = GJ[
            max(0, 0 + DI) : min(NI, NI + DI),
            max(0, 0 + DJ) : min(NJ, NJ + DJ),
            max(0, 0 + DK) : min(NK, NK + DK),
        ]
        GK1 = GK[
            max(0, 0 + DI) : min(NI, NI + DI),
            max(0, 0 + DJ) : min(NJ, NJ + DJ),
            max(0, 0 + DK) : min(NK, NK + DK),
        ]

        GI2 = GI[
            max(0 - DI, 0) : min(NI - DI, NI),
            max(0 - DJ, 0) : min(NJ - DJ, NJ),
            max(0 - DK, 0) : min(NK - DK, NK),
        ]
        GJ2 = GJ[
            max(0 - DI, 0) : min(NI - DI, NI),
            max(0 - DJ, 0) : min(NJ - DJ, NJ),
            max(0 - DK, 0) : min(NK - DK, NK),
        ]
        GK2 = GK[
            max(0 - DI, 0) : min(NI - DI, NI),
            max(0 - DJ, 0) : min(NJ - DJ, NJ),
            max(0 - DK, 0) : min(NK - DK, NK),
        ]

        for k in range(Intersection.shape[2]):
            GI1Slice = GI1[:, :, k][Intersection[:, :, k]].flatten()
            GJ1Slice = GJ1[:, :, k][Intersection[:, :, k]].flatten()
            GK1Slice = GK1[:, :, k][Intersection[:, :, k]].flatten()
            GI2Slice = GI2[:, :, k][Intersection[:, :, k]].flatten()
            GJ2Slice = GJ2[:, :, k][Intersection[:, :, k]].flatten()
            GK2Slice = GK2[:, :, k][Intersection[:, :, k]].flatten()

            Result[Lag, :] = Function(
                (GI1Slice, GJ1Slice, GK1Slice),
                (GI2Slice, GJ2Slice, GK2Slice),
                Result[Lag, :],
                Params,
            )

    return Result, LagDistance


def CalcVariogramFunction(Point1, Point2, Result, Params):
    _verify_dict_keys(Params, ["HardData"], "Params")
    Values = Params["HardData"]
    NumValues = len(Values)

    # Validate that input data does not contain NaN/Inf values.
    for iv, arr in enumerate(Values):
        if not numpy.all(numpy.isfinite(arr)):
            raise ValueError(
                f"CalcVariogramFunction: HardData[{iv}] contains NaN or Inf values"
            )

    if Result is None:
        Result = zeros(NumValues + NumValues + 1, dtype=float64)
    else:
        # Normalize Point1/Point2 to handle both scalar (PointSetScanGridStyle),
        # list (PointSetScanContStyle), and tuple-of-arrays (CubeScan) inputs.
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays
            NumPoints = len(Point1[0])
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)
            Point1 = P1
            Point2 = P2

        Values1 = zeros((NumValues, NumPoints))
        Values2 = zeros((NumValues, NumPoints))
        for i in range(NumValues):
            Values1[i] = Values[i][Point1[:]]
            Values2[i] = Values[i][Point2[:]]
        Variances = float32(Values1 - Values2) ** 2
        Result[NumValues + 0 : NumValues + NumValues] = Result[
            NumValues + 0 : NumValues + NumValues
        ] + Variances.sum(axis=1)
        Result[NumValues + NumValues] += Variances.shape[1]
        # Guard against division by zero when a lag has no point pairs.
        # Without this, empty lags produce NaN results.
        if Result[NumValues + NumValues] > 0:
            Result[0:NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] / Result[NumValues + NumValues] / 2
            )
    return Result


def CalcCovarianceFunction(Point1, Point2, Result, Params):
    _verify_dict_keys(Params, ["HardData", "SoftData"], "Params")
    Values = Params["HardData"]
    SoftData = Params["SoftData"]
    NumValues = len(Values)

    # Validate that input data does not contain NaN/Inf values.
    for iv, arr in enumerate(Values):
        if not numpy.all(numpy.isfinite(arr)):
            raise ValueError(
                f"CalcCovarianceFunction: HardData[{iv}] contains NaN or Inf values"
            )
    for iv, arr in enumerate(SoftData):
        if not numpy.all(numpy.isfinite(arr)):
            raise ValueError(
                f"CalcCovarianceFunction: SoftData[{iv}] contains NaN or Inf values"
            )

    if Result is None:
        Result = zeros(NumValues + NumValues + 1, dtype=float64)
    else:
        # Normalize Point1/Point2 to handle both scalar (PointSetScanGridStyle),
        # list (PointSetScanContStyle), and tuple-of-arrays (CubeScan) inputs.
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays.
            # Convert multi-dimensional indices to flat indices for the
            # scalar-index loop below.
            NumPoints = len(Point1[0])
            P1 = ravel_multi_index(Point1, Values[0].shape)
            P2 = ravel_multi_index(Point2, Values[0].shape)
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)

        # Accumulate covariances across all point pairs in the batch
        for idx in range(NumPoints):
            p1 = P1[idx]
            p2 = P2[idx]
            Values1 = zeros(NumValues)
            Values2 = zeros(NumValues)
            SoftValues1 = zeros(NumValues)
            SoftValues2 = zeros(NumValues)
            for i in range(NumValues):
                Values1[i] = Values[i][p1]
                Values2[i] = Values[i][p2]
                SoftValues1[i] = SoftData[i][p1]
                SoftValues2[i] = SoftData[i][p2]
            Covariances = float32((Values1 - SoftValues1) * (Values2 - SoftValues2))
            Result[NumValues + 0 : NumValues + NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] + Covariances
            )
            Result[NumValues + NumValues] += 1

        # Normalize after all pairs accumulated; guard against empty lags
        if Result[NumValues + NumValues] > 0:
            Result[0:NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] / Result[NumValues + NumValues]
            )
    return Result


def CalcIndCorrelationFunction(Point1, Point2, Result, Params):
    _verify_dict_keys(Params, ["HardData", "SoftData"], "Params")
    Values = Params["HardData"]
    SoftData = Params["SoftData"]
    NumValues = len(Values)

    # Validate that input data does not contain NaN/Inf values.
    for iv, arr in enumerate(Values):
        if not numpy.all(numpy.isfinite(arr)):
            raise ValueError(
                f"CalcIndCorrelationFunction: HardData[{iv}] contains NaN or Inf values"
            )
    for iv, arr in enumerate(SoftData):
        if not numpy.all(numpy.isfinite(arr)):
            raise ValueError(
                f"CalcIndCorrelationFunction: SoftData[{iv}] contains NaN or Inf values"
            )

    if Result is None:
        Result = zeros(NumValues + NumValues + 1, dtype=float64)
    else:
        # Normalize Point1/Point2 to handle both scalar (PointSetScanGridStyle),
        # list (PointSetScanContStyle), and tuple-of-arrays (CubeScan) inputs.
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays.
            # Convert multi-dimensional indices to flat indices for the
            # scalar-index loop below.
            NumPoints = len(Point1[0])
            P1 = ravel_multi_index(Point1, Values[0].shape)
            P2 = ravel_multi_index(Point2, Values[0].shape)
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)

        # Accumulate indicator correlations across all point pairs in the batch
        for idx in range(NumPoints):
            p1 = P1[idx]
            p2 = P2[idx]
            Values1 = zeros(NumValues)
            Values2 = zeros(NumValues)
            SoftValues1 = zeros(NumValues)
            SoftValues2 = zeros(NumValues)
            for i in range(NumValues):
                Values1[i] = Values[i][p1]
                Values2[i] = Values[i][p2]
                SoftValues1[i] = SoftData[i][p1]
                SoftValues2[i] = SoftData[i][p2]

            # Guard negative/NaN before sqrt to prevent silent NaN propagation.
            # NaN <= 0 is False, so denom[denom <= 0] alone misses NaN.
            product = SoftValues1 * (1 - SoftValues1) * SoftValues2 * (1 - SoftValues2)
            invalid = (product <= 0) | numpy.isnan(product)
            product[invalid] = 1.0
            denom = product**0.5

            Covariances = float32((Values1 - SoftValues1) * (Values2 - SoftValues2) / denom)
            Result[NumValues + 0 : NumValues + NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] + Covariances
            )
            Result[NumValues + NumValues] += 1

        # Normalize after all pairs accumulated; guard against empty lags
        if Result[NumValues + NumValues] > 0:
            Result[0:NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] / Result[NumValues + NumValues]
            )
    return Result
