# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import math
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

from .validation import GridValidator

MAX_NUM_LAGS = 10000
MAX_POINT_SET_SIZE = 1_000_000
MAX_WINDOW_VOLUME = 100_000_000
# F-03: total-work cap for the pure-Python CubeScan grid path. The C++
# grid kernel rejects window_volume × grid_volume > 1e12
# (variograms.cpp:592); the Python CubeScan previously had only the
# MAX_WINDOW_VOLUME offset cap, so a legal 100^3 grid with a large
# search template ran ~8e12 numpy ops (~2 h) and a 1000^3 grid
# allocated a 24 GB mgrid before any cap fired. The real work is
# len(LagIndexes) × NI·NJ·NK (one sliced-intersection pass per lag
# offset over the full grid), so the cap bounds that product. The
# value is Python-scaled (deliberately lower than the C++ 1e12 for
# pure-Python throughput, matching MAX_TOTAL_PAIR_LAG_WORK).
MAX_TOTAL_GRID_WORK = 1e8
# H-1/R-1: total-work cap for the pure-Python point-set scans. The pair loop
# is O(n^2) and each in-tunnel pair bins into up to NumLags lags, so the
# worst-case work is n^2 * NumLags. MAX_POINT_SET_SIZE (1e6) combined with
# MAX_NUM_LAGS (1e4) would permit up to 1e16 pair-lag operations — an
# effectively-infinite loop in pure Python. This cap rejects the product
# before the loop starts.
#
# DELIBERATE Python/C++ DIVERGENCE: the C++ kernel keeps
# MAX_TOTAL_PAIR_LAG_WORK = 1e12 (variograms.cpp:40 — ~17 min worst case at
# compiled speed, acceptable for the compiled batch kernel). Pure Python runs
# ~4e7 pair-lag ops/s, so the same 1e12 constant would allow ~6 h to 108 days
# of wall-clock work. The Python cap is therefore 1e8 (~2.5 s worst case at
# measured throughput). Parity of the constant was the original rationale;
# parity of the GUARANTEE (no effectively-infinite pure-Python run) is the
# corrected rationale.
MAX_TOTAL_PAIR_LAG_WORK = 1e8


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
        # E-M8: reject R <= 0, not just R < 0, to match the C++ kernel which
        # rejects non-positive ranges (variograms.cpp is_in_tunnel accepts no
        # pairs for a zero range). The previous four Python paths diverged on
        # R=0: accept here / ContStyle warn + silent all-zero / GridStyle +
        # CubeScan raise — one consistent loud rejection at construction
        # (the single chokepoint every scan path shares via the template)
        # replaces the divergent behaviors. (The _IsInTunnel zero-range guard
        # remains as defense for direct attribute mutation, which bypasses
        # this constructor.)
        if not math.isfinite(R1) or R1 <= 0 or not math.isfinite(R2) or R2 <= 0 or not math.isfinite(R3) or R3 <= 0:
            raise ValueError(
                f"TVEllipsoid: ranges must be finite and positive, "
                f"got R1={R1!r}, R2={R2!r}, R3={R3!r}"
            )
        if not math.isfinite(Azimut) or not math.isfinite(Dip) or not math.isfinite(Rotation):
            raise ValueError(
                f"TVEllipsoid: angles must be finite, got "
                f"Azimut={Azimut!r}, Dip={Dip!r}, Rotation={Rotation!r}"
            )
        if not 0 <= Azimut <= 360:
            raise ValueError(
                f"TVEllipsoid: Azimut must be in [0, 360], got {Azimut!r}"
            )
        if not -90 <= Dip <= 90:
            raise ValueError(
                f"TVEllipsoid: Dip must be in [-90, 90], got {Dip!r}"
            )
        if not -90 <= Rotation <= 90:
            raise ValueError(
                f"TVEllipsoid: Rotation must be in [-90, 90], got {Rotation!r}"
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
        if not math.isfinite(NumLags) or NumLags <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: NumLags must be positive, got {NumLags}"
            )
        if NumLags > MAX_NUM_LAGS:
            raise ValueError(
                f"TVVariogramSearchTemplate: NumLags {NumLags} exceeds maximum {MAX_NUM_LAGS}"
            )
        if not math.isfinite(LagWidth) or LagWidth <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: LagWidth must be positive, got {LagWidth}"
            )
        if not math.isfinite(LagSeparation) or LagSeparation <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: LagSeparation must be positive, got {LagSeparation}"
            )
        if not math.isfinite(TolDistance) or TolDistance <= 0:
            raise ValueError(
                f"TVVariogramSearchTemplate: TolDistance must be positive, got {TolDistance}"
            )
        if not math.isfinite(FirstLagDistance) or FirstLagDistance < 0:
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

    WindowVolume = (MaxI - MinI + 1) * (MaxJ - MinJ + 1) * (MaxK - MinK + 1)
    if WindowVolume > MAX_WINDOW_VOLUME:
        raise ValueError(
            f"_CalcLagsAreas: search window volume {WindowVolume} exceeds "
            f"maximum {MAX_WINDOW_VOLUME}. Reduce Ellipsoid ranges or NumLags."
        )

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

    # 2-M-13: lag binning uses the DIRECTIONAL PROJECTION of the offset onto
    # the principal anisotropy axis (|dot(offset, direction1)|), matching the
    # C++ kernels (variograms.cpp:647 grid path, :805 point-set path). The
    # previous raw Euclidean distance produced different lag assignments than
    # the C++ engine for any pair whose offset is not parallel to the
    # principal axis — the same template + data yielded different variogram
    # curves between the Python and C++ paths. Projection is the
    # GSLIB-conventional metric.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    Dist = numpy.abs(GI * D1[0] + GJ * D1[1] + GK * D1[2])

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

    # Validate coordinate arrays for NaN/Inf (F-046).
    # Corrupt coordinates produce wrong distances and garbled variogram lags.
    GridValidator.validate_coordinate_arrays(PX, PY, PZ, "PointSet")

    if len(PX) > MAX_POINT_SET_SIZE:
        raise ValueError(
            f"PointSetScanContStyle: point set size {len(PX)} exceeds "
            f"MAX_POINT_SET_SIZE ({MAX_POINT_SET_SIZE})"
        )

    # H-1/R-1: total-work cap (Python-specific 1e8 — see module constant;
    # deliberately lower than the C++ kernel's 1e12 for pure-Python
    # throughput). The scan is O(n^2) — every point pairs
    # with every other point — and each in-tunnel pair bins into up to
    # NumLags lags, so the worst-case work is n^2 * NumLags. The size cap
    # above bounds the input SIZE only; a legal 1e6-point set with a large
    # search window would run up to 1e16 pair-lag Python iterations — an
    # effectively-infinite loop. Reject the product before the loop starts.
    pair_lag_work = len(PX) * len(PX) * VariogramSearchTemplate.NumLags
    if pair_lag_work > MAX_TOTAL_PAIR_LAG_WORK:
        raise ValueError(
            f"PointSetScanContStyle: estimated pair-lag work {pair_lag_work} "
            f"exceeds maximum {MAX_TOTAL_PAIR_LAG_WORK}. Reduce the point set "
            f"size or NumLags."
        )

    MinX, MinY, MinZ, MaxX, MaxY, MaxZ = _CalcSearchTemplateWindow(VariogramSearchTemplate)

    LagIndex, LagDistance, LagStart, LagEnd = _CalcLagDistances(VariogramSearchTemplate)
    # 2-M-13: the lag-binning metric is the directional projection onto the
    # principal anisotropy axis (matching variograms.cpp:647, 805), so the
    # pre-pruning band must use the same projection — not the raw Euclidean
    # distance, which can exceed the projection and would wrongly discard
    # pairs that belong to a lag band under the projection metric.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    # E2-30: the D2/D3 axes are needed for the hemisphere tie-break below
    # (perpendicular-pair canonical cut); fetched once per scan.
    D2 = VariogramSearchTemplate.Ellipsoid.Direction2
    D3 = VariogramSearchTemplate.Ellipsoid.Direction3
    MinDistance = max(0, LagStart.min())
    MaxDistance = LagEnd.max()

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

        # 2-M-13: directional projection onto the principal anisotropy axis
        # for lag binning (|dot(offset, direction1)|) — matches the C++
        # point-set kernel (variograms.cpp:805). Computed BEFORE the band
        # filter so the pruning and the binning use the same metric.
        FDistance = numpy.abs(FDX * D1[0] + FDY * D1[1] + FDZ * D1[2])
        Filter = MinDistance <= FDistance
        Filter = bitwise_and(Filter, FDistance <= MaxDistance)

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance = FDistance[Filter]

        # F-M24: skip the self-pair (j == i) to match the C++ point-set scan
        # (variograms.cpp:793 `if (idx1 == idx2) continue;`). With
        # FirstLagDistance=0 the first lag band starts at distance 0, so each
        # point's self-pair passes the distance filter and lands in lag 0,
        # adding a zero-variance pair and +1 count that dilutes the lag-0
        # variogram the C++ never counts. Unconditional is exact: when
        # FirstLagDistance > 0 the self-pair is already excluded by the
        # MinDistance distance filter (distance 0 < MinDistance), so this
        # filter is a no-op there.
        SelfPairFilter = FIndex != i
        FDX, FDY, FDZ = FDX[SelfPairFilter], FDY[SelfPairFilter], FDZ[SelfPairFilter]
        FIndex = FIndex[SelfPairFilter]
        FDistance = FDistance[SelfPairFilter]

        # E2-30: hemisphere-restrict the window in the rotated frame. The
        # axis-aligned search window (the AABB of the rotated +Direction1
        # corners) admits BOTH +v and -v offsets for pairs whose offset is
        # (near-)perpendicular to Direction1 — both endpoints' offsets fall
        # in the box's central overlap region — so those pairs were counted
        # twice (once from each endpoint) while every other pair counted
        # once: non-uniform pair weighting under rotated ellipsoids (6/89
        # pairs double-counted at azimuth=45°; uniform at 0°). The C++
        # point-set scan has no window and counts every ordered pair
        # uniformly (variograms.cpp:886-915), so uniform weighting is the
        # parity target. The hemisphere cut admits only the +Direction1
        # endpoint — each unordered pair is counted exactly once — with a
        # canonical (dot2, dot3) tie-break for the exact dot == 0 boundary
        # so perpendicular pairs remain counted (once, not dropped).
        # R-08: a zero-distance pair between DISTINCT coincident points
        # (identical coordinates) has Signed==Signed2==Signed3==0, so the
        # tie-break admits NEITHER endpoint and the pair is dropped
        # entirely (counted 0x), while the C++ kernel counts it
        # (variograms.cpp:913 skips only idx1==idx2). The SelfPairFilter
        # above already removed the true self-pair, so every remaining zero
        # offset is a distinct coincident point; admit it from exactly one
        # endpoint (canonical index tie-break FIndex > i) so the pair
        # counts once, preserving the once-per-unordered-pair invariant.
        # When FirstLagDistance > 0 the MinDistance band filter above has
        # already excluded zero offsets, so this admission is inert there.
        Signed = FDX * D1[0] + FDY * D1[1] + FDZ * D1[2]
        Signed2 = FDX * D2[0] + FDY * D2[1] + FDZ * D2[2]
        Signed3 = FDX * D3[0] + FDY * D3[1] + FDZ * D3[2]
        ZeroOffset = (FDX == 0) & (FDY == 0) & (FDZ == 0)
        Hemisphere = (ZeroOffset & (FIndex > i)) | (Signed > 0) | (
            (Signed == 0) & ((Signed2 > 0) | ((Signed2 == 0) & (Signed3 > 0)))
        )
        FDX, FDY, FDZ = FDX[Hemisphere], FDY[Hemisphere], FDZ[Hemisphere]
        FIndex = FIndex[Hemisphere]
        FDistance = FDistance[Hemisphere]

        Filter = _IsInTunnel(VariogramSearchTemplate, column_stack((FDX, FDY, FDZ)))

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance = FDistance[Filter]

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
    if len(LagIndexes) == 0:
        raise ValueError(
            "PointSetScanGridStyle: _CalcLagsAreas returned empty LagIndexes — "
            "no lag offsets found. Check search template parameters "
            "(NumLags, LagSeparation, LagWidth, Ellipsoid ranges)."
        )
    IMin, IMax = LI.min(), LI.max()
    JMin, JMax = LJ.min(), LJ.max()
    KMin, KMax = LK.min(), LK.max()

    PI = PointSetXYZ[0]
    PJ = PointSetXYZ[1]
    PK = PointSetXYZ[2]

    # III-15: continuous (tolerance-based) lag binning requires the lag
    # band boundaries (LagStart/LagEnd) and the principal anisotropy
    # direction, exactly as PointSetScanContStyle and the C++ point-set
    # kernel (variograms.cpp:905-912) use them. The exact-integer offset
    # matching below was replaced by continuous band binning so fractional
    # grid spacing (0.5/0.25 m) is binned like integer spacing.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    # E2-30: the D2/D3 axes are needed for the hemisphere tie-break below
    # (perpendicular-pair canonical cut); fetched once per scan.
    D2 = VariogramSearchTemplate.Ellipsoid.Direction2
    D3 = VariogramSearchTemplate.Ellipsoid.Direction3
    _, LagDistance, LagStart, LagEnd = _CalcLagDistances(VariogramSearchTemplate)

    # Validate coordinate arrays for NaN/Inf (F-046)
    GridValidator.validate_coordinate_arrays(PI, PJ, PK, "PointSetXYZ")

    if len(PI) > MAX_POINT_SET_SIZE:
        raise ValueError(
            f"PointSetScanGridStyle: point set size {len(PI)} exceeds "
            f"MAX_POINT_SET_SIZE ({MAX_POINT_SET_SIZE})"
        )

    # H-1/R-1/F-27: total-work cap (Python-specific 1e8 — see module
    # constant; deliberately lower than the C++ kernel's 1e12 for
    # pure-Python throughput) shared with PointSetScanContStyle above.
    # The scan is O(n^2) and each candidate pair is binned by the
    # directional projection into up to NumLags lag bands (continuous
    # binning, matching the C++ point-set kernel variograms.cpp:905-912).
    # The old exact-integer offset matching performed three full-array
    # comparisons (FDI[j] == LI, FDJ[j] == LJ, FDK[j] == LK), each
    # O(len(LI)) where len(LI) is the number of lag-area offsets (up to
    # the search window volume, 1e8) — the n^2 × NumLags estimate
    # under-stated the real work by up to len(LI)/NumLags (~1e7×), so a
    # legal input could run ~1e14 pure-Python comparisons. III-15
    # replaced that exact matching with continuous band binning whose
    # per-pair cost IS NumLags, so n^2 × NumLags now bounds the real
    # work (same formula as the C++ point-set cap, variograms.cpp:833).
    pair_lag_work = len(PI) * len(PI) * VariogramSearchTemplate.NumLags
    if pair_lag_work > MAX_TOTAL_PAIR_LAG_WORK:
        raise ValueError(
            f"PointSetScanGridStyle: estimated pair-lag work {pair_lag_work} "
            f"exceeds maximum {MAX_TOTAL_PAIR_LAG_WORK}. Reduce the point set "
            f"size or NumLags."
        )

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

        FDI, FDJ, FDK = DI[Filter], DJ[Filter], DK[Filter]
        FIndex = Index[Filter]

        # E2-30: hemisphere-restrict the window in the rotated frame — same
        # defect and fix as PointSetScanContStyle: under rotation the
        # axis-aligned window box admits BOTH +v and -v offsets for
        # (near-)perpendicular pairs, double-counting them, while the C++
        # point-set scan (no window) counts every ordered pair uniformly
        # (variograms.cpp:886-915). Admitting only the +Direction1 endpoint
        # (canonical (dot2, dot3) tie-break at the exact dot == 0 boundary)
        # makes every unordered pair count exactly once.
        # R-08: zero-distance pairs between DISTINCT coincident points
        # (identical coordinates) have all-zero dots, so the tie-break
        # admits neither endpoint and the pair is dropped (counted 0x),
        # while the C++ kernel counts it (variograms.cpp:913 skips only
        # idx1==idx2). The (0,0,0) offset is in the lag-area window when
        # the first lag band includes distance 0, so admit zero offsets
        # from exactly one endpoint (canonical index tie-break FIndex > i):
        # the true self-pair (FIndex == i) stays excluded and each distinct
        # coincident pair counts once. Inert when the lag bands exclude
        # distance 0.
        Signed = FDI * D1[0] + FDJ * D1[1] + FDK * D1[2]
        Signed2 = FDI * D2[0] + FDJ * D2[1] + FDK * D2[2]
        Signed3 = FDI * D3[0] + FDJ * D3[1] + FDK * D3[2]
        ZeroOffset = (FDI == 0) & (FDJ == 0) & (FDK == 0)
        Hemisphere = (ZeroOffset & (FIndex > i)) | (Signed > 0) | (
            (Signed == 0) & ((Signed2 > 0) | ((Signed2 == 0) & (Signed3 > 0)))
        )
        FDI, FDJ, FDK = FDI[Hemisphere], FDJ[Hemisphere], FDK[Hemisphere]
        FIndex = FIndex[Hemisphere]

        # III-15: continuous (tolerance-based) lag binning by the
        # directional projection onto the principal anisotropy axis,
        # matching the C++ point-set kernel (variograms.cpp:905-912) and
        # PointSetScanContStyle. The old exact-integer offset matching
        # (FDI[j] == LI, FDJ[j] == LJ, FDK[j] == LK) dropped EVERY pair on
        # fractional grid spacing (0.5/0.25 m) because no integer lag-area
        # offset equals a fractional displacement — the variogram silently
        # came out empty/under-counted (live repro: 6 points at 0..2.5 →
        # 0 of 30 ordered pairs counted). Continuous band binning assigns
        # each pair to the lag whose [LagStart, LagEnd) band contains its
        # projection, exactly like the C++/ContStyle paths. The per-candidate
        # tunnel filter keeps parity with _CalcLagsAreas (the old exact
        # matching inherited the tunnel implicitly from LI/LJ/LK).
        Tunnel = _IsInTunnel(VariogramSearchTemplate, column_stack((FDI, FDJ, FDK)))
        FDI, FDJ, FDK = FDI[Tunnel], FDJ[Tunnel], FDK[Tunnel]
        FIndex = FIndex[Tunnel]
        FDistance = numpy.abs(FDI * D1[0] + FDJ * D1[1] + FDK * D1[2])
        for j in range(len(FDI)):
            # M-21: skip the self-pair (j == i), mirroring the C++ point-set
            # kernel (variograms.cpp:793) and PointSetScanContStyle's F-M24
            # skip. Without it, GridStyle counts each point's self-pair
            # (offset (0,0,0)) in lag 0, diluting the lag-0 variogram exactly
            # as the pre-F-M24 ContStyle did — reproducing the dilution the
            # ContStyle fix removed. With FirstLagDistance > 0 the self-pair
            # offset lands in no lag band, so the skip is a no-op there.
            if FIndex[j] == i:
                continue
            for Lag in range(VariogramSearchTemplate.NumLags):
                if LagStart[Lag] <= FDistance[j] < LagEnd[Lag]:
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
    # Normalize integer (e.g. uint8) masks to boolean: "non-zero indicates an
    # informed cell". Without this, Mask1 & Mask2 stays uint8 and is used as an
    # integer fancy index in the slicing below, inflating pair counts or
    # raising IndexError (F-01).
    Mask = Mask != 0
    NI, NJ, NK = Mask.shape

    LI, LJ, LK, LagIndexes, LagDistance = _CalcLagsAreas(VariogramSearchTemplate)

    if len(LagIndexes) == 0:
        raise ValueError(
            "CubeScan: _CalcLagsAreas returned empty LagIndexes — "
            "no lag offsets found. Check search template parameters "
            "(NumLags, LagSeparation, LagWidth, Ellipsoid ranges)."
        )

    # F-03: total-work cap. CubeScan performs one sliced-intersection pass
    # over the whole grid per lag offset, so the real work is
    # len(LagIndexes) × NI·NJ·NK. The C++ grid kernel rejects the same
    # product above 1e12 (variograms.cpp:592); the Python cap is scaled
    # down for pure-Python throughput. Pre-fix a legal 100^3 grid with a
    # large template ran ~8e12 numpy ops (~2 h) and a 1000^3 grid
    # allocated a 24 GB mgrid before any check fired.
    total_grid_work = len(LagIndexes) * NI * NJ * NK
    if total_grid_work > MAX_TOTAL_GRID_WORK:
        raise ValueError(
            f"CubeScan: estimated total grid work {total_grid_work} "
            f"(grid {NI}x{NJ}x{NK}, {len(LagIndexes)} lag offsets) exceeds "
            f"maximum {MAX_TOTAL_GRID_WORK}. Reduce the grid size, NumLags, "
            f"or Ellipsoid ranges."
        )

    # F-N14: guard the search-template extent against the grid size before
    # slicing. The C++ grid path bounds-checks every candidate pair with
    # is_inside (variograms.cpp:603) and silently skips out-of-bound pairs;
    # the Python slice approach below cannot skip a single offset, and an
    # offset with magnitude > the grid dimension produces an empty Mask1
    # slice and a truncated Mask2 slice (e.g. min(NI - DI, NI) == -1 for
    # DI > NI → Mask2[0:-1]) → broadcast ValueError. Reject the degenerate
    # template up front. (Offsets of magnitude exactly equal to the grid
    # dimension produce empty-on-both-sides slices — no crash, and the
    # zero-pair outcome matches the C++ skip.)
    if (
        max(abs(LI)) > NI
        or max(abs(LJ)) > NJ
        or max(abs(LK)) > NK
    ):
        raise ValueError(
            "CubeScan: search-template lag offset exceeds grid size "
            f"(grid {NI}x{NJ}x{NK}, max offset |i|={max(abs(LI))}, "
            f"|j|={max(abs(LJ))}, |k|={max(abs(LK))}). Reduce Ellipsoid "
            "ranges, NumLags, or LagSeparation so the search extent fits "
            "within the grid."
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
        # F-M14: propagate the F-02 fix (ravel_multi_index + numpy.take) from
        # the covariance/correlation siblings. The raw multi-axis indexing
        # `Values[i][Point1[:]]` only worked when the tuple length matched a
        # 3D value array (IndexError for 1D flat values) and a scalar index on
        # 3D values raised a broadcast ValueError; numpy.take on the flat
        # indices handles every combination the siblings handle.
        P1: numpy.ndarray
        P2: numpy.ndarray
        NumPoints: int
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays.
            # ravel_multi_index flattens the multi-index components to flat
            # indices matching the C-order ravel of the value arrays (F-02).
            # The tuple must have one component per value-array dimension
            # (3 components for 3D grid values, 1 for flat 1D values) —
            # ravel_multi_index requires len(multi_index) == len(shape).
            NumPoints = len(Point1[0])
            P1 = ravel_multi_index(Point1, Values[0].shape)  # type: ignore[assignment]
            P2 = ravel_multi_index(Point2, Values[0].shape)  # type: ignore[assignment]
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)

        # numpy.take with axis=None (default) indexes the raveled array, so
        # flat indices resolve correctly regardless of Values[i].ndim.
        Values1 = zeros((NumValues, NumPoints))
        Values2 = zeros((NumValues, NumPoints))
        for i in range(NumValues):
            Values1[i] = numpy.take(Values[i], P1)
            Values2[i] = numpy.take(Values[i], P2)
        # E-M10: accumulate in float64, matching the C++ kernel which sums in
        # double (variograms.cpp:706-708). The previous float32 cast of the
        # per-pair differences BEFORE squaring/summing silently diverged on
        # cross-magnitude data (e.g. permeability spans 6-8 orders). Result
        # slots are float64, so no downcast is needed — the sum lands in the
        # accumulator at full precision.
        Variances = (Values1 - Values2) ** 2
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
        P1: numpy.ndarray
        P2: numpy.ndarray
        NumPoints: int
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays.
            # ravel_multi_index flattens the multi-index components to flat
            # indices matching the C-order ravel of the value arrays (F-02).
            # The tuple must have one component per value-array dimension
            # (3 components for 3D grid values, 1 for flat 1D values) —
            # ravel_multi_index requires len(multi_index) == len(shape);
            # numpy.take with axis=None (default) then indexes the raveled
            # array regardless of Values[i].ndim.
            NumPoints = len(Point1[0])
            P1 = ravel_multi_index(Point1, Values[0].shape)  # type: ignore[assignment]
            P2 = ravel_multi_index(Point2, Values[0].shape)  # type: ignore[assignment]
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)

        # Accumulate covariances across all point pairs in the batch
        # (vectorized, mirroring CalcVariogramFunction).
        Values1 = zeros((NumValues, NumPoints))
        Values2 = zeros((NumValues, NumPoints))
        SoftValues1 = zeros((NumValues, NumPoints))
        SoftValues2 = zeros((NumValues, NumPoints))
        for i in range(NumValues):
            Values1[i] = numpy.take(Values[i], P1)
            Values2[i] = numpy.take(Values[i], P2)
            SoftValues1[i] = numpy.take(SoftData[i], P1)
            SoftValues2[i] = numpy.take(SoftData[i], P2)
        # E-M10: accumulate in float64 (the C++ kernel sums in double,
        # variograms.cpp:706-708). The float32 cast before the product+sum
        # silently diverged on cross-magnitude data.
        Covariances = (Values1 - SoftValues1) * (Values2 - SoftValues2)
        Result[NumValues + 0 : NumValues + NumValues] = (
            Result[NumValues + 0 : NumValues + NumValues] + Covariances.sum(axis=1)
        )
        Result[NumValues + NumValues] += NumPoints

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
        P1: numpy.ndarray
        P2: numpy.ndarray
        NumPoints: int
        if isinstance(Point1, tuple):
            # CubeScan path: Point1 is (I, J, K) tuple of 1D index arrays.
            # ravel_multi_index flattens the multi-index components to flat
            # indices matching the C-order ravel of the value arrays (F-02).
            # The tuple must have one component per value-array dimension
            # (3 components for 3D grid values, 1 for flat 1D values) —
            # ravel_multi_index requires len(multi_index) == len(shape);
            # numpy.take with axis=None (default) then indexes the raveled
            # array regardless of Values[i].ndim.
            NumPoints = len(Point1[0])
            P1 = ravel_multi_index(Point1, Values[0].shape)  # type: ignore[assignment]
            P2 = ravel_multi_index(Point2, Values[0].shape)  # type: ignore[assignment]
        else:
            P1 = numpy.atleast_1d(numpy.asarray(Point1)).ravel()
            P2 = numpy.atleast_1d(numpy.asarray(Point2)).ravel()
            NumPoints = len(P1)

        # Accumulate indicator correlations across all point pairs in the batch
        # (vectorized, mirroring CalcVariogramFunction).
        Values1 = zeros((NumValues, NumPoints))
        Values2 = zeros((NumValues, NumPoints))
        SoftValues1 = zeros((NumValues, NumPoints))
        SoftValues2 = zeros((NumValues, NumPoints))
        for i in range(NumValues):
            Values1[i] = numpy.take(Values[i], P1)
            Values2[i] = numpy.take(Values[i], P2)
            SoftValues1[i] = numpy.take(SoftData[i], P1)
            SoftValues2[i] = numpy.take(SoftData[i], P2)

        # II-34: exclude pairs whose soft-prob variance product is
        # non-positive (soft prob 0 or 1) instead of substituting
        # denom=1.0. The old substitution let the unnormalized raw
        # covariance term dominate the sum, silently diluting/inflating
        # the "correlation" outside [-1,1] whenever a soft probability is
        # exactly 0 or 1 (realistic for indicator data). The standard
        # guard (and the C++ kernel, variograms.cpp calc_ind_correlation
        # family) excludes the pair entirely. The pair-count slot is
        # incremented only for retained pairs so the average is computed
        # over the valid subset.
        product = SoftValues1 * (1 - SoftValues1) * SoftValues2 * (1 - SoftValues2)
        # NaN <= 0 is False, so an explicit isnan check is required in
        # addition to the <= 0 test (the input isfinite gate above rules
        # NaN out of Values/SoftData, but keep the guard defensive).
        valid = (product > 0) & ~numpy.isnan(product)

        denom = numpy.zeros_like(product)
        denom[valid] = product[valid] ** 0.5

        # Excluded pairs contribute 0 covariance (and are not counted in
        # the pair-count slot below). Divide only where valid to avoid
        # 0/0 NaN intermediate values (numpy.where evaluates both branches,
        # so masking after division would still warn on invalid entries).
        with numpy.errstate(divide="ignore", invalid="ignore"):
            raw = (Values1 - SoftValues1) * (Values2 - SoftValues2) / denom
        # E-M10: keep the float64 accumulation (the C++ kernel sums in
        # double, variograms.cpp:706-708). The .astype(float32) downcast
        # before the sum silently diverged on cross-magnitude indicator data.
        Covariances = numpy.where(valid, raw, 0.0)
        Result[NumValues + 0 : NumValues + NumValues] = (
            Result[NumValues + 0 : NumValues + NumValues] + Covariances.sum(axis=1)
        )
        Result[NumValues + NumValues] += int(valid.sum())

        # Normalize after all pairs accumulated; guard against empty lags
        if Result[NumValues + NumValues] > 0:
            Result[0:NumValues] = (
                Result[NumValues + 0 : NumValues + NumValues] / Result[NumValues + NumValues]
            )
    return Result
