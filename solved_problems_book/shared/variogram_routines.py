import numpy as np

class TVEllipsoid:
    Direction1 = [1, 0, 0]
    Direction2 = [0, 1, 0]
    Direction3 = [0, 0, 1]
    R1 = 1
    R2 = 1
    R3 = 1
    def __init__(self, R1, R2, R3, Azimut=0, Dip=0, Rotation=0):
        Azimut = np.radians(Azimut)
        Dip = np.radians(Dip)
        Rotation = np.radians(Rotation)

        A = np.array([
             [np.cos(Azimut), -np.sin(Azimut), 0],
             [np.sin(Azimut),  np.cos(Azimut), 0],
             [0, 0, 1]
             ])

        B = np.array([
             [np.cos(Dip), 0, -np.sin(Dip)],
             [0, 1, 0],
             [np.sin(Dip), 0,  np.cos(Dip)]
             ])

        C = np.array([
             [1, 0, 0],
             [0, np.cos(Rotation), -np.sin(Rotation)],
             [0, np.sin(Rotation),  np.cos(Rotation)]
             ])

        ABC = A @ B @ C

        self.Direction1 = ABC[:, 0]
        self.Direction2 = ABC[:, 1]
        self.Direction3 = ABC[:, 2]

        self.R1 = R1
        self.R2 = R2
        self.R3 = R3

class TVVariogramSearchTemplate:
    LagWith = 0.5
    LagSeparation = 1
    TolDistance = 1
    NumLags = 10
    FirstLag = 0
    Ellipsoid = TVEllipsoid(1, 1, 1)
    def __init__(self, LagWidth, LagSeparation, TolDistance, NumLags, Ellipsoid, FirstLagDistance=0):
        self.LagWidth = LagWidth
        self.LagSeparation = LagSeparation
        self.TolDistance = TolDistance
        self.NumLags = NumLags
        self.Ellipsoid = Ellipsoid
        self.FirstLagDistance = FirstLagDistance

def _IsInTunnel(VariogramSearchTemplate, V):
    # E-H8: projections are taken in absolute value and normalized by the
    # corresponding ellipsoid range, exactly like the core twin
    # (src/geo_bsd/variogram.py:236-259) and the C++ kernel
    # (variograms.cpp:282-307). The previous version omitted fabs and the
    # S1 = SS1/R1 normalization: with R1 != 1 the directional gate used the
    # raw unscaled projection, widening the tunnel half-angle (e.g.
    # gammabar.py R1=10, TolDistance=4: gate 68.2 deg vs reference 14.0 deg).
    SS1 = np.abs(V @ VariogramSearchTemplate.Ellipsoid.Direction1)
    SS2 = np.abs(V @ VariogramSearchTemplate.Ellipsoid.Direction2)
    SS3 = np.abs(V @ VariogramSearchTemplate.Ellipsoid.Direction3)

    S1 = SS1 / VariogramSearchTemplate.Ellipsoid.R1
    S2 = SS2 / VariogramSearchTemplate.Ellipsoid.R2
    S3 = SS3 / VariogramSearchTemplate.Ellipsoid.R3

    Dist = np.power(np.power(S2, 2) + np.power(S3, 2), 0.5)
    Result = np.array(np.bitwise_and(Dist <= 1, VariogramSearchTemplate.TolDistance * Dist <= S1))

    return np.reshape(Result, len(Result))

def _CalcSearchTemplateWindow(VariogramSearchTemplate):
    Max = 1E10
    MinI = Max
    MaxI = -Max
    MinJ = Max
    MaxJ = -Max
    MinK = Max
    MaxK = -Max
    for i in range(0, 2):
        for j in range(-1, 2, 2):
            for k in range(-1, 2, 2):
                DI = VariogramSearchTemplate.Ellipsoid.Direction1 * VariogramSearchTemplate.LagSeparation * VariogramSearchTemplate.NumLags * i
                DJ = VariogramSearchTemplate.Ellipsoid.Direction2 * VariogramSearchTemplate.Ellipsoid.R2 * j
                DK = VariogramSearchTemplate.Ellipsoid.Direction3 * VariogramSearchTemplate.Ellipsoid.R3 * k
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
    LagDistance = np.array(list(LagIndexes)) * VariogramSearchTemplate.LagSeparation + VariogramSearchTemplate.FirstLagDistance
    LagWidth = VariogramSearchTemplate.LagWidth
    LagStart = LagDistance - LagWidth / 2
    LagEnd = LagDistance + LagWidth / 2

    return LagIndexes, LagDistance, LagStart, LagEnd

def _CalcLagsAreas(VariogramSearchTemplate):
    (MinI, MinJ, MinK, MaxI, MaxJ, MaxK) = _CalcSearchTemplateWindow(VariogramSearchTemplate)
    MinI = int(np.floor(MinI))
    MinJ = int(np.floor(MinJ))
    MinK = int(np.floor(MinK))
    MaxI = int(np.ceil(MaxI))
    MaxJ = int(np.ceil(MaxJ))
    MaxK = int(np.ceil(MaxK))

    I = np.zeros([])
    J = np.zeros([])
    K = np.zeros([])
    LagIndexes = np.zeros([])

    (Index, LagDistance, LagStart, LagEnd) = _CalcLagDistances(VariogramSearchTemplate)

    GI, GJ, GK = np.mgrid[MinI:MaxI+1, MinJ:MaxJ+1, MinK:MaxK+1]

    GI = GI.reshape(np.prod(GI.shape), 1)
    GJ = GJ.reshape(np.prod(GJ.shape), 1)
    GK = GK.reshape(np.prod(GK.shape), 1)

    ActivePoints = _IsInTunnel(VariogramSearchTemplate, np.column_stack((GI, GJ, GK)))

    GI = GI[ActivePoints]
    GJ = GJ[ActivePoints]
    GK = GK[ActivePoints]

    # 2-M-13: lag binning uses the directional projection of the offset onto
    # the principal anisotropy axis (|dot(offset, Direction1)|), matching the
    # core twin (src/geo_bsd/variogram.py:362-363) and the C++ grid kernel.
    # The previous raw Euclidean distance assigned grid offsets to different
    # lag bands whenever the offset was not parallel to the principal axis.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    Dist = np.abs(GI * D1[0] + GJ * D1[1] + GK * D1[2])

    for i in Index:
        Filter = np.bitwise_and(LagStart[i] <= Dist, Dist < LagEnd[i])
        NumPoints = np.sum(Filter)
        I = np.row_stack((I, GI[Filter].reshape(NumPoints, 1)))
        J = np.row_stack((J, GJ[Filter].reshape(NumPoints, 1)))
        K = np.row_stack((K, GK[Filter].reshape(NumPoints, 1)))
        LagIndexes = np.row_stack((LagIndexes, np.ones((NumPoints, 1)) * i))

    return I[1:], J[1:], K[1:], LagIndexes[1:], LagDistance

def PointSetScanContStyle(VariogramSearchTemplate, PointSet, Function, Params):
    PX = PointSet['X']
    PY = PointSet['Y']
    PZ = PointSet['Z']

    MinX, MinY, MinZ, MaxX, MaxY, MaxZ = _CalcSearchTemplateWindow(VariogramSearchTemplate)

    LagIndex, LagDistance, LagStart, LagEnd = _CalcLagDistances(VariogramSearchTemplate)
    # 2-M-13: the lag-binning metric is the directional projection onto the
    # principal anisotropy axis (|dot(offset, Direction1)|), matching the core
    # twin (src/geo_bsd/variogram.py:533-543) and the C++ point-set kernel
    # (variograms.cpp:902). The previous raw Euclidean distance assigned pairs
    # to different lag bands whenever the offset was not parallel to the
    # principal axis (projection 500 + lateral 500 -> Euclidean 707 -> next
    # lag). The pre-pruning band must use the same projection as the binning.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    MinDistance = max(0, min(LagStart))
    MaxDistance = max(LagEnd)

    if Function is not None:
        Result = Function(0, 0, None, Params)
        Result = np.reshape(Result, (1, len(Result)))
        Result = np.repeat(Result, VariogramSearchTemplate.NumLags, 0)

    Index = np.array(range(0, len(PX)))
    for i in range(len(PX)):
        X1, Y1, Z1 = PX[i], PY[i], PZ[i]
        DX, DY, DZ = PX - X1, PY - Y1, PZ - Z1

        Filter = MinX <= DX
        Filter = np.bitwise_and(Filter, DX <= MaxX)
        Filter = np.bitwise_and(Filter, MinY <= DY)
        Filter = np.bitwise_and(Filter, DY <= MaxY)
        Filter = np.bitwise_and(Filter, MinZ <= DZ)
        Filter = np.bitwise_and(Filter, DZ <= MaxZ)

        FDX, FDY, FDZ = DX[Filter], DY[Filter], DZ[Filter]
        FIndex = Index[Filter]

        # 2-M-13: directional projection onto the principal anisotropy axis
        # for lag binning (|dot(offset, Direction1)|) — matches the core twin
        # (variogram.py:537) and the C++ point-set kernel (variograms.cpp:902).
        # Computed BEFORE the band filter so the pruning and the binning use
        # the same metric.
        FDistance = np.abs(FDX * D1[0] + FDY * D1[1] + FDZ * D1[2])
        Filter = np.bitwise_and(MinDistance <= FDistance, FDistance <= MaxDistance)

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance = FDistance[Filter]

        # F-M24: skip the self-pair (j == i), mirroring the core twin
        # (variogram.py:545-557) and the C++ point-set kernel
        # (variograms.cpp:890 `if (idx1 == idx2) continue;`). The previous
        # II-46 filter excluded every zero-distance pair; the index-based
        # skip is exact (a non-self pair with zero projection is a valid
        # lag-0 pair under the projection metric). With FirstLagDistance=0
        # the first lag band starts at 0, so each point's self-pair passes
        # the band filter and lands in lag 0, adding a zero-variance pair
        # and +1 count that dilutes the lag-0 variogram the core never
        # counts. Unconditional is exact: when FirstLagDistance > 0 the
        # self-pair is already excluded by the MinDistance filter.
        SelfPairFilter = FIndex != i
        FDX, FDY, FDZ = FDX[SelfPairFilter], FDY[SelfPairFilter], FDZ[SelfPairFilter]
        FIndex = FIndex[SelfPairFilter]
        FDistance = FDistance[SelfPairFilter]

        Filter = _IsInTunnel(VariogramSearchTemplate, np.column_stack((FDX, FDY, FDZ)))

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance = FDistance[Filter]

        for Lag in LagIndex:
            Filter = np.bitwise_and(LagStart[Lag] <= FDistance, FDistance < LagEnd[Lag])
            for j in FIndex[Filter]:
                Result[Lag, :] = Function(i, j, Result[Lag, :], Params)

    return Result, LagDistance

def PointSetScanGridStyle(VariogramSearchTemplate, PointSetXYZ, Function, Params):
    LI, LJ, LK, _LagIndexes, LagDistance = _CalcLagsAreas(VariogramSearchTemplate)
    # III-15: continuous (tolerance-based) lag binning requires the lag band
    # boundaries (LagStart/LagEnd) and the principal anisotropy direction,
    # exactly as the core twin (variogram.py:629-630) and the C++ point-set
    # kernel (variograms.cpp:905-912) use them.
    D1 = VariogramSearchTemplate.Ellipsoid.Direction1
    _, LagDistance, LagStart, LagEnd = _CalcLagDistances(VariogramSearchTemplate)
    IMin, IMax = min(LI), max(LI)
    JMin, JMax = min(LJ), max(LJ)
    KMin, KMax = min(LK), max(LK)

    PI = PointSetXYZ[0]
    PJ = PointSetXYZ[1]
    PK = PointSetXYZ[2]

    if Function is not None:
        Result = Function(0, 0, None, Params)
        Result = np.reshape(Result, (1, len(Result)))
        Result = np.repeat(Result, VariogramSearchTemplate.NumLags, 0)

    Index = np.array(range(0, len(PI)))
    for i in range(len(PI)):
        I1, J1, K1 = PI[i], PJ[i], PK[i]
        DI, DJ, DK = PI - I1, PJ - J1, PK - K1

        Filter = IMin <= DI
        Filter = np.bitwise_and(Filter, DI <= IMax)
        Filter = np.bitwise_and(Filter, JMin <= DJ)
        Filter = np.bitwise_and(Filter, DJ <= JMax)
        Filter = np.bitwise_and(Filter, KMin <= DK)
        Filter = np.bitwise_and(Filter, DK <= KMax)

        FPI, FPJ, FPK = PI[Filter], PJ[Filter], PK[Filter]
        FDI, FDJ, FDK = FPI - I1, FPJ - J1, FPK - K1
        FIndex = Index[Filter]

        # III-15: continuous (tolerance-based) lag binning by the directional
        # projection onto the principal anisotropy axis, matching the core
        # twin (variogram.py:686-715) and the C++ point-set kernel
        # (variograms.cpp:905-912). The old exact-integer offset matching
        # (FDI[j] == LI, FDJ[j] == LJ, FDK[j] == LK) dropped EVERY pair on
        # fractional grid spacing (0.5/0.25 m) because no integer lag-area
        # offset equals a fractional displacement — the variogram silently
        # came out empty/under-counted. Continuous band binning assigns each
        # pair to the lag whose [LagStart, LagEnd) band contains its
        # projection, exactly like the C++/ContStyle paths. The per-candidate
        # tunnel filter keeps parity with _CalcLagsAreas (the old exact
        # matching inherited the tunnel implicitly from LI/LJ/LK).
        Tunnel = _IsInTunnel(VariogramSearchTemplate, np.column_stack((FDI, FDJ, FDK)))
        FDI, FDJ, FDK = FDI[Tunnel], FDJ[Tunnel], FDK[Tunnel]
        FIndex = FIndex[Tunnel]
        FDistance = np.abs(FDI * D1[0] + FDJ * D1[1] + FDK * D1[2])
        for j in range(len(FDI)):
            # M-21: skip the self-pair (j == i), mirroring the core twin
            # (variogram.py:704-712) and PointSetScanContStyle's F-M24 skip.
            # Without it, GridStyle counts each point's self-pair (offset
            # (0,0,0)) in lag 0, diluting the lag-0 variogram exactly as the
            # pre-F-M24 ContStyle did. With FirstLagDistance > 0 the self-pair
            # offset lands in no lag band, so the skip is a no-op there.
            if FIndex[j] == i:
                continue
            if Function is not None:
                for Lag in range(VariogramSearchTemplate.NumLags):
                    if LagStart[Lag] <= FDistance[j] < LagEnd[Lag]:
                        Result[Lag, :] = Function(i, FIndex[j], Result[Lag, :], Params)

    return Result, LagDistance

def CalcVariogramFunction(Point1, Point2, Result, Params):
    Values = Params['HardData']
    NumValues = len(Values)
    if Result is None:
        Result = np.zeros(NumValues + NumValues + 1, dtype=np.float32)
    else:
        Values1 = np.zeros(NumValues)
        Values2 = np.zeros(NumValues)
        for i in range(NumValues):
            Values1[i] = Values[i][Point1]
            Values2[i] = Values[i][Point2]
        Variances = np.float32(Values1 - Values2)**2
        Result[NumValues + 0:NumValues + NumValues] = Result[NumValues + 0:NumValues + NumValues] + Variances
        Result[NumValues + NumValues] += 1
        Result[0:NumValues] = Result[NumValues + 0:NumValues + NumValues] / Result[NumValues + NumValues] / 2
    return Result

def CalcCovarianceFunction(Point1, Point2, Result, Params):
    Values = Params['HardData']
    SoftData = Params['SoftData']
    NumValues = len(Values)
    if Result is None:
        Result = np.zeros(NumValues + NumValues + 1, dtype=np.float32)
    else:
        Values1 = np.zeros(NumValues)
        Values2 = np.zeros(NumValues)
        SoftValues1 = np.zeros(NumValues)
        SoftValues2 = np.zeros(NumValues)
        for i in range(NumValues):
            Values1[i] = Values[i][Point1]
            Values2[i] = Values[i][Point2]
            SoftValues1[i] = SoftData[i][Point1]
            SoftValues2[i] = SoftData[i][Point2]
        Covariances = np.float32((Values1 - SoftValues1)*(Values2 - SoftValues2))
        Result[NumValues + 0:NumValues + NumValues] = Result[NumValues + 0:NumValues + NumValues] + Covariances
        Result[NumValues + NumValues] += 1
        # E-M46: no /2 — the one-sided scans (PointSetScanContStyle /
        # PointSetScanGridStyle window bounds) count each unordered pair
        # exactly once, mirroring the core twin
        # (src/geo_bsd/variogram.py:1005-1007 has no /2). The C++ /2
        # (variograms.cpp:723,925) is justified only for its two-sided
        # idx1 x idx2 cross-product with fabs, where each pair is counted
        # twice. The previous /2 halved the shared covariance.
        Result[0:NumValues] = Result[NumValues + 0:NumValues + NumValues] / Result[NumValues + NumValues]
    return Result

def CalcIndCorrelationFunction(Point1, Point2, Result, Params):
    Values = Params['HardData']
    SoftData = Params['SoftData']
    NumValues = len(Values)
    if Result is None:
        Result = np.zeros(NumValues + NumValues + 1, dtype=np.float32)
    else:
        Values1 = np.zeros(NumValues)
        Values2 = np.zeros(NumValues)
        SoftValues1 = np.zeros(NumValues)
        SoftValues2 = np.zeros(NumValues)
        for i in range(NumValues):
            Values1[i] = Values[i][Point1]
            Values2[i] = Values[i][Point2]
            SoftValues1[i] = SoftData[i][Point1]
            SoftValues2[i] = SoftData[i][Point2]
        # II-34: exclude pairs whose soft-prob variance product is
        # non-positive (soft prob 0 or 1) instead of substituting
        # denom=1.0. The old substitution (III-22) let the unnormalized raw
        # covariance term dominate the sum, silently diluting/inflating the
        # "correlation" outside [-1,1] whenever a soft probability is
        # exactly 0 or 1 (realistic for indicator data). The standard guard
        # (and the core twin, src/geo_bsd/variogram.py:1066-1095) excludes
        # the pair entirely. The pair-count slot is incremented only for
        # retained pairs so the average is computed over the valid subset.
        product = SoftValues1 * (1 - SoftValues1) * SoftValues2 * (1 - SoftValues2)
        valid = (product > 0) & ~np.isnan(product)

        denom = np.zeros_like(product)
        denom[valid] = product[valid] ** 0.5

        # Excluded pairs contribute 0 covariance (and are not counted in the
        # pair-count slot below). Divide only where valid to avoid 0/0 NaN
        # intermediate values (np.where evaluates both branches, so masking
        # after division would still warn on invalid entries).
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = (Values1 - SoftValues1) * (Values2 - SoftValues2) / denom
        Covariances = np.where(valid, raw, 0.0).astype(np.float32)
        Result[NumValues + 0:NumValues + NumValues] = Result[NumValues + 0:NumValues + NumValues] + Covariances
        Result[NumValues + NumValues] += int(valid.sum())
        # E-M46: no /2 — the one-sided scans count each unordered pair
        # exactly once, mirroring the core twin (variogram.py:1098-1101 has
        # no /2).
        Result[0:NumValues] = Result[NumValues + 0:NumValues + NumValues] / Result[NumValues + NumValues]
    return Result
