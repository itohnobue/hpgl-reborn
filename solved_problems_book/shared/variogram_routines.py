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
    SS1 = V @ VariogramSearchTemplate.Ellipsoid.Direction1
    SS2 = V @ VariogramSearchTemplate.Ellipsoid.Direction2
    SS3 = V @ VariogramSearchTemplate.Ellipsoid.Direction3

    S2 = SS2 / VariogramSearchTemplate.Ellipsoid.R2
    S3 = SS3 / VariogramSearchTemplate.Ellipsoid.R3

    Dist = np.power(np.power(S2, 2) + np.power(S3, 2), 0.5)
    Result = np.array(np.bitwise_and(Dist <= 1, VariogramSearchTemplate.TolDistance * Dist <= SS1))

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

    Dist = np.power(np.power(GI, 2) + np.power(GJ, 2) + np.power(GK, 2), 0.5)

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
    MinDistance2 = max(0, min(LagStart)) ** 2
    MaxDistance2 = max(LagEnd) ** 2

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

        FDistance2 = FDX ** 2 + FDY ** 2 + FDZ ** 2
        Filter = MinDistance2 <= FDistance2
        Filter = np.bitwise_and(Filter, FDistance2 <= MaxDistance2)

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance2 = FDistance2[Filter]

        Filter = _IsInTunnel(VariogramSearchTemplate, np.column_stack((FDX, FDY, FDZ)))

        FDX, FDY, FDZ = FDX[Filter], FDY[Filter], FDZ[Filter]
        FIndex = FIndex[Filter]
        FDistance2 = FDistance2[Filter]

        FDistance = FDistance2 ** 0.5

        for Lag in LagIndex:
            Filter = np.bitwise_and(LagStart[Lag] <= FDistance, FDistance < LagEnd[Lag])
            for j in FIndex[Filter]:
                Result[Lag, :] = Function(i, j, Result[Lag, :], Params)

    return Result, LagDistance

def PointSetScanGridStyle(VariogramSearchTemplate, PointSetXYZ, Function, Params):
    LI, LJ, LK, LagIndexes, LagDistance = _CalcLagsAreas(VariogramSearchTemplate)
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

        for j in range(len(FDI)):
            LFilter = FDI[j] == LI
            LFilter = np.bitwise_and(LFilter, FDJ[j] == LJ)
            LFilter = np.bitwise_and(LFilter, FDK[j] == LK)

            ActiveLags = LagIndexes[LFilter]

            if Function is not None:
                I2, J2, K2 = FPI[j], FPJ[j], FPK[j]
                for Lag in ActiveLags:
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
        Result[0:NumValues] = Result[NumValues + 0:NumValues + NumValues] / Result[NumValues + NumValues] / 2
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
        Covariances = np.float32((Values1 - SoftValues1)*(Values2 - SoftValues2) / (SoftValues1 * (1 - SoftValues1) * SoftValues2 * (1-SoftValues2)) ** 0.5)
        Result[NumValues + 0:NumValues + NumValues] = Result[NumValues + 0:NumValues + NumValues] + Covariances
        Result[NumValues + NumValues] += 1
        Result[0:NumValues] = Result[NumValues + 0:NumValues + NumValues] / Result[NumValues + NumValues] / 2
    return Result
