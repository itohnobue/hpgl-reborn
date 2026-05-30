import ctypes as C

import numpy

# NumPy 2.0+ compatibility
from numpy import ctypeslib as NC

# Since numpy>=2.0 is required, always use direct ctypes.CDLL
_load_lib_func = lambda libpath: C.CDLL(str(libpath))

ndpointer = NC.ndpointer

#from _cvariogram import CStackLayers

from .hpgl_wrap import _safe_load_library

cvar = _safe_load_library('_cvariogram', __file__)

class vector_t(C.Structure):
    _fields_ = [("data", C.c_double * 3)]

class ellipsoid_t(C.Structure):
    _fields_ = [
        ("direction1", vector_t),
        ("direction2", vector_t),
        ("direction3", vector_t),
        ("R1", C.c_double),
        ("R2", C.c_double),
        ("R3", C.c_double)]

class variogram_search_template_t (C.Structure):
    _fields_ = [
        ("lag_width", C.c_double),
        ("lag_separation", C.c_double),
        ("tol_distance", C.c_double),
        ("num_lags", C.c_int),
        ("first_lag_distance", C.c_double),
        ("ellipsoid", ellipsoid_t)]


class hard_data_t (C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("mask", C.POINTER(C.c_ubyte)),
        ("data_shape", (C.c_int * 3)),
        ("data_strides", (C.c_int * 3)),
        ("mask_shape", (C.c_int * 3)),
        ("mask_strides", (C.c_int * 3))]

class cont_point_set_t (C.Structure):
    _fields_ = [
        ("xs", C.POINTER(C.c_float)),
        ("ys", C.POINTER(C.c_float)),
        ("zs", C.POINTER(C.c_float)),
        ("values", C.POINTER(C.c_float)),
        ("size", C.c_int)]

class float_data_t (C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("data_shape", (C.c_int * 3)),
        ("data_strides", (C.c_int * 3))]


cvar.fill_ellipsoid_directions.restype = None
cvar.fill_ellipsoid_directions.argtypes = [
    C.POINTER(ellipsoid_t), C.c_double, C.c_double, C.c_double]

cvar.calc_variograms.restype = None
cvar.calc_variograms.argtypes = [
            C.POINTER(variogram_search_template_t),
            C.POINTER(hard_data_t),
            NC.ndpointer(dtype = numpy.float32),
            C.c_int,
            C.c_int]

cvar.calc_variograms_from_point_set.restype = None
cvar.calc_variograms_from_point_set.argtypes = [
            C.POINTER(variogram_search_template_t),
            C.POINTER(cont_point_set_t),
            NC.ndpointer(dtype = numpy.float32),
            C.c_int]

cvar.cvar_stack_layers.restype = None
cvar.cvar_stack_layers.argtypes = [
    C.POINTER(float_data_t),
    C.POINTER(C.c_int),
    C.c_int,
    C.c_int,
    C.c_float,
    C.c_int,
    C.POINTER(float_data_t)]

def checked_create(T, **kargs):
    fields = []
    for f, _ in T._fields_:
        fields.append(f)
    for k in kargs.keys():
        if k in fields:
            fields.remove(k)
    assert len(fields) == 0, "No values for parameters: %s" % fields
    result = T(**kargs)
    # Preserve references to input values to prevent garbage collection
    # while C code holds pointers to the underlying data
    result._array_refs = tuple(kargs.values())
    return result

def __strides(array):
    ndim = array.ndim
    if ndim == 1:
        return (1, array.shape[0], array.shape[0])
    elif ndim == 2:
        return (1, array.shape[0], array.shape[0] * array.shape[1])
    elif ndim >= 3:
        return (array.strides[0] // array.itemsize, array.strides[1] // array.itemsize, array.strides[2] // array.itemsize)
    else:
        raise ValueError(f"__strides: array must have at least 1 dimension, got ndim={ndim}")

def _c_array(t, size, values):
    arr = (t * size)(*values)
    arr._array_refs = tuple(values)
    return arr

class Ellipsoid:
    def __init__(self, R1, R2, R3, azimuth, dip, rotation):
        vec = checked_create(vector_t, data = _c_array(C.c_double, 3, (0,0,0)))
        self.ell = checked_create(
            ellipsoid_t,
            direction1 = vec,
            direction2 = vec,
            direction3 = vec,
            R1 = R1,
            R2 = R2,
            R3 = R3)
        cvar.fill_ellipsoid_directions(C.byref(self.ell), azimuth, dip, rotation)


class VariogramSearchTemplate:
    def __init__(self, lag_width, lag_separation, tol_distance, num_lags, first_lag_distance, ellipsoid):
        self.templ = checked_create(
            variogram_search_template_t,
            lag_width = lag_width,
            lag_separation = lag_separation,
            tol_distance = tol_distance,
            num_lags = num_lags,
            first_lag_distance = first_lag_distance,
            ellipsoid = ellipsoid.ell)

        self.num_lags = num_lags
        self.lag_separation = lag_separation


def CalcVariograms(templ, hard_data, percent=100):
    if templ.num_lags <= 0:
        raise ValueError("CalcVariograms: num_lags must be positive")
    if percent < 1 or percent > 100:
        raise ValueError(f"CalcVariograms: percent must be in [1, 100], got {percent}")
    if not isinstance(hard_data[0], numpy.ndarray) or hard_data[0].dtype != numpy.float32:
        raise TypeError(f"CalcVariograms: hard_data[0] must be a float32 ndarray, got {type(hard_data[0]).__name__}")
    if not isinstance(hard_data[1], numpy.ndarray) or hard_data[1].dtype != numpy.uint8:
        raise TypeError(f"CalcVariograms: hard_data[1] must be a uint8 ndarray, got {type(hard_data[1]).__name__}")
    variogram = numpy.array([0] * templ.num_lags, dtype='float32')

    hd = checked_create(
        hard_data_t,
        data = hard_data[0].ctypes.data_as(C.POINTER(C.c_float)),
        mask = hard_data[1].ctypes.data_as(C.POINTER(C.c_ubyte)),
        data_shape = _c_array(C.c_int, 3, hard_data[0].shape),
        mask_shape = _c_array(C.c_int, 3, hard_data[1].shape),
        data_strides = _c_array(C.c_int, 3, __strides(hard_data[0])),
        mask_strides = _c_array(C.c_int, 3, __strides(hard_data[1])))

    cvar.calc_variograms(
        C.byref(templ.templ),
        C.byref(hd),
        variogram,
        variogram.size,
        percent)

    lags_borders = numpy.zeros(templ.num_lags)

    for k in range(templ.num_lags):
        lags_borders[k] = k * templ.lag_separation

    return (lags_borders, variogram)

def CalcVariogramsFromPointSet(templ, point_set, variogram):
    if templ.num_lags <= 0:
        raise ValueError("CalcVariogramsFromPointSet: num_lags must be positive")
    for key in ("X", "Y", "Z", "Property"):
        if key not in point_set:
            raise ValueError(f"CalcVariogramsFromPointSet: point_set missing required key '{key}'")
    if variogram is None:
        variogram = numpy.array([0] * templ.num_lags, dtype='float32')

    ps = checked_create(
        cont_point_set_t,
        xs = point_set["X"].ctypes.data_as(C.POINTER(C.c_float)),
        ys = point_set["Y"].ctypes.data_as(C.POINTER(C.c_float)),
        zs = point_set["Z"].ctypes.data_as(C.POINTER(C.c_float)),
        values = point_set["Property"].ctypes.data_as(C.POINTER(C.c_float)),
        size = point_set["Property"].size)

    cvar.calc_variograms_from_point_set(
        C.byref(templ.templ),
        C.byref(ps),
        variogram,
        variogram.size)

    lags_borders = numpy.zeros(templ.num_lags)

    for k in range(templ.num_lags):
        lags_borders[k] = k * templ.lag_separation

    return (lags_borders, variogram)

def _create_float_data(array):
    return checked_create(
        float_data_t,
        data = array.ctypes.data_as(C.POINTER(C.c_float)),
        data_shape = _c_array(C.c_int, 3, array.shape),
        data_strides = _c_array(C.c_int, 3, __strides(array)))

def CStackLayers(layers, markers, nz, scalez, blank_value, result):
    if len(layers) == 0:
        raise ValueError("CStackLayers: layers list is empty")
    if nz <= 0:
        raise ValueError(f"CStackLayers: nz must be positive, got {nz}")
    layers2 = []
    for layer in layers:
        layers2.append(_create_float_data(layer))

    result2 = _create_float_data(result)
    cvar.cvar_stack_layers(
        _c_array(float_data_t, len(layers2), layers2),
        _c_array(C.c_int, len(markers), markers),
        len(layers2),
        nz,
        scalez,
        blank_value,
        C.byref(result2))

