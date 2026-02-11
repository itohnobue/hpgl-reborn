import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import ContProperty, IndProperty


def prop2array(prop, x, y, z, undefined_value):
    if prop.data.size != (x * y * z):
        print("Error! Property size is:", prop.data.size, "but given x*y*z size is:", x * y * z)
        return
    if isinstance(prop, ContProperty):
        values = np.zeros(prop.data.size, dtype=np.float64)
        values_right = np.zeros((x, y, z), dtype=np.float64)
    else:
        values = np.zeros(prop.data.size, dtype=np.int32)
        values_right = np.zeros((x, y, z), dtype=np.int32)

    for i in range(prop.data.size):
        if prop.mask.flat[i] > 0:
            values[i] = prop.data.flat[i]
        else:
            values[i] = undefined_value

    values = np.array(values).reshape(z, y, x)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                values_right[i, j, k] = values[k, j, i]

    return values_right


def array2prop(array_prop, undefined_value):
    x = np.size(array_prop, 0)
    y = np.size(array_prop, 1)
    z = np.size(array_prop, 2)

    if array_prop.dtype != np.dtype('int32'):
        print("Array is float, creating continuous property...")
        data = np.zeros(array_prop.size, dtype='float32')
        mask = np.zeros(array_prop.size, dtype='uint8')
    else:
        print("Array is int, creating indicator property...")
        indicators = np.unique(array_prop)
        indicators_t = []
        for k in range(indicators.size):
            if indicators[k] != undefined_value:
                indicators_t.append(indicators[k])
        print("Indicators is:", indicators_t)
        data = np.zeros(array_prop.size, dtype='uint8')
        mask = np.zeros(array_prop.size, dtype='uint8')

    values_for_prop = np.zeros((z, y, x), dtype=array_prop.dtype)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                values_for_prop[k, j, i] = array_prop[i, j, k]

    size_p = x * y * z
    values_for_prop = values_for_prop.reshape(size_p)

    for k in range(size_p):
        if values_for_prop[k] != undefined_value:
            data[k] = values_for_prop[k]
            mask[k] = 1

    if array_prop.dtype != np.dtype('int32'):
        return ContProperty(data, mask)
    else:
        return IndProperty(data.astype('uint8'), mask, len(indicators_t))
    print("Done")
