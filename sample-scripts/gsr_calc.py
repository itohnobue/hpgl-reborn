import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_bsd import ContProperty


def gsr_calc(prop, x, y, z):
    if isinstance(prop, ContProperty):
        values_result = np.zeros([x, y, z], dtype=float)
    else:
        values_result = np.zeros([x, y, z], dtype=int)
    n = 0
    for k in range(z):
        for j in range(y):
            for i in range(x):
                if prop.mask.flat[n] > 0:
                    values_result[i, j, k] = prop.data.flat[n]
                else:
                    values_result[i, j, k] = -99999
                n = n + 1

    gsr = np.empty(z, dtype=float)

    if not isinstance(prop, ContProperty):
        czeros = 0.0
        cones = 0.0
        for k in range(z):
            for i in range(x):
                for j in range(y):
                    if values_result[i, j, k] == 1:
                        cones = cones + 1
                    if values_result[i, j, k] == 0:
                        czeros = czeros + 1
            if (cones + czeros) == 0:
                gsr[k] = 0
            else:
                gsr[k] = cones / (cones + czeros)
            czeros = 0.0
            cones = 0.0
    else:
        mean = 0.0
        num = 0.0
        for k in range(z):
            for i in range(x):
                for j in range(y):
                    if values_result[i, j, k] != -99999:
                        mean = mean + values_result[i, j, k]
                        num = num + 1
            if num == 0:
                gsr[k] = 0
            else:
                gsr[k] = mean / num
            mean = 0.0
            num = 0.0

    return gsr
