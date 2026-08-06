import os
import numpy as np
from python_property import load_property_python, save_property_python


def getconcube(cube):
    [x, y, z] = cube.shape
    # E-M21: int16 wrapped on >32767-cell cubes (negative labels/indices →
    # wrong connectivity). int32 covers any cube whose size fits the queue
    # arrays (coordinates and labels < x*y*z).
    conncube = np.zeros((x, y, z), dtype=np.int32)
    qI = np.zeros((x * y * z), dtype=np.int32)
    qJ = np.zeros((x * y * z), dtype=np.int32)
    qK = np.zeros((x * y * z), dtype=np.int32)
    compnum = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if (cube[i, j, k] == 1) and (conncube[i, j, k] == 0):
                    compnum = compnum + 1
                    conncube[i, j, k] = compnum
                    iqb = -1
                    iqe = 0
                    qI[iqe] = i
                    qJ[iqe] = j
                    qK[iqe] = k
                    while iqb < iqe:
                        iqb = iqb + 1
                        ni = qI[iqb]
                        nj = qJ[iqb]
                        nk = qK[iqb]
                        if ni > 0:
                            if (cube[ni - 1, nj, nk] == 1) and (conncube[ni - 1, nj, nk] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni - 1
                                qJ[iqe] = nj
                                qK[iqe] = nk
                                conncube[ni - 1, nj, nk] = compnum
                        if nj > 0:
                            if (cube[ni, nj - 1, nk] == 1) and (conncube[ni, nj - 1, nk] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni
                                qJ[iqe] = nj - 1
                                qK[iqe] = nk
                                conncube[ni, nj - 1, nk] = compnum
                        if nk > 0:
                            if (cube[ni, nj, nk - 1] == 1) and (conncube[ni, nj, nk - 1] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni
                                qJ[iqe] = nj
                                qK[iqe] = nk - 1
                                conncube[ni, nj, nk - 1] = compnum
                        if ni < (x - 1):
                            if (cube[ni + 1, nj, nk] == 1) and (conncube[ni + 1, nj, nk] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni + 1
                                qJ[iqe] = nj
                                qK[iqe] = nk
                                conncube[ni + 1, nj, nk] = compnum
                        if nj < (y - 1):
                            if (cube[ni, nj + 1, nk] == 1) and (conncube[ni, nj + 1, nk] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni
                                qJ[iqe] = nj + 1
                                qK[iqe] = nk
                                conncube[ni, nj + 1, nk] = compnum
                        if nk < (z - 1):
                            if (cube[ni, nj, nk + 1] == 1) and (conncube[ni, nj, nk + 1] == 0):
                                iqe = iqe + 1
                                qI[iqe] = ni
                                qJ[iqe] = nj
                                qK[iqe] = nk + 1
                                conncube[ni, nj, nk + 1] = compnum
    save_property_python(conncube, x, y, z, "SAVE_CUB.INC")
    return compnum


if __name__ == "__main__":
    # E2-45: getconcube() was never invoked — running the script was a
    # silent no-op. Run the documented workflow (README: "Concurrency
    # benchmark / stress test") on the shipped binary cube SAVE.INC
    # (286×10×1, 0/1 values) and report the component count.
    cube = load_property_python(
        286, 10, 1, os.path.join(os.path.dirname(__file__), "SAVE.INC"), True
    )
    print("Connected components:", getconcube(cube))
