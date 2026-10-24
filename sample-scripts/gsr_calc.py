import sys
import os
import numpy as np

# F-355: the package lives in src/geo_bsd (readme.md:269-274), not at the
# repository root — inserting '..' left `import geo_bsd` raising
# ModuleNotFoundError before any script logic ran.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from geo_bsd import ContProperty, load_cont_property


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


if __name__ == "__main__":
    # F-354: gsr_calc() was never invoked — the script is listed in
    # sample-scripts/README.md as a runnable calculation, but running it
    # printed nothing and exited 0 (a silent no-op, the class the four
    # sibling scripts were fixed for). Run it on the sample cube shipped
    # next to this script (CUBE.INC, 286x10x1 = 2860 cells) and print the
    # per-layer GSR. basedir pins the path-containment base to this script's
    # directory so the load does not depend on the current working directory.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cube = load_cont_property(
        os.path.join(_script_dir, "CUBE.INC"), -99, (286, 10, 1),
        basedir=_script_dir,
    )
    print(gsr_calc(_cube, 286, 10, 1))
