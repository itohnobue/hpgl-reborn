import numpy as np


def load_property_python(x, y, z, filename, intype=False):
    values = []
    if intype is False:
        intype = int
    elif intype is True:
        intype = float
    values_right = np.zeros((x, y, z), dtype=intype)
    with open(filename) as f:
        for line in f:
            if "--" in line:
                line = line[:line.index("--")]
            ss = line.split()
            for s in ss:
                try:
                    values += [intype(s.strip())]
                except Exception:
                    pass
    values = np.array(values).reshape(z, y, x)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                values_right[i, j, k] = values[k, j, i]

    return values_right


def save_property_python(prop_array, x, y, z, filename, cube_name="CUBE"):
    with open(filename, "w+") as f:
        f.write(cube_name)
        prop_array = np.array(prop_array).reshape(x * y * z)
        for i in range(x * y * z):
            if i % 12 == 0:
                f.write("\n")
            f.write(str(prop_array[i]))
            f.write(" ")
        if (x * y * z) % 12 >= 0:
            f.write("\n")
        f.write("/")
