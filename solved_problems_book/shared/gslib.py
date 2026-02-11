import numpy as np


def load_gslib_file(filename):
    dict_data = {}
    list_prop = []

    with open(filename) as f:
        name = f.readline()
        num_p = int(f.readline())

        for i in range(num_p):
            list_prop.append(str(f.readline().strip()))

        for i in range(len(list_prop)):
            dict_data[list_prop[i]] = np.array([])

        for line in f:
            points = line.split()
            for j in range(len(points)):
                dict_data[list_prop[j]] = np.concatenate(
                    (dict_data[list_prop[j]], np.array([np.float64(points[j])]))
                )
    return dict_data
