import numpy as np


def load_gslib_file(filename):
    dict_data = {}
    list_prop = []

    with open(filename) as f:
        f.readline()  # Skip caption line
        num_p = int(f.readline())

        # E-M50: validate the header like the core loader
        # (src/geo_bsd/routines.py:762-784) — num_p sanity and duplicate
        # property names. Duplicate names silently corrupt data (both
        # columns accumulate into the same dict key, overwriting the
        # earlier property's array).
        if num_p < 1:
            raise ValueError(
                f"load_gslib_file: num_p must be at least 1, got {num_p}"
            )
        seen = set()
        for i in range(num_p):
            prop_name = str(f.readline().strip())
            if prop_name in seen:
                raise ValueError(
                    f"load_gslib_file: duplicate property name '{prop_name}' "
                    f"in GSLIB header. Each property name must be unique."
                )
            seen.add(prop_name)
            list_prop.append(prop_name)

        for i in range(len(list_prop)):
            dict_data[list_prop[i]] = np.array([])

        for line in f:
            if not line.strip():
                continue
            points = line.split()
            # E-M50: token-count validation (core LoadGslibFile parity,
            # routines.py:858-881). A GSLIB data line must carry exactly
            # num_p tokens — fewer silently dropped the trailing columns,
            # more raised an opaque IndexError on list_prop.
            if len(points) != num_p:
                raise RuntimeError(
                    f"load_gslib_file: expected {num_p} values per data line, "
                    f"got {len(points)} tokens in line: {line.strip()!r}"
                )
            for j in range(len(points)):
                dict_data[list_prop[j]] = np.concatenate(
                    (dict_data[list_prop[j]], np.array([np.float64(points[j])]))
                )

    # E-M50/F-M18: GSLIB missing-value trimming — finite values outside the
    # ±1.0e21 window (strict inequality per the GSLIB convention,
    # gslib_ref.py:27-33) are missing sentinels, not real data. Convert them
    # to NaN (the numpy missing marker) so downstream mean/variogram/
    # kriging do not silently compute with third-party sentinel magnitudes
    # (core loader convention, routines.py:902-910).
    for key in list_prop:
        values = dict_data[key]
        dict_data[key] = np.where(np.abs(values) > 1.0e21, np.nan, values)

    return dict_data
