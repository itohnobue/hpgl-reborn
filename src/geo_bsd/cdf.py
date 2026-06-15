import numpy

__all__ = ["CdfData", "calc_cdf"]


class CdfData:
    """Empirical cumulative distribution function (CDF) data container.

    Holds the sorted unique values and their cumulative probabilities
    computed from a property.

    Parameters
    ----------
    values : numpy.ndarray
        1D array of sorted unique property values (float32).
    probs : numpy.ndarray
        1D array of cumulative probabilities corresponding to each
        value in ``values``. Probabilities are in [0, 1] and
        monotonically non-decreasing.
    """
    def __init__(self, values, probs):
        self.values = numpy.require(values, 'float32')
        self.probs = numpy.require(probs, 'float32')

def calc_cdf(prop):
    """Compute the empirical CDF from a ``ContProperty``.

    Counts unique values among informed (unmasked) cells and
    accumulates cumulative probabilities.

    Parameters
    ----------
    prop : ContProperty
        Continuous property with ``data`` and ``mask`` attributes.

    Returns
    -------
    CdfData
        Object with ``values`` (sorted unique property values) and
        ``probs`` (cumulative probabilities).

    Raises
    ------
    ValueError
        If no informed values exist (all cells masked).

    Notes
    -----
    Supports both 1D (flat) and 3D (grid) property data. The output
    ``CdfData`` is used as input to ``geo_bsd.sgs_simulation``.
    """
    # Handle both 1D (flat) and 3D (grid) property data
    if prop.data.ndim == 3:
        dx, dy, dz = prop.data.shape
        total_cells = dx * dy * dz
        data_flat = prop.data.flat
        mask_flat = prop.mask.flat
    else:
        total_cells = prop.data.size
        data_flat = prop.data.flat
        mask_flat = prop.mask.flat

    counts = {}
    full_count = 0
    for i in range(total_cells):
        if mask_flat[i] != 0:
            value = data_flat[i]
            full_count += 1
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
    full_count = float(full_count)
    if full_count == 0:
        raise ValueError("calc_cdf: no informed values (all cells are masked)")
    values = numpy.sort(list(counts.keys()))
    if values.size == 0:
        size = 0
    elif values.size == 1:
        size = 1
    else:
        size = values.size - 1
    values = numpy.resize(values, size)
    probs = numpy.zeros(values.shape)
    last_prob = 0.0
    for i in range(size):
        probs[i] = last_prob + counts[values[i]] / full_count
        last_prob = probs[i]
    return CdfData(values = values, probs = probs)
