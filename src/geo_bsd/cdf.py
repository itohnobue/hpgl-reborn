# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
import warnings

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
        self.values = numpy.require(values, "float32")
        self.probs = numpy.require(probs, "float32")
        if len(self.values) != len(self.probs):
            raise ValueError(
                f"CdfData: values length ({len(self.values)}) "
                f"must match probs length ({len(self.probs)})"
            )


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
    data_flat = prop.data.flat
    mask_flat = prop.mask.flat

    informed = data_flat[mask_flat != 0]
    full_count = float(informed.size)
    if full_count == 0:
        raise ValueError("calc_cdf: no informed values (all cells are masked)")
    if numpy.any(numpy.isnan(informed)) or numpy.any(numpy.isinf(informed)):
        warnings.warn(
            "calc_cdf: informed values contain NaN or Inf values. "
            "Filtering them out before computing the CDF.",
            stacklevel=2,
        )
        informed = informed[numpy.isfinite(informed)]
        full_count = float(informed.size)
        if full_count == 0:
            raise ValueError(
                "calc_cdf: no informed values after filtering NaN/Inf — "
                "all informed cells contained NaN or Inf"
            )
    values, counts = numpy.unique(informed, return_counts=True)
    # numpy.unique returns sorted unique values
    size = values.size
    probs = numpy.zeros(size)
    last_prob = 0.0
    for i in range(size):
        probs[i] = last_prob + counts[i] / full_count
        last_prob = probs[i]
    return CdfData(values=values, probs=probs)
