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
        # Warn if input arrays are higher precision than float32 — numpy.require
        # silently downcasts, losing precision without notice.
        for name, arr in (("values", numpy.asarray(values)), ("probs", numpy.asarray(probs))):
            if arr.dtype == numpy.float64:
                warnings.warn(
                    f"CdfData: {name} is float64 and will be silently downcast "
                    f"to float32 — potential precision loss.",
                    stacklevel=2,
                )
        # F-M25: force C-contiguity. numpy.require without requirements
        # PRESERVES the input's memory layout (an F-ordered input stays
        # F-contiguous), but create_nonparam_cdf passes these arrays to the
        # C++ non-parametric CDF via raw ctypes pointers that read linearly —
        # a non-C-contiguous layout would be misread. All 9 geo.py sites pass
        # an explicit layout requirement; ours must be "C" (the values/probs
        # are consumed linearly, not per-property-cell like the Fortran data).
        self.values = numpy.require(values, "float32", requirements=["C"])
        self.probs = numpy.require(probs, "float32", requirements=["C"])

        # Validate values and probs for NaN / Inf BEFORE range checks.
        # NaN falsifies all three range/diff comparisons below
        # (NaN < 0.0, NaN > 1.0, and numpy.diff with NaN are all False),
        # so NaN passes the existing checks silently.
        if not numpy.all(numpy.isfinite(self.values)):
            raise ValueError("CdfData: values contain NaN or Inf")
        if not numpy.all(numpy.isfinite(self.probs)):
            raise ValueError("CdfData: probabilities contain NaN or Inf")

        if len(self.values) != len(self.probs):
            raise ValueError(
                f"CdfData: values length ({len(self.values)}) "
                f"must match probs length ({len(self.probs)})"
            )

        # Validate data integrity beyond length check
        if len(self.probs) > 0:
            # Probabilities must be in [0, 1]
            if numpy.any(self.probs < 0.0) or numpy.any(self.probs > 1.0):
                raise ValueError(
                    "CdfData: probabilities must be in [0, 1] range"
                )

            # Probabilities must be monotonically non-decreasing (CDF property)
            if numpy.any(numpy.diff(self.probs) < 0.0):
                raise ValueError(
                    "CdfData: probabilities must be monotonically non-decreasing"
                )

            # Values should be monotonically non-decreasing (sorted unique values)
            if numpy.any(numpy.diff(self.values) < 0.0):
                raise ValueError(
                    "CdfData: values must be monotonically non-decreasing"
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
    # F-04: the final cumulative probability is exactly 1.0 (sum of counts
    # == full_count). A p=1.0 CDF value feeds the SGS back-transform where
    # the C++ inverse maps it to the median — silently destroying the max
    # datum. Clamp the last value to the largest float32 strictly below
    # 1.0 (nextafter), matching cpp-A's tail-saturation convention in
    # gaussian_distribution.h/non_parametric_cdf.h, so the max datum maps
    # to a large-but-finite normal score instead of the median.
    #
    # F-M11: apply the clamp AFTER the float32 downcast. The float64
    # accumulation above can (a) leave the tail in [1-2^-25, 1.0) — below
    # the old `>= 1.0` threshold, yet rounded to exactly 1.0f by the float32
    # downcast (clamp bypassed → max datum maps to the median again); and
    # (b) when full_count >= ~2^25, round EARLIER cumulative probabilities
    # up to exactly 1.0f, producing a non-monotonic tail [.., 1.0f,
    # 0.99999994f] and a spurious monotonicity ValueError. Downcasting first
    # lets the clamp see the exact float32 values the CDF stores. Rounding
    # a non-decreasing sequence is monotone, so after clamping probs[-1] the
    # only possible violation is a suffix of exactly-1.0f entries above the
    # clamped tail — cap that suffix down to keep the float32 output
    # monotonically non-decreasing.
    probs = numpy.require(probs, "float32")
    if size > 0 and probs[-1] >= 1.0:
        clamped = numpy.nextafter(numpy.float32(1.0), numpy.float32(0.0))
        probs[-1] = clamped
        for i in range(size - 2, -1, -1):
            if probs[i] > clamped:
                probs[i] = clamped
            else:
                break
    return CdfData(values=values, probs=probs)
