# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""GSLIB reference-fact table (got-20260802092630 — recurring class).

The GSLIB contract facts below are the SINGLE source of truth for every
HPGL path that reads, writes, or transforms GSLIB-format data. Each run in
the v2.0.x series drifted on one facet (sentinel window, transform space,
ndmin, ordrel, OK->SK downgrade) because the facts were duplicated as inline
constants and comments across geo.py / routines.py / variogram.py /
read_inc_file.cpp. A path may NOT rely on a different interpretation — if a
new path needs a GSLIB contract fact, it MUST reference this table.

Every fact is documented against the canonical GSLIB source convention.
"""

from __future__ import annotations

import numpy

# ============================================================================
# Reference-fact table
# ============================================================================

# ---------------------------------------------------------------------------
# 1. Sentinel window (missing-value markers)
# ---------------------------------------------------------------------------
# GSLIB convention: values with magnitude > 1.0e21 are missing-value
# sentinels ("less than -1.0e21 or greater than 1.0e21"). The comparison is
# STRICT inequality at the window edge: an exact ±1.0e21 value is treated as
# a real (extreme) data value, not a sentinel. Readers convert out-of-window
# values to NaN (the numpy missing marker); writers reject them so a file the
# reader would round-trip to NaN is never produced.
GSLIB_SENTINEL_WINDOW = 1.0e21

# ---------------------------------------------------------------------------
# 2. Data-space vs normal-score space
# ---------------------------------------------------------------------------
# GSLIB property files carry DATA-space values. Normal-score transforms
# (normal_score, cdf_transform, SGS with a transform CDF) operate in a
# separate normal-score space; mixing the two (e.g. comparing data-space
# output against normal-score thresholds) silently corrupts results. Every
# transform must state which space its output lives in.
GSLIB_DATA_SPACE = "data"
GSLIB_NORMAL_SCORE_SPACE = "normal-score"

# ---------------------------------------------------------------------------
# 3. ndmin — original-data-only
# ---------------------------------------------------------------------------
# GSLIB ndmin (minimum neighbours) gates on the count of ORIGINAL conditioning
# data nodes, not the total number of neighbouring nodes (which for SGS/SIS
# includes already-simulated nodes). Sequential simulation.h:104-114 skips a
# node whose ORIGINAL-data neighbour count is below ndmin, even when the total
# (original + simulated) count is inflated. Python-side expected-cell counts
# must mirror this per-node original-data gate, not a total-count gate.
GSLIB_NDMIN_ORIGINAL_DATA_ONLY = True

# ---------------------------------------------------------------------------
# 4. OK -> SK downgrade (missing-secondary cokriging)
# ---------------------------------------------------------------------------
# When a cokriging secondary is undefined at the target node, the secondary
# equation is dropped from the system entirely and the node is kriged as
# primary-only (the F-22 / GSLIB convention). The secondary MEAN is NOT
# substituted while keeping the full-variance secondary equation — that would
# produce an estimate that is not BLUP. Applies identically when the secondary
# variance is not a strictly-positive finite value (II-10).
GSLIB_OK_SK_DOWNGRADE_DROPS_SECONDARY_EQUATION = True

# ---------------------------------------------------------------------------
# 5. ordrel — order-relations correction
# ---------------------------------------------------------------------------
# GSLIB order-relations (ordrel) correction enforces non-decreasing posterior
# indicator probabilities P_0 <= P_1 <= ... after kriging. It operates on the
# POSTERIOR PROBABILITY space; applying it in another space (or skipping it
# where the sampler requires monotonic CDF input) silently permutes category
# selection. HPGL applies the correction only where the sampler consumes
# probabilities as a cumulative CDF.
GSLIB_ORDREL_SPACE = "posterior-probability"


# ============================================================================
# Helpers — the table facts as callable predicates (used by every boundary)
# ============================================================================

def is_gslib_missing_sentinel(values: numpy.ndarray) -> numpy.ndarray:
    """Boolean mask of GSLIB missing-value sentinels (|v| > 1.0e21, strict).

    Args:
        values: Float array of candidate values.

    Returns:
        Boolean array, True where the value is a GSLIB missing sentinel.
    """
    arr = numpy.asarray(values, dtype=numpy.float64)
    return numpy.abs(arr) > GSLIB_SENTINEL_WINDOW


def is_gslib_sentinel(value: float) -> bool:
    """Single-value check for the GSLIB missing sentinel (|v| > 1.0e21)."""
    return abs(float(value)) > GSLIB_SENTINEL_WINDOW
