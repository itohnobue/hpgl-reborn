#!/usr/bin/env python3
"""
Standalone script: calculates the mean of valid values in an INC or GSLIB file.

Valid values are the data tokens that are neither the undefined marker (-99)
nor a missing-value sentinel (NaN, or a magnitude above 1.0e21 — the GSLIB
convention documented in gslib_ref.py:44-50). GSLIB header fields, "--"
comment lines and the INC end-of-data marker "/" are not data: the token
rules mirror the library readers (geo.py:736-804).

This is a minimal, self-contained demo that does NOT use the HPGL library.
For programmatic use, prefer `geo_bsd.calc_mean()` which operates on HPGL
ContProperty objects.

Usage: calc_mean.py <file>
"""

import math
import sys

if len(sys.argv) != 2:
    print("Usage: calc_mean.py <file>")
    sys.exit()

filename = sys.argv[1]

UNDEFINED_VALUE = -99.0
# GSLIB missing-value window. The comparison is STRICT at the edge: an exact
# ±1.0e21 value is data, not a sentinel (gslib_ref.py:44-50, geo.py:800-804).
SENTINEL_WINDOW = 1.0e21


def is_missing(value):
    """True for the undefined marker, a GSLIB sentinel and NaN."""
    return (
        math.isnan(value)
        or value == UNDEFINED_VALUE
        or value < -SENTINEL_WINDOW
        or value > SENTINEL_WINDOW
    )


def _data_tokens(line):
    """Data tokens of one line: a "--" token starts a comment that runs to the
    end of the line, and "/" is the INC end-of-data marker."""
    tokens = []
    for token in line.split():
        if token.startswith("--"):
            break
        if not token.startswith("/"):
            tokens.append(token)
    return tokens


def _data_values(line):
    """Float values of one line; caption and property-name tokens are skipped."""
    values = []
    for token in _data_tokens(line):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def _gslib_header_lines(content):
    """Number of GSLIB header lines at the top of ``content``, or 0 for an INC file.

    F-376: a GSLIB file starts with a caption line, a bare-integer property
    count and that many property-name lines (property_writer.cpp:400-404);
    averaging those numbers in as data is the defect. The name lines must be
    non-numeric, so an INC caption followed by integer data
    (sample-scripts/CUBE.INC starts "CUBE" / "1" / "1") is never mistaken for
    a header — the fingerprint is the one the library readers apply
    (geo.py:687-720), narrowed here so a data file is never truncated.
    """
    if len(content) < 3:
        return 0
    try:
        float(content[0].split()[0])
        return 0  # numeric caption → pure data file, not a GSLIB header
    except ValueError:
        pass
    try:
        nvar = int(content[1])
    except ValueError:
        return 0
    if nvar < 1 or len(content) < 2 + nvar:
        return 0
    for line in content[2:2 + nvar]:
        if _data_values(line):
            return 0  # numeric name line → INC data, not a GSLIB header
    return 2 + nvar


with open(filename) as f:
    # "--" comment lines carry numeric text ("Copyright 2009", "1. Redistributions")
    # that must never reach the mean.
    content = [
        line for line in f
        if line.strip() and not line.strip().startswith("--")
    ]

values = []
for line in content[_gslib_header_lines(content):]:
    values += _data_values(line)

total = 0
count = 0
for v in values:
    if not is_missing(v):
        total += v
        count += 1
if count == 0:
    print(f"calc_mean.py: no valid values in '{filename}'", file=sys.stderr)
    sys.exit(1)
print(total)
print(count)
print(total / count)
