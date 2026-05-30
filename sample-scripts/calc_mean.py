#!/usr/bin/env python3
"""
Standalone script: calculates the mean of valid values (non -99) in an INC file.

This is a minimal, self-contained demo that does NOT use the HPGL library.
For programmatic use, prefer `geo_bsd.calc_mean()` which operates on HPGL
ContProperty objects.

Usage: calc_mean.py <file>
"""

import sys

if len(sys.argv) != 2:
    print("Usage: calc_mean.py <file>")
    sys.exit()

filename = sys.argv[1]

values = []
with open(filename) as f:
    for line in f:
        ss = line.split()
        for s in ss:
            try:
                values += [float(s.strip())]
            except Exception:
                pass

total = 0
count = 0
for v in values:
    if v != -99:
        total += v
        count += 1
print(total)
print(count)
print(total / count)
