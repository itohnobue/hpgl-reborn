#!/usr/bin/env python3

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
