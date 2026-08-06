# F-54: no script produces RESULT_SIS.INC (the SIS example workflow in
# ntg_calc_hist.py writes RESULT.INC). Reading a nonexistent file made
# this statistics helper fail with FileNotFoundError. Point it at the real
# SIS output produced by the sample workflow.
# E-M18: the producer writes RESULT.INC CWD-relative, so a __file__-anchored
# read here broke whenever the scripts were run from a directory other than
# the repo root. Read CWD-relative to match the producer — both scripts then
# stay aligned when run from the same working directory (any directory;
# write_property's PathValidator base is the process CWD, so a
# __file__-anchored write outside the CWD would be rejected anyway).
with open("RESULT.INC") as f:
    lines = f.readlines()
d = {}
for line in lines:
    if line in d:
        d[line] += 1
    else:
        d[line] = 1

print(d)
