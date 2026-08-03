import os

# F-54: no script produces RESULT_SIS.INC (the SIS example workflow in
# ntg_calc_hist.py writes RESULT.INC). Reading a nonexistent file made
# this statistics helper fail with FileNotFoundError. Point it at the real
# SIS output produced by the sample workflow.
with open(os.path.join(os.path.dirname(__file__), "..", "RESULT.INC")) as f:
    lines = f.readlines()
d = {}
for line in lines:
    if line in d:
        d[line] += 1
    else:
        d[line] = 1

print(d)
