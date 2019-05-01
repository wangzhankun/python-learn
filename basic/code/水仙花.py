from math import *
outlst = []
for a in range(100,1000):
    s = str(a)
    out = 0
    for b in s:
        out += int(b) ** 3
    if out == a:
        outlst.append(s)
print(", ".join(outlst))