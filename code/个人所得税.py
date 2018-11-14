from math import *
a = int(input())
if a <= 3500:
    a = 0
else:
    a -= 3500
out = 0
lst = [80000,55000,35000,9000,4500,1500,0]
lt  = [0.45,0.35,0.30,0.25,0.20,0.10,0.03]
for k in range(7):
    x , y = lst[k], lt[k]
    if a < x:
        continue
    out = out + (a - x) * y
    a = x
if out == 0:
    print(0)
else:
    print("%f"%(out))