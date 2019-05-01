'''
费马大定理断言：当整数n>2时，关于a，b，c的方程a^n=b^n+c^n没有正整数解。
该定理被提出后，历经三百多年，经历多人猜想辩证，
最终在1995年被英国数学家安德鲁·怀尔斯证明。
当然，可以找到大于1的4个整数满足完美立方等式：
a^3=b^3+c^3+d^3（例如12^3=6^3+8^3+10^3）。
编写一个程序，对于任意给定的正整数N（N ≤100），
寻找所有的四元组（a,b,c,d），满足a^3=b^3+c^3+d^3，其中1<a，b，c，d≤N。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
'''
from math import *
N = int(input())
lst = []
for a in range(2,N+1):
    for b in range(1,a):
        for c in range(1,b+1):
            if a ** 3 < b ** 3 + c ** 3 + 1:
                break
            for d in range(1,c+1):
                #print(b,c,d)
                if a ** 3 < b ** 3 + c ** 3 + d ** 3:
                    break
                if a ** 3 == b ** 3 + c ** 3 + d ** 3:
                    lst.append((a,d,c,b))
out = sorted(lst,key = lambda x:(x[0],x[1],x[2],x[3]))
for i in out:
    print("Cube = {},Tripe = ({},{},{})".format(i[0],i[1],i[2],i[3]))