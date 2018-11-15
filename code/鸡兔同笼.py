#输出最少动物数和最多动物数
from math import *
n = int(input())#测试用例组数
lst = []
for i in range(0,n):
    a = int(input())#笼子里脚的数量总数
    for x in range(0,int(a / 2) + 1):
        for y in range(0, int(a / 4) + 1):
            if a == 2 * x + 4 * y:
                lst.append(x+y)
    if lst:#list若为空则为false
        lst.sort()
        print("{} {}".format(lst[0],lst[-1]))
    else:
        print("0 0")