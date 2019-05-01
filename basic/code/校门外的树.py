'''
某校门外长度为L的马路上有一排树，每两棵相邻的树之间的间隔都是1m。
可以把马路看成一个数轴，马路的一端在数轴0的位置，另一端在L的位置；
数轴上的每个整数点，即0,1,2，...，L都种有一棵树。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
马路上有一些区域要用来建地铁，这些区域用它们在数轴上的起始点和终止点表示。
已知任一区域的起始点和终止点的坐标都是整数，区域之间可能有重合的部分。
现在要把这些区域中的树（包括区域端点处的两棵树）移走，任务是计算将这些树都移走后，马路上有剩多少棵树。
'''
'''
思路：
本题主要难点在于区域归并
可以将一对坐标作为一个tuple放入list当中
依据第一个坐标进行升序排列
按照栈结构弹出两个栈顶元素进行归并：
    如果top1[0]>top2[1]||top1[1]<top2[0],则不能进行归并：
        计算top1的减少树的数目
        并将top2入栈
    否则可以进行归并：
        对四个数字进行排序，取最小值和最大值作为两端点，并压入原栈
'''


def GetCoordinate(lst):
    for i in range(N):
        x, y = map(int, input().split())
        lst.append((x, y))


def Judge(lst):
    if len(lst) < 2:
        return lst[0][1] - lst[0][0] + 1
    coordiate1 = lst.pop()
    coordiate2 = lst.pop()
    if coordiate1[0] > coordiate2[1] or coordiate1[1] < coordiate2[0]:
        lst.append(coordiate2)
        return coordiate1[1] - coordiate1[0] + 1 + Judge(lst)
    x = coordiate2[0]
    y = coordiate1[1] if coordiate1[1] > coordiate2[1] else coordiate2[1]
    lst.append((x, y))
    return Judge(lst)

L, N = map(int, input().split())
lst = []
GetCoordinate(lst)
lst.sort(key=lambda x: x[0])
decrease = Judge(lst)
print(L + 1 - decrease)