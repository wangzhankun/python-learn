from math import *
'''
给出4个小于10的正整数，可以使用加、减、乘、除4种运算以及括号把4个数连接起来得到一个表达式。
现在问题是，是否存在一种方式使得所得表达式的结果等于24。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
这里加、减、乘、除以及括号的运算结果和运算优先级跟平常定义一致。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
例如，对于5，5，5，1，可知5×（5－1／5）＝24。又如，对于1，1，4，2无论如何都不能得到24‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
'''
'''
思路：
 对于长度为4的算式，我们可以先拿出2个数运算，之后4个数就变成3个数了,然后3→2,2→1；
 在1处判断并退栈。其次，对于拿出来的两个数，我们有“  +  -  *  /  ” 四种运算，
 但实际上有6种，“-”有两种，“/”有两种，然后我们对这6种运算分别递归即可。最后需要注意的是：
    ①在进行除法运算时，我们要判断分母是否为零；
    ②题目要求除法运算为实数运算，所以在最后判断结果是否为24时，要用<=1e-6判断。
'''


def Search(n):
    if n == 1:
        if int(number[0] - NUMBER_TO_BE_CAL) == 0:
            return True
        else:
            return False
    for i in range(n):
        for j in range(i+1,n):
            a = number[i]
            b = number[j]
            expa = expression[i]
            expb = expression[j]
            number[j] = number[n-1]
            expression[j] = expression[n-1]
            # a+b
            expression[i] =  '(' + expa + '+' + expb + ')'
            number[i] = a + b
            if Search(n-1):
                return True
            # a-b
            expression[i] = '(' + expa + '-' + expb + ')'
            number[i] = a - b
            if Search(n-1):
                return True
            # b-a
            expression[i] = '(' + expb + '-' + expa + ')'
            number[i] = b - a
            if Search(n-1):
                return True
            # a*b
            expression[i] = '(' + expa + '*' + expb + ')'
            number[i] = a * b
            if Search(n-1):
                return True
            # a/b
            if b != 0:
                expression[i] = '(' + expa + '/' + expb + ')'
                number[i] = a / b
                if Search(n-1):
                    return True
            # b/a
            if a != 0:
                expression[i] = '(' + expb + '/' + expa + ')'
                number[i] = b / a
                if Search(n-1):
                    return True
            number[i] = a
            number[j] = j
            expression[i] = expa
            expression[j] = expb
    return False

COUNT_OF_NUMBER = 4
NUMBER_TO_BE_CAL = 24
op = ['(','+','-','*','/',')']
expression = []
number = []
for i in range(4):
    a = input()
    expression.append(a)
    number.append(int(a))
if Search(COUNT_OF_NUMBER):
    print("YES")
    print("%s" % (expression[0]))
else:
    print("NO")
