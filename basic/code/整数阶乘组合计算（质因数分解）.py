import math
'''
给定正整数n、a，求最大的k，使n！可以被a^k整除但不能被a^(k+1)整除。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬

n和a采用一次input()调用输入，两个数使用逗号(,)分隔，直接输出k值。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
'''
'''
解题思路：
这道题目应当先计算a的质因数，并找到最大质因数m
n!中能分解出几个m，就找到了能整除a^k中k的最大值
'''

def getPrime(lst, num):
    isPrime = True
    i = 2
    square = int(math.sqrt(num)) + 1
    while i <= square:
        if num % i == 0:
            lst.append(i)
            isPrime = False
            getPrime(lst, num / i)
            break
        i += 1
    if isPrime:
        lst.append(num)

def findK(num, n):
    while n % num != 0:
        n -= 1
    j = 0
    while n >= num:
        i = n
        while i % num == 0:
            i = i / num
            j += 1
        n -= num
    return j

n, a = eval(input())
prime_a = []
getPrime(prime_a,a)
prime_a.sort()
m = prime_a[-1]#m为最大质因数
j = 0
while a % m == 0:
    a /= m
    j += 1
out = findK(m,n) / j
print(int(out))