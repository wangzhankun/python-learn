lst = []
lst_prime = [2,3,5,7]
lst_out = []
for i in range(2,101):
    lst.append(i)
while lst:
    flag = 1
    x = lst.pop()
    for i in lst_prime:
        if x % i == 0 and x not in lst_prime:
            flag = 0
    if flag:
        lst_out.append(x)
lst_out.sort()
for i in lst_out:
    print("{}".format(i),end = " ")
print('\n')
