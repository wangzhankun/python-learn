'''
一个由n(n>1)个数字组成的列表 ls，输出一个列表lt，其中lt中第i个元素等于ls中除ls[i]之外所有元素的乘积。‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
'''
def Cal(i,lst_input):
    out = 1
    for j in range(len(lst_input)):
        if i != j:
            out *= lst_input[j]
    return out
lst_input = eval(input())
lst_output = []
for i in range(len(lst_input)):
    lst_output.append(Cal(i,lst_input))
print(lst_output)