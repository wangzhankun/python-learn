#%%[markdown]
# # numpy基础篇
# 以下内容来自https://www.numpy.org.cn/article/basics/understanding_numpy.html
#%%
import numpy as np

a = np.array([1.1,2.2,3.3], dtype=np.float64)
a, a.dtype

#%%
print(a.shape)
print(a[0])
print(a[1])

#%%
a[0] = -1
print(a)

#%%
random_array = np.random.random((5))
print(random_array)

#%%
# 创建二维数组
my_2d_array = np.zeros((2,3))
print(my_2d_array)
print(my_2d_array[1][2])

#%%
my_array = np.array([[4,3],[5,6]])
print(my_array)

#%%
# 提取第2列的所有元素
print(my_array[:, 1])
# 提取第1行所有元素
print(my_array[0])

#%% [markdown]
# ## 数组操作
# 使用NumPy，你可以轻松地在数组上执行数学运算。例如，你可以添加NumPy数组，
# 你可以减去它们，你可以将它们相乘，甚至可以将它们分开。 以下是一些例子：
#%%
a = np.array([[1.,2.],[3.,4.]])
b = np.array([[5.,6.],[7.,8.]])
sumab = a + b
difference = a - b
product = a * b
quotient = a / b
#%%
print("sum = \n", sumab)
print("difference = \n", difference)
print("product = \n", product)
print("quotient = \n", quotient)
#%%
matrix_product1 = a.dot(b)
matrix_product2 = b.dot(a)
print(matrix_product1)
print(matrix_product2)
