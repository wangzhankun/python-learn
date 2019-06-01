#%% [markdown]
# # NumPy简单入门教程
# NumPy是Python中的一个运算速度非常快的一个数学库，它非常重视数组。
# 它允许你在Python中进行向量和矩阵计算，并且由于许多底层函数实际上是用C编写的，
# 因此你可以体验在原生Python中永远无法体验到的速度。
#%%
import numpy as np
#%% [markdown]
# ## 数组基础
# ### 一维数组
#%%
# 一维数组
a = np.array([0,1,2,3,4])
b = np.array((0,1,2,3,4,))
c = np.arange(5)
d = np.linspace(0, 2, num=5, endpoint=True)

print(a)
print(b)
print(c)
print(d)
#%% [markdown]
# 上面的代码显示了创建数组的4种不同方法。最基本的方法是将序列传递给
# NumPy的array()函数; 你可以传递任何序列（类数组），
# 而不仅仅是常见的列表（list）数据类型。
# ### 多维数组
#%%
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
#%%
two_dim_array = np.array([[11,12,13,14,15],
                [16,17,18,19,20],
                [21,22,23,24,25]])
print(two_dim_array[1:,4])
print(two_dim_array)

#%% [markdown]
# ## 多维数组切片
# 关于二维数组其实就可以对其这样理解，先对array进行切片取list，再对list进行切片
# 多维数组可以类似理解
#%%
print(a)
print(a[0,1:4])
print(a[1:4, 0])
print(a[::2,::2])

#%% [markdown]
# ## 数组属性
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a))
print(a.dtype)
print(a.size) # 数组中一共有多少个元素
print(a.shape)
print(a.itemsize) # 每个item占用多少字节
print(a.ndim) # 维数， 2维
print(a.nbytes) # 一共占用了多少字节

#%% [markdown]
# ## 基本操作符
a = np.arange(25)
a = a.reshape(5,5)
b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78]).reshape((5,5))

print(a)
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a.dot(b))
print(b.dot(a))
print(a > b)
print(a < b)

#%% [markdown]
# 除了 dot() 之外，这些操作符都是对数组进行逐元素运算。
# 比如 (a, b, c) + (d, e, f) 的结果就是 (a+d, b+e, c+f)。
# 它将分别对每一个元素进行配对，然后对它们进行运算。
# 它返回的结果是一个数组。注意，当使用逻辑运算符比如 “<” 和 “>” 的时候，
# 返回的将是一个布尔型数组，这点有一个很好的用处，后边我们会提到。
# dot() 函数计算两个数组的点积。它返回的是一个标量
# （只有大小没有方向的一个值）而不是数组。

#%% [markdown]
# ## 数组特殊运算符
#%%
a = np.arange(10)
print(a)
print(a.sum())
print(a.min())
print(a.max())
print(a.cumsum())
#%% [markdown]
# sum()、min()和max()函数的作用非常明显。将所有元素相加，找出最小和最大元素。    <br/>
# cumsum(a, axis=None, dtype=None, out=None)  <br/>
# a.cumsum(axis=None, dtype=None, out=None)    <br/>
# 返回：沿着指定轴的元素累加和所组成的数组，其形状应与输入数组a一致    <br/>
# 其中cumsum函数参数：    <br/>
# - a:数组
# - axis：轴索引，整型， 若a为n维数组，则axis的取值范围为[0,n-1]
# - dtype: 指定返回结果的数据类型
# - out: 数据类型为数组。用来放置结果的替代输出数组，它必须具有与输出结果具有相同的形状和缓冲长度
#%%
# 一维数组
arr = np.arange(1,10)
result = arr.cumsum() # 此时，axis只能取0
print(result)
#%%
# 二维数组
arr = np.arange(1,10).reshape((3,3))
print(arr)
result1 = arr.cumsum(0)
result2 = arr.cumsum(1)
result3 = arr.cumsum()
print(result1)
print(result2)
print(result3)
#%%
# 三维数组
arr = np.arange(1,9).reshape((2,2,2))
print(arr)
result0 = arr.cumsum()
result1 = arr.cumsum(0)
result2 = arr.cumsum(1)
result3 = arr.cumsum(2)
print(result0)
print(result1)
print(result2)
print(result3)

#%% [markdown]
# ## 索引进阶
# ### 花式索引
# 花式索引 是获取数组中我们想要的特定元素的有效方法。
#%%
a = np.arange(0,100,10)
indices = [1, 5,-1, 4]
b = a[indices]
print(a)
print(b)

#%% [markdown]
# ## 布尔屏蔽
# 布尔屏蔽是一个有用的功能，它允许我们根据我们指定的条件检索数组中的元素。
#%%
import matplotlib.pyplot as plt
a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a, b)
mask = b >= 0
print(mask)
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()

#%% [markdown]
# ## 缺省索引
# 不完全索引是从多维数组的第一个维度获取索引或切片的一种方便方法。
# 例如，如果数组a=[1，2，3，4，5]，[6，7，8，9，10]，
# 那么[3]将在数组的第一个维度中给出索引为3的元素，这里是值4。
#%%
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b)
print(c)

#%% [markdown]
# ## where函数
# where() 函数是另外一个根据条件返回数组中的值的有效方法。
# 只需要把条件传递给它，它就会返回一个使得条件为真的元素的列表。    <br/>
# numpy.where(condition[,x,y]) function returns the indices of elements in an input array
# where the given condition is satisfied    <br/>
# - condition: where True, yield x, otherwise yield y
# - x,y: Values from which to choose. x,y and condition need to be broadcastable to some shape.
# - out: [ndarray or tuple of ndarrays] if both x and y are specified, the output array
# contains elements of x where condition is True, and elements from y elsewhere.

#%%
a = np.arange(0, 100, 10)
b = np.where(a < 50)
c = np.where(a >= 50)[0]
print(a)
print(b)
print(c)
#%%
np.where([[True, False], [True, True]],
            [[1,2],[3,4]],
            [[5,6],[7,8]])
# array([[1, 6],
#       [3, 4]])
#%% [markdown]
# 上面这个例子的条件为[[True,False], [True,False]]，分别对应最后输出结果的四个值。
# 第一个值从[1,9]中选，因为条件为True，所以是选1。第二个值从[2,8]中选，
# 因为条件为False，所以选8，后面以此类推。类似的问题可以再看个例子：
#%%
a = 10
np.where([[a>5,a<5], [a==10, a==7]],
            [["chosen","not chosen"],["chosen", "not chosen"]],
            [["not chosen", "chosen"],["not chosen", "chosen"]])
# array([['chosen', 'chosen'],
#       ['chosen', 'chosen']], dtype='<U10')

#%% [markdown]
# np.where(condition)    <br/>
# 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 
# (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，
# 输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
#%%
a = np.array([2,4,6,8,10])
np.where(a > 5)
#%%
a[np.where(a > 5)]
#%%
np.where([[0,1],[1,0]])
# (array([0, 1], dtype=int64), array([1, 0], dtype=int64))
# [[0,1],[1,0]]的真值为两个1，各自的第一维坐标为[0,1]，第二维坐标为[1,0]
#%%
a = np.array([[1,2,3], [4,5,6]])
print(a)
print('Indices of elements <4')
b = np.where(a<4)
print(b)
print('Elements which are <4')
print(a[b])

#%%
a = np.arange(10)
np.where(a,1,-1)
# array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
# 由于0为false，故而第一个输出-1

#%%
a = np.arange(10)
np.where(a>5, 1, -1)
# array([-1, -1, -1, -1, -1, -1,  1,  1,  1,  1])
# 满足条件输出1，否则输出-1
#%%
# 数值类型转换
b = a.astype(int)
print(a.dtype)
print(b)
print(type(b))

#%%
# 快速创建数值全为1的多维数组
np.ones((5,6), dtype=np.int64)

#%%
# 快速创建数值全为0的多维数组
np.zeros((3,4))

#%%
np.eye(5, 4, 2)

#%% [markdown]
# 从已知数据文件、函数创建ndarray
# - frombuffer (buffer): 将缓冲区转换为1维数组
# - fromfile (file, dtype, count, sep): 从文本或二进制文件中构建多维数组
# - fromfunction (function, shape): 通过函数返回值来创建多维数组
# - fromiter (iterable, dtype, count): 从可迭代对象创建1维数组

#%%
np.fromfunction(lambda a,b: a+b, (5,4))

#%%
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a
#%%
a.T

#%%
a.dtype

#%%
# 虚部部分
a.imag

#%%
# 实部部分
a.real

#%%
a.size

#%%
a.shape

#%%
# 一个元素的字节数
a.itemsize

#%%
# a的元素的总字节数
a.nbytes

#%%
# 输出数组尺寸
a.ndim

#%%
# 用来遍历数组时，输出每个维度中步进的字节数组。
a.strides

#%%
# ndarray多维数组
one = np.array([7,2,9,10])
two = np.array([[5.2,3.0,4.5],
                [9.1, 0.1, 0.3]])
three = np.array([[[1, 1], [1, 1], [1, 1]],
                  [[1, 1], [1, 1], [1, 1]],
                  [[1, 1], [1, 1], [1, 1]],
                  [[1, 1], [1, 1], [1, 1]]])

#%%
one.shape, two.shape, three.shape

#%%
# 重设形状
print(a.shape)
np.reshape(a,(9,1))
#%%
np.arange(10).reshape((2,5))

#%%
# 数组展开
# ravel的目的是将任意形状的数组的扁平化，变为1维数组
print(a)
print(np.ravel(a, order='C'))
print(np.ravel(a, order='F'))

#%%
# 轴移动到新的位置
# np.moveaxis(a, source, destination)
# - a数组
# - source 要移动的轴的原始位置
# - destination 要移动的轴的目标位置
a = np.ones((1, 2, 3))
a
#%%
a.shape, np.moveaxis(a, 0, -1).shape

#%%
print(a)
print(np.moveaxis(a,0,-1))

#%%
