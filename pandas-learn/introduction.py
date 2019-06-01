#%%
# https://www.shiyanlou.com/courses/?fee=free&page_size=20&category=%E5%85%A8%E9%83%A8&tag=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0&sort=default&preview=false
import pandas as pd 
#%%
print(pd.__version__)

#%%
arr = [0,1,2,3,4]
s1 = pd.Series(arr) # 不指定索引，默认从0开始
s1

#%%
# 从ndarray创建series
import numpy as np 
n = np.random.randn(5)
index = ['a','b','c','d','e']
s2 = pd.Series(n,index=index)
s2

#%%
# 从字典创建series
d = {'a':1,'b':2,'c':3}
s3 = pd.Series(d)
s3

#%%
# series 纵向拼接
s4 = s3.append(s1)
s4

#%%
# 按照指定索引删除元素
print(s4)
s4 = s4.drop('a')
print(s4)

#%%
#修改指定索引元素,这也可以当做添加元素
s4['A'] = 6
s4

#%%
# 根据索引进行相加，如果索引不同则设置为nan
# 加法
print(s4)
print(s3)
print(s4.add(s3))

#%%
# 减法
print(s4)
print(s4.sub(s3))

#%%
# 乘法
print(s4.mul(s3))

# 除法
print(s4.div(s3))

#%%
# 求中位数
print(s4.median())
#求和
print(s4.sum())
# 求最大值
print(s4.max())
# 求最小值
print(s4.min())

#%% 
# dataframe
dates = pd.date_range('today',periods=6) # 定义时间序列作为index
num_arr = np.random.randn(6,4)
print(dates)
print(num_arr)
columns = ['A','B','C','D']
df1 = pd.DataFrame(num_arr,index=dates,columns=columns)
print(df1)

#%%
# 通过字典创建dataframe
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(data, index=labels)
print(df2)
print(df2.dtypes)
print(df2.describe())
df2

#%%
# 预览dataframe的前5行
df2.head()
# 后3行
df2.tail(3)

#%%
# 查看索引
df2.index
#%%
# 查看列明
df2.columns
#%%
# 查看数值
df2.values
#%%
df2.describe()

#%%
# 转置
df2.T

#%%
df2.sort_values(by='age')

#%%
df2[1:3]
#%%
df2['age']
#%%
df2[['age','animal']]
#%%
# 通过位置查询
df2.iloc[1:3] # 查询2,3行
#%%
# 拷贝副本
df3 = df2.copy()
df3
#%%
df3.isnull()
#%%
num = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=df3.index)
# 添加新的一列
df3['No'] = num
df3
#%%
# 修改第2行与第1列对应的值3.0 → 2.0
df3.iat[1,0] = 2
df3
#%%
df3.loc['f','age'] = 1.5
df3
#%%
# 添加一列值，命名为(f,age)
df3['f','age'] = 1.5
df3
#%%
df3.mean()
#%%
df3['visits'].sum()

#%%
# 字符串操作
string = pd.Series(['A','B','C','AaBb','Baca',np.nan,'CABA','dog','cat'])
print(string)
string.str.lower()
#%%
string.str.upper()

#%%
# 对缺失值进行填充
df4 = df3.copy()
print(df4)
df4.fillna(value=3)
#%%
# 对缺失值的行进行删除
df5 = df3.copy()
print(df5)
df5.dropna(how='any') # 任何存在nan的行都将删除

#%%
left = pd.DataFrame({'key':['foo1','foo2'],'one':[1,2]})
right = pd.DataFrame({'key':['foo2','foo3'],'two':[4,5]})

print(left)
print(right)

#%%
pd.merge(left,right,on='key')

#%%
df3.to_csv('animal.csv')
#%%
df_animal = pd.read_csv('animal.csv')
df_animal
#%%
df3.to_excel('animal.xlsx',sheet_name='sheet1')

#%%
pd.read_excel('animal.xlsx','sheet1',index_col=None,na_values=['NA'])
