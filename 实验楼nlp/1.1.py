#%% [markdown]
## 字符串操作
### count()返回特定子串在字符串中出现的次数
# In[]
seq = '12345,1234,123,12,1'
seql = '1'
a = seq.count(seql)
a

#%% [markdown]
### strip()去除字符串首尾的指定符号。无指定时，默认去除空格符和换行符

# In[]
seq = '我们正在使用实验楼，实验楼学到很多！'
seq.strip()
seq.strip('！')
seq.strip('我们')

#### lstrip()去除字符串左边指定的符号
#### rstrip()去除字符串右边指定的符号

# In[]
seq = '12321'
print(seq.lstrip('1'))
print(seq.strip('1'))
print(seq.rstrip('1'))

### 字符串拼接
#### '+'直接拼接
#%%
seq1 = '实'
seq2 = '验'
seq3 = '楼'
seq = seq1 + seq2 + seq3
print(seq)

### 字符串拼接

#### join()方法
#%%
seq = ['2018', '10', '31']
seq = '-'.join(seq)
print(seq)

### 字符串比较
#### 这里需要加载operator工具
#%%
import operator
seq1 = '字符串1号'
seq2 = '字符串2号'
print(operator.gt(seq1,seq2))

##### 直接使用运算符
#%%
print(seq1)
print(seq2)
print(seq1 < seq2)

### 大小写转换
#### upper(),lower()
#%%
seq = 'appLE'
print(seq.upper())
print(seq.lower())

### 查找字符串
#### find未找到则返回-1
#%%
seq = '这是一段字符串'
seq1 = '字符串'
print(seq.find(seq1))
print(seq.find('无'))

### 字符串截取
#%%
seq = '这是字符串'
seq1 = seq[0:4]
print(seq1)
print(seq[0])
print(seq[2:4])


### 字符串切分
#%%
seq = '今天天气很好，我们出去玩'
print(seq.split('，'))
seq = '2018-11-11'
print(seq.split('-'))
seq = 'I have an apple'
print(seq.split(' '))

### 字符串翻转
#%%
seq = '12345'
print(seq[::-1])


### 字符串代替
#%%
seq = '2018-11-11'
print(seq.replace('-','/'))

### 以某字符串开头结尾
#%%
seq = 'abcdefg'
print(seq.startswith('a'))
print(seq.endswith('f'))



## 正则表达式
#### 不规则日期年份的提取
#%%
import re
# 连续四个字符，每个字符是0-9
pattern = re.compile(r'[0-9]{4}')
time = '2018-01-01'
match = pattern.search(time)
match.group()

#%%
import re
pattern = re.compile(r'[0-9]{4}')
times = ('2018-01-01', '01/01/2019', '01.2017.01')
for time in times:
    match = pattern.search(time)
    if match:
        print('年份有：', match.group())

### findall
#%%
import re
# 识别数字
pattern = re.compile(r'\d')
print(pattern.findall('o1n2m3k4'))
# 识别非数字
pattern = re.compile(r'\D')
print(pattern.findall('o1j2k3n4'))

### match
# match 与 search功能一样，但是只匹配一次,并且从开头开始匹配
#%%
import re
pattern = re.compile('c')
print(pattern.match('comcdc').group())
pattern = re.compile('1')
# 无法匹配1，因为match从开头匹配
print(pattern.match('abcdefg1').group())