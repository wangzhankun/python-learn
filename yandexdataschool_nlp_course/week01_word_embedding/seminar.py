#%%
import numpy as np
data = list(open('yandexdataschool_nlp_course\week01_word_embedding\quora.txt',encoding='utf-8'))
data[50]

#%% [markdown]
# 使用nltk进行处理文本。因为文本里面含有大量特殊符号标点，引用nltk会使得处理变得简单

#%%
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer() #实例化对象
print(tokenizer.tokenize(data[50]))

#%%
# lowercase everything and extract tokens with tokenizer
# data_tok should be a list of lists of tokens for each line in data
data_tok = [tokenizer.tokenize(i.lower()) for i in data]
print(data_tok)

#%%
#assert all(isinstance(row, (list, tuple)) for row in data_tok), "please convert each line into a list of tokens (strings)"