#%%
import tensorflow as tf 
import time
import numpy as np 
from nltk.tokenize import WordPunctTokenizer
from collections import Counter

#%%
# load dataset
with open("word2vec_skipgram\\text8") as f:
    text = f.read()
print(type(text))
print(len(text))
text = [word for word in text.split()]
#%%
print(text[:100])
print(len(text))
#%%
# preprocess
tokenizer = WordPunctTokenizer()
words_count = Counter(text)
#删除单词数目少于5的噪声
words = [tokenizer.tokenize(word.lower())  for word in text if words_count[word] > 5]
#%%
print(words[:50])
print(type(words))
print("Total words: {}".format(len(words)))

