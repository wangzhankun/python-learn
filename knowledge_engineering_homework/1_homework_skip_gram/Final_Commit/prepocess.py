#%%
import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
#%%
def prepocess(text, freq=5):
    """输入文本并对文本进行预处理。处理包括：将文本句子转为单个的按原有文本顺序排列的列表，
    同时去除频数低于5的噪声，此外对高频词汇进行subsampling处理
    
    Arguments:
        text {str} -- 输入文本
        freq {int} -- 频数上线
    
    Returns:
        list -- 训练集
    """
    # 使用nltk便于处理
    tokenizer = WordPunctTokenizer()
    words = tokenizer.tokenize(text)
    words_count = Counter(words)  # 每个单词的对应的数量
    words = [word.lower() for word in words if words_count[word] > freq]

    # 高频抽样代码参考自网络
    # subsampling
    t = 1e-5
    threshold = 0.8 # 剔除概率阈值
    # 统计单词出现频次,这里必须重新计算
    words_count = Counter(words) # 每个单词的对应的数量
    total_count = len(words) # 所有单词的总数
    # 计算频次
    word_freqs = {word:count/total_count for word,count in words_count.items()}
    # 计算被删除的概率
    delete_probability = {word: 1-np.sqrt(t / word_freqs[word]) for word in words_count}
    # 对单词进行采样
    train_words  = [word for word in words if delete_probability[word] < threshold]
    
    return train_words

#%%
debug = False
isLinux = False
if isLinux:
    save_path = "/data/"
else:
    save_path = "E:/aboutme/STUDY/python-learn/knowledge_engineering_homework/1_homework_skip_gram/Final_Commit/"

#%%
if isLinux:
    with open("/data/text8", "r") as f:
        text = f.read()
else:
    with open("G:/Deep-Learning-master/word2vec_skipgram/text8", "r") as f:
        text = f.read()
#%%
if debug == True:
    text = text[:1000*1000]
    words = prepocess(text, freq=0)
else:
    words = prepocess(text,freq=5)

# 构建映射表
vocabulary = set(words) # 独特词汇表
vocab_int = {word:num for num,word in enumerate(vocabulary)}
int_vocab = {num:word for num,word in enumerate(vocabulary)}
dataset = [vocab_int[word] for word in words]



#%%
if debug:
    with open(save_path+"debug_train_words_dict.txt", "w") as f:
        for num,word in enumerate(vocabulary):
            f.write(str(num) + " " + word + " ")
    with open(save_path+"debug_train_words.txt", "w") as f:
        for i in range(len(words)):
            f.write(words[i]+" ")
else:
    with open(save_path+"train_words_dict.txt", "w") as f:
        for num,word in enumerate(vocabulary):
            f.write(str(num) + " " + word + " ")
    with open(save_path+"train_words.txt", "w") as f:
        for i in range(len(words)):
            f.write(words[i]+" ")



#%%
# if debug:
#     with open(save_path+"debug_train_words.txt", "r") as f:
#         data = f.read().split()
#         print(len(data))
#         for i in range(0, len(data), 2):
#             print(data[i], data[i+1])


#%%
# 构建映射表
# vocabulary = set(words) # 独特词汇表
# vocab_int = {}
# int_vocab = {}

# if debug:
#     with open(save_path+"debug_train_words.txt", "r") as f:
#         data = f.read().split()
#         for i in range(0, len(data), 2):
#            vocab_int[data[i+1]] = data[i]
#            int_vocab[data[i]] = data[i+1]

#     dataset = [vocab_int[word] for word in words]

# else:
#     with open(save_path+"train_words.txt", "r") as f:
#         data = f.read().split()
#         for i in range(0, len(data), 2):
#            vocab_int[data[i+1]] = data[i]
#            int_vocab[data[i]] = data[i+1]

#     dataset = [vocab_int[word] for word in words]
#%%
if debug:
    with open(save_path+"debug_train_words_dict.txt", "r") as f:
        data = f.read().split()
        for i in range(0, len(data), 2):
            vocab_int[data[i+1]] = data[i]
            int_vocab[int(data[i])] = data[i+1]

    dataset = [vocab_int[word] for word in words]

for i in range(1,100):
    print(int_vocab[i])

#%%
