from collections import Counter
import random
import numpy as np

def preprocess(text, freq=5):
    '''
    对文本进行预处理,对文本中符号进行替换
    ----
    * text: 文本数据
    * freq: 词频阈值
    ----
    * 
    '''
    text = text.lower()
    text = text.replace('.',' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    word_counts = Counter(words)
    words = [word for word in words if word_counts[word] > freq] # 频数较低的噪音去除
    return words

def subsampling(words):
    """
    subsampling
    ----
    P(w_i) = 1 - sqrt(t/(f(w_i)))
    f(w_i) = number/total
    """
    threshold = 1e-5
    word_counts = Counter(words)
    num_total_words = len(words)
    freqs = {word:count/num_total_words for word, count in word_counts.items()}
    p_drop = {word:1-np.sqrt(threshold/freqs[word]) for word in word_counts}
    train_words = [word for word in words if random.random() < (1-p_drop[word])]
    return train_words