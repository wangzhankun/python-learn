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
def load_words(load_path, debug=True):
    if debug:
        with open(load_path+"debug_train_words.txt", "r") as f:
                words = f.read().split()
    else:
        with open(load_path+"train_words.txt", "r") as f:
                words = f.read().split()
    return words

def load_data(words, load_path, debug=True):
    vocab_int = {}
    int_vocab = {}
    if debug:
        with open(load_path+"debug_train_words_dict.txt", "r") as f:
            data = f.read().split()
            for i in range(0, len(data), 2):
                vocab_int[data[i+1]] = int(data[i])
                int_vocab[int(data[i])] = data[i+1]

        dataset = [vocab_int[word] for word in words]

    else:
        with open(load_path+"train_words_dict.txt", "r") as f:
            data = f.read().split()
            for i in range(0, len(data), 2):
                vocab_int[data[i+1]] = int(data[i])
                int_vocab[int(data[i])] = data[i+1]

        dataset = [vocab_int[word] for word in words]

    return vocab_int,int_vocab,dataset

#%%
with tf.Session() as sess:
    saver.restore(sess, path + "model/skip_gram29.ckpt")

    embedding = sess.run("embedding/embedding:0")
    sess.run(tf.global_variables_initializer())
    tmp_embed = embedding
    print(type(tmp_embed))
    num_rows, num_cols = tmp_embed.shape
    with open(path+"embedding.txt", "w", encoding="utf-8") as f:
        for i in range(num_rows):
            f.write(int_vocab[i] + " ")
            for j in range(num_cols):
                f.write(str(tmp_embed[i][j]) + " ")
            f.write("\n")

    # 以下是验证部分，抄自网络
    # 随机挑选一些单词
    valid_size = 16
    valid_window = 1000
    # 从不同位置各选8各单词
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples, 
                            random.sample(range(1000, 1000+valid_window), valid_size//2))
    
    valid_size = len(valid_examples)
    # 验证单词集
    with tf.name_scope('constant'):
        valid_dataset = tf.constant(valid_examples, dtype = tf.int32, name='valid_dataset')
    
    # 计算每个单词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    sim = similarity.eval()
    for i in range(valid_size):
        valid_word = int_vocab[valid_examples[i]]
        top_k = 8
        nearest = (-sim[i,:]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            closed_word = int_vocab[nearest[k]]
            log = '%s %s, ' % (log, closed_word)
        print(log)

    embed_mat = sess.run(normalized_embedding)
    

#%%
debug = False
isLinux = False
if isLinux:
    path = "/home/wang/skip_gram/"
else:
    path = "G:/Download/Final_Commit/"
    
saver = tf.train.import_meta_graph(path + "model/skip_gram29.ckpt.meta")
words = load_words(path,debug=debug)
vocab_int, int_vocab, dataset = load_data(words,path, debug=debug)

#%%
viz_words = 100
tsne = TSNE()
#embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])


#%%
#可视化部分抄写自网络
fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter((*embed_tsne[idx, :]), color='steelblue')
    plt.annotate(int_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
plt.savefig(path+'skip_gram.png')
plt.show()

#%%
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    tmp_embed = sess.run(embedding)
    print(type(tmp_embed))
    num_rows, num_cols = tmp_embed.shape
    with open(path+"embedding.txt", "w", encoding="utf-8") as f:
        for i in range(num_rows):
            f.write(int_vocab[i] + " ")
            for j in range(num_cols):
                f.write(str(tmp_embed[i][j]) + " ")
            f.write("\n")
    

with open(path + "time.txt", "w") as f:
    f.write(str(time.time() - start_time))

#%%
