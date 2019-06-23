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

def load_data(words, debug=True):
    vocab_int = {}
    int_vocab = {}
    if debug:
        with open(save_path+"debug_train_words_dict.txt", "r") as f:
            data = f.read().split()
            for i in range(0, len(data), 2):
                vocab_int[data[i+1]] = int(data[i])
                int_vocab[int(data[i])] = data[i+1]

        dataset = [vocab_int[word] for word in words]

    else:
        with open(save_path+"train_words_dict.txt", "r") as f:
            data = f.read().split()
            for i in range(0, len(data), 2):
                vocab_int[data[i+1]] = int(data[i])
                int_vocab[int(data[i])] = data[i+1]

        dataset = [vocab_int[word] for word in words]

    return vocab_int,int_vocab,dataset

def get_valid_feed(dataset, valid_size=1000, window_size=5):
    """这是用于得到验证数据的函数。旨在训练中计算loss值提供方便
    
    Arguments:
        dataset {list} -- 数据集
    
    Keyword Arguments:
        valid_size {int} -- 待计算的loss的feed的大小 (default: {1000})
    """
    random_idx = np.random.choice(range(len(dataset)), valid_size, replace=False)
    x, y = [], []
    for idx in random_idx:
        # 中心词
        batch_x = dataset[idx]
        # 上下文
        batch_y = get_targets(dataset, idx, window_size)
        # 由于一个中心词会对应多个上下文，因此需要统一长度
        x.extend([batch_x] * len(batch_y))
        y.extend(batch_y)
    return x,y


#%%
def get_targets(dataset, center_word_idx, window_size=5):
    # 参考自网络
    '''
    获得中心词的上下文单词列表
    words: 单词列表
    idx: input words的索引号
    window_size: 窗口大小
    '''
    # 这里要考虑中心词前面单词不够的情况
    start_idx = center_word_idx - window_size if (center_word_idx - window_size) > 0 else 0
    end_idx = center_word_idx + window_size
    targets = dataset[start_idx:center_word_idx] + dataset[center_word_idx+1:end_idx+1]
    return targets


#%%
def get_batches(dataset, batch_size, window_size = 5):
    # 参考自网络
    '''
    构造一个获取batch的生成器
    '''
    # 有多少个小批量，每个批量的单词数是batch_size
    num_batches = len(dataset) // batch_size
    
    # 仅取full batches
    dataset = dataset[:num_batches * batch_size]
    
    for idx in range(0, len(dataset), batch_size):
        x,y = [], []
        batch = dataset[idx:idx+batch_size]
        for i in range(len(batch)):
            # 中心词
            batch_x = batch[i]
            # 上下文
            batch_y = get_targets(batch, i, window_size)
            # 由于一个中心词会对应多个上下文，因此需要统一长度
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x,y


#%%
# 训练
def train(train_graph, dataset, batch_size=100, window_size=5, epoches=100, save_path="final_skip_gram_model"):

    with tf.Session(graph=train_graph) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(save_path+"logs/",sess.graph)
        sess.run(tf.global_variables_initializer())
        # 保存图
        saver = tf.train.Saver(max_to_keep=3)
        

        with open(save_path+"cost.txt", "w") as f:
            for epoch in range(epoches):
                batches = get_batches(dataset, batch_size, window_size) #迭代器
                cost = 0
                
                for x,y in batches:#小批量随机梯度下降
                    feed = {inputs:x, labels:np.array(y)[:,None]}
                    train_cost,_ = sess.run([loss,optimizer], feed_dict = feed)
                    cost += train_cost
                    
                f.write(str(cost) + " " + str(epoch+1) + " ")
                    
                # 每个周期都进行一次保存操作
                if debug == False:
                    x,y = get_valid_feed(dataset)
                    feed = {inputs:x, labels:np.array(y)[:,None]}
                    result = sess.run(merged, feed_dict = feed)
                    writer.add_summary(result, epoch)
                else:
                    x,y = get_valid_feed(dataset)
                    feed = {inputs:x, labels:np.array(y)[:,None]}
                    result = sess.run(merged, feed_dict=feed)
                    writer.add_summary(result, epoch)
                    
                saver.save(sess, save_path + 'model/skip_gram'+str(epoch)+'.ckpt')
                #embed_mat = sess.run(normalized_embedding)


    # # 可视化部分抄自网络
    # viz_words = 500
    # tsne = TSNE()
    # embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

    # fig, ax = plt.subplots(figsize=(14, 14))
    # for idx in range(viz_words):
    #     plt.scatter((*embed_tsne[idx, :]), color='steelblue')
    #     plt.annotate(int_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    # plt.savefig(save_path+'skip_gram.png')
    # plt.show()





#%%
start_time = time.time()
debug = False
isLinux = True
if isLinux:
    save_path = "/data/Final_Commit/"
else:
    save_path = "E:/aboutme/STUDY/python-learn/knowledge_engineering_homework/1_homework_skip_gram/Final_Commit/"

words = load_words(save_path,debug=debug)
vocab_int, int_vocab, dataset = load_data(words,debug=debug)


# embedding 的大小
vocab_size = len(set(words))
embedding_row_dim = vocab_size
embedding_col_dim = 200

num_sampled = 5 # 负例数量
learn_skip = 0.01 # 学习率

#%%
# train_graph = define_graph(tf.Graph())
# print(train_graph)
#train_graph
train_graph = tf.Graph()

with train_graph.as_default():
    # 输入，定义placeholder
    with tf.name_scope("inputs"):
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

    # 嵌入层权重矩阵
    with tf.name_scope('embedding'):
        embedding = tf.Variable(tf.random_uniform([embedding_row_dim, embedding_col_dim]), name='embedding')
        tf.summary.histogram('embedding/embedding', embedding)
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)

    # 运算层
    # 注意name_scope中不能有“：”，否则会报错
    with tf.name_scope('layer_with_learn_skip_' + str(learn_skip) +"_num_sampled" + str(num_sampled) + "_dim_" + str(embedding_col_dim)):
        # 权重矩阵
        with tf.name_scope("weights"):
            Weight = tf.Variable(tf.truncated_normal([embedding_row_dim, embedding_col_dim], stddev = 0.1), name='Weight')
            tf.summary.histogram('embedding/weights', Weight)
        # 偏差。官方推荐不初始化为0    
        with tf.name_scope('biases'):
            biase = tf.Variable(tf.zeros(embedding_row_dim)+0.1, name='biase')
            tf.summary.histogram('embedding/biases', biase)
        # 定义损失函数    
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(Weight, biase, labels, embed, num_sampled, vocab_size))
            tf.summary.scalar('loss', loss)
        # 优化器
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer().minimize(loss)

    
    # 以下是验证部分，抄自网络
    # 随机挑选一些单词
    valid_size = 16
    valid_window = 100
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

            
#%%
if debug == True:
    train(train_graph=train_graph, dataset=dataset[:100*10], batch_size=100, window_size=5, epoches=100,save_path=save_path)
else:
    train(train_graph=train_graph, dataset=dataset, batch_size=100, window_size=5, epoches=30,save_path=save_path)



#%%
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    tmp_embed = sess.run(embedding)
    print(type(tmp_embed))
    num_rows, num_cols = tmp_embed.shape
    with open(save_path+"embedding.txt", "w", encoding="utf-8") as f:
        for i in range(num_rows):
            f.write(int_vocab[i] + " ")
            for j in range(num_cols):
                f.write(str(tmp_embed[i][j]) + " ")
            f.write("\n")
    

with open(save_path + "time.txt", "w") as f:
    f.write(str(time.time() - start_time))

#%%
