import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import preprocess
import random
from sklearn.manifold import TSNE


startTime = time.time()

with open('E:/aboutme/STUDY/python-learn/word2vec_skipgram/text8','r') as f:
    text = f.read()
    f.close()
f = open("知识工程作业/作业一skip gram的python实现/skip_gram_tensorflow/skip_gram_tensorflowlog.log",'w') 

# 清洗文本并分词
words = preprocess.preprocess(text)

# 取出unique words
vocab = set(words) 
# 建立索引表
vocab_to_int = {w:c for c,w in enumerate(vocab)} 
int_to_vocab = {c:w for c,w in enumerate(vocab)}
# subsampling
words = preprocess.subsampling(words)
dataset = [vocab_to_int[word] for word in words]
# embedding的维度
num_of_unique_words = len(vocab)
dimension_row = num_of_unique_words
dimension_column = 300


def random_batch(data, batch_size, window_size=5):
    """得到每次随机化处理后的input和label
    
    Arguments:
        data {list} -- 训练集
        batch_size {int} -- 每次训练的大小
        window_size {int} -- 每次的上下文大小
    
    Returns:
        list -- 后续训练的输入
        list -- 标签
    """
    num_batches = len(data) // batch_size
    #data = data[:num_batches * batch_size]
    data = data[:num_batches * batch_size]
    
    # 随机化处理
    random_index = np.random.choice(range(len(data)), len(data), replace=False)

    for i in random_index:
        random_inputs = []
        random_labels = []
        batch = data[i:i+batch_size]
        for idx in range(len(batch)):
            tmp_context = get_context(batch, idx, window_size=window_size)
            random_labels.extend(tmp_context) # context word
            random_inputs.extend([data[idx]] * len(tmp_context)) # target

        yield random_inputs, random_labels


def get_context(words, idx, window_size=5):
    """
    获得input word的上下文单词列表
    ----
    * words: 单词列表
    * idx: input words的索引号
    * window_size: 窗口大小
    ----
    return：
    * context：input word的上下文
    """
    # 这里要考虑input word前面单词不够的情况
    start_idx = idx - window_size if (idx - window_size) > 0 else 0
    end_idx = idx + window_size
    return words[start_idx:idx] + words[idx+1:end_idx+1]


# 参数
batch_size = 100 # 分批处理每批的数量
num_neg_sampled = 10 # 负样本数量


# model
train_graph = tf.Graph()
with train_graph.as_default():
    # 输入以及labels的形状可变，因此shape设置为none
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, shape=[None], name="inputes")
        lables = tf.placeholder(tf.int32, shape=[None, None], name="labels")
    # 这里的embeeding就是我们要的词向量表示
    with tf.name_scope('embedding'):
        embeddings = tf.Variable(tf.random_uniform([dimension_row, dimension_column], -1.0, 1.0))
        selected_embed = tf.nn.embedding_lookup(embeddings, inputs)
    with tf.name_scope('nce_weights'):
        nce_weights = tf.Variable(tf.random_uniform([dimension_row, dimension_column], -1.0, 1.0))
    with tf.name_scope('nce_biases'):
        nce_biases = tf.Variable(tf.zeros([dimension_row]))

    # Loss and optimizer
    # 这里添加了negative sampling，其中num_neg_sampled即为每个词的选择的样例
    with tf.name_scope('loss'):
        loss = tf.nn.nce_loss(nce_weights, nce_biases, lables, selected_embed, num_neg_sampled, dimension_row)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(loss)
    # 这里设置的学习率是0.01
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver() # 保存图

with tf.Session(graph=train_graph) as sess:
    tf.summary.FileWriter("知识工程作业/作业一skip gram的python实现/skip_gram_tensorflow/logs",sess.graph)


# 验证
with train_graph.as_default():
    valid_size = 16 # 待验证的单词集合的大小
    valid_widow = 100

    valid_examples = np.array(random.sample(range(valid_widow), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_widow), valid_size//2))
    valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embedding = embeddings / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))



# Train
with tf.Session(graph=train_graph) as sess:
    init = tf.global_variables_initializer() # 初始化所有的变量
    sess.run(init)
    epoches = 10
    loss = 0
    iteration = 1
    for epoch in range(epoches):
        start_epoche_time = time.time()
        batches = random_batch(dataset, batch_size, window_size=5)
        for batch_inputs,batch_labels in batches:
            _, train_loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, lables: np.array(batch_labels)[:,None]})
            loss += train_loss

            iteration += 1
            end_epoche_time = time.time()
            if iteration % 100 == 0:
                log = 'Iteration: %04d' % iteration
                log_tmp = "cost={:.6f}".format(loss)
                log = '%s\n%s' % (log,log_tmp)
                log_tmp = 'Training loss:{:.4f}'.format(loss/100)
                log = '%s\n%s' % (log, log_tmp)
                log_tmp = '{:.4f} sec/batch'.format((end_epoche_time - start_epoche_time)/100)
                log = '%s\n%s' % (log, log_tmp)
                f.write(log)

                loss = 0
                start_epoche_time = end_epoche_time

            if iteration % 1000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i,:]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        closed_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, closed_word)
                    f.write(log)
                save_path = saver.save(sess, "checkpoints/text8.ckpt")
        

    trained_embeddings = embeddings.eval()
    embed_mat = sess.run(normalized_embedding)

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/checkpoints'))
    embed_mat = sess.run(embeddings)



# visualizing
viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words,:])
fig,ax = plt.subplots(figsize=(14,14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx,:], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx,0], embed_tsne[idx,1]), alpha=0.7)
plt.savefig("知识工程作业/作业一skip gram的python实现/skip_gram_tensorflow/tensorflow.png")
plt.show()

print("trained time:\n", time() - startTime)