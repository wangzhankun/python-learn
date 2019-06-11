import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import preprocess

startTime = time.time()

with open('E:/aboutme/STUDY/python-learn/word2vec_skipgram/text8','r') as f:
    text = f.read()
    f.close()


# 清洗文本并分词
words = preprocess.preprocess(text)

# 取出unique words
vocab = set(words) 
# 建立索引表
vocab_to_int = {w:c for c,w in enumerate(vocab)} 
int_to_vocab = {c:w for c,w in enumerate(vocab)}
# subsampling
words = preprocess.subsampling(words)

# embedding的维度
num_of_unique_words = len(vocab)
dimension_row = num_of_unique_words
dimension_column = 300


def random_batch(data, size):
    """将data（训练集）进行随机化处理。
    
    Arguments:
        data {list} -- 训练集
        size {int} -- 每次训练的大小
    
    Returns:
        list -- 后续训练的输入
        list -- 标签
    """
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0]) # target
        random_labels.append([data[i][1]]) # context word

    return random_inputs, random_labels


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


def get_standard_dataset(words, tokens, window_size=5):
    """
    将纯文本训练集格式化为TensorFlow可接受的格式。
    ----
    parameter:
    * words: the words from text
    * tokens: the dictionary of the unique words
    * window_size
    ----
    return:
    * skip_grams（list）
    """
    skip_grams = []
    for i in range(1, len(words) - 1):
        target = tokens[words[i]]
        context = get_context(words, target, window_size=window_size)
        for w in context:
            skip_grams.append([target, tokens[w]])
    
    return skip_grams



skip_grams = get_standard_dataset(words, vocab_to_int, window_size=5)

# 参数
batch_size = 50 # 分批处理每批的数量
num_neg_sampled = 10 # 负样本数量


# model
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[batch_size])
    lables = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embeddings = tf.Variable(tf.random_uniform([dimension_row, dimension_column], -1.0, 1.0))
    selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

    nce_weights = tf.Variable(tf.random_uniform([dimension_row, dimension_column], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([dimension_row]))

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, lables, selected_embed, num_neg_sampled, dimension_row))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    saver = tf.train.Saver() # 保存图


# Train
with tf.Session(graph=train_graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epoches = len(words) // batch_size
    for epoch in range(epoches):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, lables: batch_labels})

        if(epoch + 1) % 1000 == 0:
            print('Epoch: %04d' % (epoch + 1), "cost={:.6f}".format(loss))

    trained_embeddings = embeddings.eval()

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(embeddings)



# visualizing
for i, label in enumerate(vocab):
    if i == 500:
        break
    x,y = trained_embeddings[i]
    plt.scatter(x,y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()

print("trained time:\n", time() - startTime)