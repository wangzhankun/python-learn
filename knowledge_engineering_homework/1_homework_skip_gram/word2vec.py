import numpy as np
import softmax
import random
import sigmoid

int_len_tokens = 0

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """
    - predicted: 输入词向量
    - target: 目标词向量的索引
    - outputVectors: 输出向量矩阵

    ---
    return
    - cost: 代价
    - gradPred: 输出词向量矩阵的梯度
    - grad: 中心词的梯度
    """
    v_hat = predicted # 中心词向量
    z = np.dot(outputVectors, v_hat) # 预测得分
    y_hat = softmax.softmax(z) # 预测输出y_hat

    cost = -np.log(y_hat[target])

    z = y_hat.copy()
    z[target] -= 1.0
    grad = np.outer(z, v_hat) # 计算中心词的梯度
    gradPred = np.dot(outputVectors.T, z) # 计算输出词向量矩阵的梯度

    return cost, gradPred, grad

def getNegativeSamples(target, dataset, K):
    """ 对K个非目标词向量进行采样"""

    indices = [None] * K
    for k in range(K):
        newidx = sampleTokenIdx(dataset)
        while newidx == target:
            newidx = sampleTokenIdx(dataset)
        indices[k] = newidx
    return indices

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    cost = 0
    z = sigmoid.sigmoid(np.dot(outputVectors[target], predicted))

    cost -= np.log(z)
    grad[target] += predicted * (z - 1.0)
    gradPred += outputVectors[target] * (z - 1.0)

    for k in range(K):
        samp = indices[k + 1]
        z = sigmoid.sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z

    return cost, gradPred, grad

def skipgram(centerWord, contextWords, tokens, dataset, inputVectors, outputVectors, word2vecCostAndGradient=softmaxCostAndGradient):
    """
    return：
    - cost: 代价值
    - granIn: 输入词向量矩阵梯度之和
    - granOut: 输出词向量矩阵梯度之和
    """
    cost = 0.0
    granIn = np.zeros(inputVectors.shape) 
    granOut = np.zeros(outputVectors.shape)

    cword_idx = centerWord
    #cword_idx = tokens[centerWord] # 得到中心词的索引
    v_hat = inputVectors[cword_idx] # 得到中心词的词向量

    # 循环预测上下文中每个单词,这里可以尝试修改成矩阵计算，for循环太慢
    for j in contextWords:
        u_idx = tokens[j] # 目标词的索引
        # 计算一个中心词预测一个上下文词的情况
        c_cost, c_grad_in, c_grad_out = word2vecCostAndGradient(v_hat, u_idx, outputVectors, dataset)
        cost += c_cost # 所有代价求和
        granIn[cword_idx] += c_grad_in # 中心词向量梯度求和
        granOut += c_grad_out # 输出词向量矩阵梯度求和


    return cost, granIn, granOut


def sampleTokenIdx(tokens):
    global int_len_tokens
    return random.randint(0, int_len_tokens-1)

def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表
    words: 单词列表
    idx: input words的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size+1) # 随机生成目标窗口大小
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx+1:end_point+1])
    return list(targets)

def getRandomContext(dataset, num_of_words, target_window=5):
    idx = random.randint(0, num_of_words-1)
    return idx, get_targets(dataset, idx, target_window)

def word2vec_sgd_wrapper(tokens, wordVectors, dataset, window_size = 5, batchsize = 50, word2vecCostAndGradient=softmaxCostAndGradient):
    """
    这里采用了小批量梯度下降的方法。

    输入：
    - tokens: 单词字典索引
    - wordVectors：wordEebedding
    - dataset：数据集
    - window_size: 上下文窗口大小
    - batchsize: 小批量梯度下降中批量大小
    输出：
    - cost: 代价
    - grad: 梯度
    """



    global int_len_tokens
    int_len_tokens = len(tokens)
    cost = 0.0
    grad = np.zeros(wordVectors.shape)

    N = wordVectors.shape[0] # 独特的单词的数量
    inputVectors = wordVectors[:N // 2, :]
    outputVectors = wordVectors[N // 2:, :]

    for i in range(batchsize):
        target_window = random.randint(1, window_size) # 随机生成目标窗口大小
        centerword, context = getRandomContext(dataset, len(tokens), target_window) # 每次随机选择一个单词作为centerword，并返回其上下文

        c, gin, gout = skipgram(centerword, context, tokens, dataset, inputVectors, outputVectors, word2vecCostAndGradient)

        cost += c / batchsize
        grad[:N // 2, :] += gin / batchsize
        grad[N // 2:, :] += gout / batchsize

    return cost, grad