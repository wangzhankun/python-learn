import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import preprocess
from word2vec import *
from sgd import *

random.seed(0)
np.random.seed(0)
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

# embedding的维度
num_of_unique_words = len(vocab)
dimension_row = num_of_unique_words
dimension_column = 300

# 包含input Vectors 以及 output Vectors
wordEmbedding = np.concatenate(
    ((np.random.rand(dimension_row, dimension_column) - 0.5) /
       dimension_column, np.zeros((dimension_row, dimension_column))),
    axis=0)


wordEmbedding = sgd(
    lambda vec: word2vec_sgd_wrapper(vocab_to_int, vec, words, window_size=5,batchsize=50,
    word2vecCostAndGradient=negSamplingCostAndGradient), wordEmbedding, step=0.3, iterations=4000, postprocessing=None, useSaved=True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordEmbedding[:num_of_unique_words,:], wordEmbedding[num_of_unique_words:,:]),
    axis=0)

with open("word_embedding_output", 'w') as f:
    f.write(wordVectors)
    f.close()


visualizeWords = [
    "the", "a", "an",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

visualizeIdx = [vocab_to_int[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
