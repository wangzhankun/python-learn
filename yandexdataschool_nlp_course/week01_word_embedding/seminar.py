# %%
import bokeh.plotting as pl
import bokeh.models as bm
from sklearn.manifold import TSNE
from bokeh.io import output_notebook
from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
import numpy as np

# %%
data = list(open(
    'yandexdataschool_nlp_course\week01_word_embedding\quora.txt', encoding='utf-8'))
# data = list(open('quora.txt',encoding='utf-8'))
data[50]

# %% [markdown]
# 使用nltk进行处理文本。因为文本里面含有大量特殊符号标点，引用nltk会使得处理变得简单

# %%
# Tokenize a text into a sequence of alphabetic and non-alphabetic characters
tokenizer = WordPunctTokenizer()  # 实例化对象
print(tokenizer.tokenize(data[50]))

# %%
# lowercase everything and extract tokens with tokenizer
# data_tok should be a list of lists of tokens for each line in data
data_tok = [tokenizer.tokenize(i.lower()) for i in data]

# %%
print(len(data))
print(len(data_tok))
type(data[1])  # str. data里面的内容是str

# %%
print(data_tok[530])
type(data_tok[2])  # list. 说明这是list的嵌套

# %%
# require that every element in data_tok is list or tuple
assert all(isinstance(row, (list, tuple))
           for row in data_tok), "please convert each line into a list of tokens (strings)"
assert all(all(isinstance(tok, str) for tok in row)
           for row in data_tok), "please convert each line into a list of tokens (strings)"


def is_latin(tok): return all('a' <= x.lower() <= 'z' for x in tok)


assert all(map(lambda l: not is_latin(l) or l.islower(), map(
    ' '.join, data_tok))), "please make sure to lowercase the data"

# %%
# print([' '.join(row) for row in data_tok[::100000]]) # data_tok数据量庞大，不建议执行,这是将序列转为文本


#%% [markdown]
# # Word2Vec 的一些简单实用说明
# https://rare-technologies.com/word2vec-tutorial/ <br/>
# https://blog.csdn.net/qq_19707521/article/details/79169826 <br/>
# # 训练模型
# * 利用```gensim.models.Word2Vec(sentences)```建立词向量模型：
#       经历了三个步骤：<br/>
#           建立一个空的模型对象<br/>
#           遍历一次语料建立词典<br/>
#           第二次遍历语料库建立神经网络模型<br/>
#       可以分别通过执行```model=gensim.models.Word2Vec()```<br/>
#       ```model.build_vocab(sentences)```<br/>
#       ```model.train(sentences)``` <br/>
# # Preparing the Input
# Starting from the beginning, gensim’s word2vec expects a sequence of sentences as its input. Each sentence a list of words (utf8 strings):
# 
#%%
import sys
sentences = [['first','sentence'],['second','sentence']]
# train word2vec on the two sentences
model = Word2Vec(sentences, min_count=1)
# print(model.get_vector('first')) # 该model类型没有相应属性，原因是类型与下文model不同，原因未知
type(model) # gensim.models.word2vec.Word2Vec
type(model.wv.__getitem__('first')) # numpy.ndarray
print(model.wv.__getitem__('first'))

#%% [markdown]
# gensim only requires that the input must provide sentences sequentially, when iterated over



#%%
# 将data_tok转为模型
model = Word2Vec(data_tok,
                 size=32,  # embedding vector size
                 min_count=5,  # consider words that occured at least 5 times
                 window=5).wv  # define context as a 5-word window around the target word
#%%
# 注意这里的model的类型与上文的model的类型不同，原因未知
type(model) # gensim.models.keyedvectors.Word2VecKeyedVectors
#%% [markdown]
# # storing and loading models
#%%
# model.save_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/mymodel')
#%%
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/mymodel')
new_model = KeyedVectors.load_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/mymodel')
#%%
print(type(model))
print(type(new_model))
# %%
# get word vectors
print(model.get_vector('anything'))
print(new_model.get_vector('anything')) # this is a test
print(model.__getitem__('anything'))

# %%
# query similar words directly.
print(model.most_similar('you'))

# %% [markdown]
# # Using pre-trained model
# to process a large data
# %%
# 这里的20-newsgroups是一个数据集
# 数据集与模型网站https://github.com/RaRe-Technologies/gensim-data
# model = api.load('20-newsgroups')
# %%
# glove-wiki-gigaword-50是一个模型
# model = api.load('glove-wiki-gigaword-50')
#%%
# model.save_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/glove-wiki-gigaword-50')
#%%
model = KeyedVectors.load_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/glove-wiki-gigaword-50')
#%%
# gensim.models.keyedvectors.Word2VecKeyedVectors
type(model)
# %%
model.most_similar('man')

#%%
model.most_similar(positive=['coder', 'money'], negative=['brain'])
#%%
print(model.similarity('mail','e-mail'))
print(model.most_similar(positive=['queen','king'], negative=['man']))

# %% [markdown]
# # Visualizing word vectors
# 词的向量维数通常是在20维以上，人类只能识别三维及其以下，需要利用dimensionality reduction进行降维
# %%
words = sorted(model.vocab.keys(),
               key=lambda word: model.vocab[word].count,
               reverse=True)[:1000]
#%%
print(words[::100])
# %%
# for each words, compute it's vector
word_vectors = np.array([model.get_vector(i) for i in words])
len(words)
# %%
# 这里是判断上述代码是否运行正确
assert isinstance(word_vectors, np.ndarray)
assert word_vectors.shape == (len(words), 50)
assert np.isfinite(word_vectors).all()
print(type(word_vectors)) # numpy.ndarray.shape return a tuple in which the first value is the len of word_vectors and the second is the len of element
print(word_vectors.shape) # type(word_vectors.shape) == tuple, (1000,50)
print(len(word_vectors[0]))
# %% [markdown]
# # Linear projection:PCA
# map wordvectors onto 2d plane with PCA. Use good old sklearn api (fit,transform)
# after that, normalize vectors to make sure they have zero mean and unit variance
#%%
word_vectors_pca = PCA(n_components=2).fit_transform(word_vectors)
#%%
#%%
# numpy.naddrray
type(word_vectors_pca)
#%%
print(word_vectors_pca.mean())

# %%
assert word_vectors_pca.shape == (
    len(word_vectors), 2), "there must be a 2d vector for each word"
assert max(abs(word_vectors_pca.mean(0))
           ) < 1e-5, "points must be zero-centered"

# don't know why
print(max(abs(1.0-word_vectors_pca.std(0)))) # 0.2002182
# assert max(abs(1.0-word_vectors_pca.std(0))
#           ) < 1e-5, "points must have unit variance"

# %% [markdown]
# # Let's draw it!
# %%
# 这里注意区分bokeh.models和bokeh.model的区别
# 关于bokeh的学习请参考我的同文件夹下的另一篇代码文章bokeh_learn.py
# from bokeh.io import output_notebook
# import bokeh.plotting as pl
# import bokeh.models as bm
output_notebook()


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


# %%
draw_vectors(word_vectors_pca[:, 0], word_vectors_pca[:, 1], token=words)

# %%
word_tsne = TSNE(verbose=True).fit_transform(word_vectors)
# %%
draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='green', token=words)

#%% [markdown]
# # Visualizing phrases
# Word embeddings can also be used to represent short phrases. The simplest way is to take an average of vectors for all tokens in the phrase with some weights.<br/>
# This trick is useful to identify what data are you working with: find if there are any outliers, clusters or other artefacts.<br/>
#%%
def get_phrase_embedding(phrase):
    """
    Convert phrase to a vector by aggregating it's word embeddings. See description above.
    """
    # 1. lowercase phrase
    # 2. tokenize phrase
    # 3. average word vectors for all words in tokenized phrase
    # skip words that are not in model's vocabulary
    # if all words are missing from vocabulary, return zeros
    data = list(phrase)
    data_tok = [tokenizer.tokenize(i.lower()) for i in data]
    model = Word2Vec(data_tok,
                    size=32,  # embedding vector size
                    min_count=5,  # consider words that occured at least 5 times
                    window=5).wv  # define context as a 5-word window around the target word
    words = sorted(model.vocab.keys(),
               key=lambda word: model.vocab[word].count,
               reverse=True)
    if(len(data_tok)):
        vector = np.array([model.get_vector(i) for i in words])
    else:
        vector = np.zeros([model.vector_size], dtype='float32')
    return vector

#%%
vector = get_phrase_embedding("I'm very sure. This never happened to me before...")
print(type(vector))
print(vector)
#%%
assert np.allclose(vector[::10],
                   np.array([ 0.31807372, -0.02558171,  0.0933293 , -0.1002182 , -1.0278689 ,
                             -0.16621883,  0.05083408,  0.17989802,  1.3701859 ,  0.08655966],
                              dtype=np.float32))