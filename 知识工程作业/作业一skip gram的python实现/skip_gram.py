#%%
import numpy as np
import random
import softmax
import random
import sigmoid
#%%
inputVectors = np.random.randn(5,3)
outputVectors = np.random.randn(5, 3)

sentence = ['a', 'e', 'd', 'b', 'd', 'c','d', 'e', 'e', 'c', 'a']
centerword = 'c'
context = ['a', 'e', 'd', 'd', 'd', 'd', 'e', 'e', 'c', 'a']
tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

inputVectors[tokens['c']]

#%%
print(softmax.softmax(np.array([1,2])))
x = np.array([[1,2],[-1,-2]])
print(sigmoid(x))

