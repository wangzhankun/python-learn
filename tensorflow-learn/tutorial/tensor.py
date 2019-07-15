#%%
import tensorflow as tf
import numpy as np


#%%
# 创建0阶标量
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.1415926, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)


#%%
# 1阶矢量
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers = tf.Variable([3.1415926, 2.71828], tf.float32)
first_primes = tf.Variable([2,3,4,5], tf.int16)
its_very_complicated = tf.Variable([12.3-4.85j, 7], tf.complex64)


#%%
# 2阶
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, True]], tf.bool)
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int16)
rank_of_squares = tf.rank(squarish_squares)
sess = tf.Session()
print(sess.run(rank_of_squares))

#%%
# 4阶张量
my_image = tf.zeros([10, 299, 299, 3])
print(my_image)
print(sess.run(my_image))

#%%
rank = tf.rank(my_image)
print(sess.run(rank))

#%%
my_matrix = tf.random_uniform([3,3,3])
print(my_matrix)
print(sess.run((my_matrix, my_matrix[1, 2, 1], my_matrix[1,2], my_matrix[1])))

#%%
print(my_image.shape)
print(type(my_image.shape))

#%%
zeros = tf.zeros(my_image.shape[1])
print(sess.run(zeros))
zeros = tf.zeros(my_image.shape[:2])
print(zeros.shape)

#%%
# 改变形状
rank_three_tensor = tf.ones([3,4,5])
matrix = tf.reshape(rank_three_tensor, [6,10])
matrixB = tf.reshape(matrix, [3,-1])
matrixAlt = tf.reshape(matrixB, [4,3,-1])
print(matrix.shape)
print(matrixB.shape)
print(matrixAlt.shape)
#%%
errorMatrix = tf.reshape(matrix, [4,2, -1])
print(errorMatrix.shape)

#%%
# 数据类型
# 可以将任意数据结构序列化为string并存储在tf.Tensor中
float_tensor = tf.cast(tf.constant([1,2,3]), dtype=tf.float32)



#%%
constant = tf.constant([1,2,3])
tensor = constant * constant
with tf.Session().as_default():
    print(tensor.eval())
    

#%%
p = tf.placeholder(tf.float32)
t = p + 1.
with tf.Session().as_default():
    # t.eval() # this will fail, since the placeholder did not get a value
    print(t.eval(feed_dict={p:2.}))

#%%
