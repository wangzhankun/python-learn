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
