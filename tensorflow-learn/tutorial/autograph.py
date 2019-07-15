#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph

import numpy as np
import matplotlib.pyplot as plt

#%%
tf.enable_eager_execution()

#%% [markdown]
# # 自动转换python流
# AutoGraph会将如下函数进行转换
# ```[python]
# def square_if_positive(x):
#     if x > 0:
#         x = x * x
#     else:
#         x = 0.
#     return x
# ```
# 转换为使用图构建过程的函数
# ```[python]
# print(autograph.co_code(square_if_positive))
# from __future__ import print_function
# import tensorflow as tf

# def tf__square_if_positive(x):
#   try:
#     with tf.name_scope('square_if_positive'):

#       def if_true():
#         with tf.name_scope('if_true'):
#           x_1, = x,
#           x_1 = x_1 * x_1
#           return x_1,

#       def if_false():
#         with tf.name_scope('if_false'):
#           x_2, = x,
#           x_2 = 0.0
#           return x_2,
#       x = ag__.utils.run_cond(tf.greater(x, 0), if_true, if_false)
#       return x
#   except:
#     ag__.rewrite_graph_construction_error(ag_source_map__)



# tf__square_if_positive.autograph_info__ = {}
# ```


#%%
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0.
    return x
print(autograph.to_code(square_if_positive))

#%%
