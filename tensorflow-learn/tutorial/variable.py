#%%
import tensorflow as tf
import numpy as np


#%%
# 在TensorFlow中，tf.Variable会存储持久性张量。
# 创建变量

my_variable = tf.get_variable("my_variale", [1, 2, 3], dtype=tf.float32)


#%%
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)


#%%
other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23,42]))

#%%
# 初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.report_uninitialized_variables())) # 打印未被初始化的变量

#%%
sess.run(my_variable)

#%%
sess.run(other_variable)

#%%
c_0 = tf.constant(0, name="c")#operation named c
c_1 = tf.constant(2, name="c")#operation named c_1
with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")#operation named "outer/c"
    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c")#operation named "outer/inner/c"

    c_4 = tf.constant(4, name="c")#operation named "outer/c_1"

    with tf.name_scope("inner"):
        c_5 = tf.constant(5, name="c")#operation named "outer/inner_1/c"

#%%
# create a default in-process session
with tf.Session() as sess:
    pass
# create a remote session
with tf.Session("grpc://example.org:2222"):
    pass

#%%
x = tf.constant([[37., -23], [1., 4.]])
w = tf.Variable(tf.random_uniform([2,2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer
with tf.Session() as sess:
    sess.run(init_op)

    print(sess.run(output))

    y_val, output_val = sess.run([y, output])
    print(y_val)
    print(output_val)


#%%
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
    print(sess.run(y, {x:[1.0, 2.0, 3.0]}))
    print(sess.run(y, {x: [0, 0, 5]}))

    # sess.run(y) 
    # sess.run(y,{x:37})# error


#%%
y = tf.matmul([[37, -23.], [1., 4.]], tf.random_uniform([2, 2]))
with tf.Session() as sess:
    # define options for the 'sess.run()' call
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    #define a continer for the retured metadata
    metadata = tf.RunMetadata()
    sess.run(y, options=options, run_metadata=metadata)

    #print the subgraphs that executed on each device
    print(metadata.partition_graphs)
    #print the timings of each operation that excuted
    print(metadata.step_stats)


#%%
