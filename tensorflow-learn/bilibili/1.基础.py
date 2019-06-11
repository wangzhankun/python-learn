#%%
import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_pre = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y_data - y_pre))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

#%%
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[3.], input2:[2.]}))

#%%
inputs = np.arange()