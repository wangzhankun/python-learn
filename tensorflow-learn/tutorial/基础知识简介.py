#%%
import tensorflow as tf
import numpy as np



#%%
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
print(a)
print(b)
print(total)


#%%
writer = tf.summary.FileWriter('tensorflow-learn/tutorial')
writer.add_graph(tf.get_default_graph())

#%%
sess = tf.Session()
print(sess)

#%%
print(sess.run(total))

#%%
print(sess.run({'ab':(a,b), 'total':(total)}))

#%%
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
# 不同时运行，值不同
print(vec)
print(sess.run(vec))
print(sess.run(out1))
print(sess.run(out2))
# 同时运行，值相同
print(sess.run((vec,out1,out2)))

#%%
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print(x)
print(z)

#%%
print(sess.run(z, feed_dict={x:3, y:5.5}))
print(sess.run(z, feed_dict={y:4, x:1}))
print(sess.run(z, feed_dict={x:[1,2,3], y:[2,4,6]}))

#%%
writer.add_graph(tf.get_default_graph())

#%%
my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
print(type(slices))
print(type(next_item))

#%%
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break



#%%
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

#%%
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

#%%
init = tf.global_variables_initializer()
sess.run(init)

#%%
print(sess.run(y, {x:[[1,2,3], [4,5,6]]}))

#%%
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

#%%
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

#%%
print(sess.run(inputs))

#%%
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)


#%%
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)


#%%
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

#%%
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))


#%%
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


#%%
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

#%%
# 完整训练程序如下
import tensorflow as tf
import numpy as np
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

print("y_pred:\n", sess.run(y_pred))

#%%
