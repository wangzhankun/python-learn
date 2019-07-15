#%%
import tensorflow as tf
import numpy as np

#%%
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1 + 1)
inc_v2 = v2.assign(v2 - 1)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    inc_v1.op.run()
    inc_v2.op.run()

    save_path = saver.save(sess, "tensorflow-learn/tutorial/model.ckpt")
    print("Model saved in path: %s" % save_path)

#%%
tf.reset_default_graph()
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "tensorflow-learn/tutorial/model.ckpt")
    print("Model restored.")

    print("v1:", v1.eval())
    print("v2:", v2.eval())

#%%
# 选择要保存和恢复的变量
tf.reset_default_graph()

v1 = tf.get_variable("v1", [3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer=tf.zeros_initializer)

# add ops to save and restore only 'v2' using the name 'v2'
saver = tf.train.Saver({"v2":v2})
with tf.Session() as sess:
    # Initialize v1 since the saver will not.
    v1.initializer.run()
    # only restore v2
    saver.restore(sess, "tensorflow-learn/tutorial/model.ckpt")
    print("v1: ", v1.eval())
    print("v2: ", v2.eval())



#%%
#检查某个检查点中的变量
from tensorflow.python.tools import inspect_checkpoint as chkp
tf.reset_default_graph()
chkp.print_tensors_in_checkpoint_file("tensorflow-learn/tutorial/model.ckpt", tensor_name='', all_tensors=True)


#%%
chkp.print_tensors_in_checkpoint_file("tensorflow-learn/tutorial/model.ckpt", tensor_name='v1',all_tensors=False)

#%%
chkp.print_tensors_in_checkpoint_file("tensorflow-learn/tutorial/model.ckpt", tensor_name='v2', all_tensors=False)


#%%
# 构建和加载SavedModel
'''
tf.saved_model.simple_save(
    session, 
    export_dir, 
    inputs={"x":x, "y":y},
    outputs={"z":z}
    )
'''
#手动构建SavedModel
