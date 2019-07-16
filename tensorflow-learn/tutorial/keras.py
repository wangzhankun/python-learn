#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

print(tf.__version__)
print(tf.keras.__version__)

#%%
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

#%%
# create a sigmoid layer
layers.Dense(64, activation='sigmoid')
layers.Dense(64, activation=tf.sigmoid)

#A linear layer with L1 regularization of factor 0.01 applid to the kernel matrix
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

#A linear layer with L2 regularization of factor 0.01 applid to the bias vector
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix
# 正交矩阵
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))


#%%
# set up training 
model = tf.keras.Sequential(
    # add a densel-conncted layer with 64 units to the model
    [
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])

#%%
# mean squared error regression
model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss='mse',
    metrics=[tf.keras.metrics.categorical_accuracy]
)
# categorical classification
model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.001),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy]
)


#%%
# for small datasets, use in-memory NumPy arrays to train and evaluate
def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n,n_class))
    labels[np.arange(n), classes] = 1
    return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

model.fit(data, labels, epochs=100, batch_size=32)


#%%
# use Datasets API to scale to large datasets or multi-device training
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)
# since the dataset yields batches of data, this snippet does not require a batch_size
#%%
# validation
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset,validation_steps=3)

#%%
# evaluate and predict
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

#%%
result = model.predict(data, batch_size=32)
print(result.shape)

#%%
# 使用函数式API构建全连接网络
inputs = tf.keras.Input(shape=(32,))#returns a placeholder tensor
# A layer instance is callable on the tensor, and return a tensor
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)



#%%
model = tf.keras.Model(inputs=inputs, outputs=predictions)
# The compile step specifies the training configuration
model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
model.fit(data, labels, batch_size=32, epochs=10)

#%% [markdown]
# ## 模型子类化
# 通过对```tf.keras.Model```进行子类化并定义自己的前向传播来构建自定义的模型。
# 在```__init__```方法中创建层并将它们设置为类实例的属性。在call方法中定义前向传播。
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # Define your layers here
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in '__init'__)
        x = self.dense_1(inputs)
        return self.dense_2(x)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


#%%
model = MyModel(num_classes=10)
model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(data, labels, batch_size=32, epochs=5)

#%% [markdown]
# ## 自定义层
# * build 创建层的权重。使用add_weight方法添加权重
# * call 定义前向传播
# * compute_output_shape 指定在给定输入形状的情况下如何计算层的输出形状
# * 通过实现get_config和from_config类方法序列化层
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for the layer.
        self.kernel = self.add_weight(
            name='kernel',
            shape=shape,
            initializer='uniform',
            trainable=True
        )
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    
    def get_config(self, input_shape):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#%%
model = tf.keras.Sequential(
    [
        MyLayer(10),
        layers.Activation('softmax')
        ]
)
# The compile step specifies the training configuration
model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(data, labels, batch_size=32, epochs=5)

#%% [markdown]
# ## 回调
# 回调是传递给模型的对象，用于在训练期间自定义该模型并拓展其行为
# * tf.keras.callbacks.ModelCheckpoint 定期保存模型的检查点
# * tf.keras.callbacks.LearningRateScheduler 动态更改学习速率
# * tf.keras.callbacks.EarlyStopping 在验证效果不再改进时终端训练
# * tf.keras.callbacks.TensorBoard 使用tensorboard监控模型的行为
callbacks = [
    # Interrupt training if 'val_loss' stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
    # Write TensorBoard logs to './logs/' directory
    tf.keras.callbacks.TensorBoard(log_dir='tensorflow-learn/tutorial/logs')
]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
    validation_data=(val_data, val_labels))



#%% [markdown]
# ## 保存和恢复
# ### 仅限权重
# 使用```tf.keras.Model.save_weights```保存并加载模型的权重
model = tf.keras.Sequential(
    [
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Save weights to TensorFlow Checkpoint file
model.save_weights("tensorflow-learn/tutorial/models/weight")
# Restore the model's state
model.load_weights('tensorflow-learn/tutorial/models/weight')

#%% [markdown]
# 默认情况下，会以TensorFlow检查点文件格式保存模型的权重。
# 权重也可以另存为Keras HDF5格式。
#%%
# save weights to a HDF5 file
model.save_weights('tensorflow-learn/tutorial/models/weight.h5', save_format='h5')
model.load_weights('tensorflow-learn/tutorial/models/weight.h5')


#%% [markdown]
# 可以保存模型的配置，此操作会对模型架构（不包含权重）进行序列化。
# 即使没有定义原始模型的代码，保存的配置也可以重新创建并初始化相同的模型。
# Keras支持JSON和YAML序列化格式。
json_string = model.to_json()
json_string

#%%
import json
import pprint
pprint.pprint(json.loads(json_string))

#%%
fresh_mode = tf.keras.models.model_from_json(json_string)

#%%
yaml_string = model.to_yaml()
print(yaml_string)

#%%
fresh_mode = tf.keras.models.model_from_yaml(yaml_string)

#%% [markdown]
# ### 保存整个模型
# 整个模型都可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。
# 这样，可以对模型设置检查点并稍后从完全相同状态继续训练。
model = tf.keras.Sequential(
    [
        layers.Dense(10, activation='softmax', input_shape=(32, )),
        layers.Dense(10, activation='softmax')
        ]
)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(data, labels, batch_size=32, epochs=10)
# only save to a HDF5 file
model.save('tensorflow-learn/tutorial/models/model.h5')
model = tf.keras.models.load_model('tensorflow-learn/tutorial/models/model.h5')


#%% [markdown]
# ## Eager Execution
# Eager Execution是一种命令式编程环境，可立即评估操作。此环境对于Keras并不是必需的。
# 但是受tf.keras支持，并且可以检查程序和调试
# 所有tf.keras模型构建API斗鱼Eager Execution兼容。



#%% [markdown]
# ## Estimator
# Estimator API用于分布式环境训练模型。