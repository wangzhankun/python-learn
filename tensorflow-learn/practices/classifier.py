#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#%% 
# 加载Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#%%
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
    'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


#%%
train_images.shape
#(60000,28,28)

#%% [markdown]
# ## 数据处理
print(train_images[0])
train_images = train_images / 255.0 # 归一化
test_images = test_images / 255.0



#%% [markdown]
# ## 构建模型
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),#将二维数组转为28*28=784的一维数组
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=5)

#%% [markdown]
# ## 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#%% [markdown]
# ## 进行预测
predictions = model.predict(test_images)

#%% [markdown]
# 预测是10个数字的数组，描述了模型的信心。
predictions[0] # 第一个预测

#%%
np.argmax(predictions[0]) # 最高置信度的index

#%%
test_labels[0] # 与预测相同，为9

#%%
# 用图表查看全部10个类别
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]
    ), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#%%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

#%%
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

#%%
plt.figure(figsize=(20, 16))
for i in range(1, 21, 2):
    plt.subplot(5, 4, i)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(5, 4, i+1)
    plot_value_array(i, predictions, test_labels)
plt.show()

#%% [markdown]
# 对单个图像进行预测
img = test_images[0]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single))

#%%
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

#%%
prediction_result = np.argmax(predictions_single[0])

#%%
