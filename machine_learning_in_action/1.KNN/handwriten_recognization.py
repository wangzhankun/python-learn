#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

#%%
def load_data(filedir):
    filenames = os.listdir(filedir)
    imgs = []
    labels = [] #文件名就是label
    print(len(filenames))
    for filename in filenames:
        labels.append(int(filename[0]))
        with open(filedir + "/" + filename) as f:
            img = []
            for line in f.readlines():
                for i in range(len(line)-1):
                    img.append(int(line[i]))
            img = np.array(img).reshape(-1)
            # print(img.shape)
            imgs.append(img)
    return np.array(imgs), np.array(labels)



def classfy(train_data, train_labels, test_data, k=20):
    test_data = np.tile(test_data, (train_data.shape[0], 1))
    distance = np.sum((train_data - test_data) ** 2, axis=1)**0.5
    # print(distance.shape)
    indices = distance.argsort()
    # distance = distance[indices[:20]]
    # print(distance)
    labels = train_labels[indices[:20]]
    return np.argmax(np.bincount(labels))


#%%
train_images, train_labels = load_data("machine_learning_in_action/1.KNN/trainingDigits")
test_images, test_labels = load_data("machine_learning_in_action/1.KNN/testDigits")

#%%
print(train_images.shape)

#%%
predictions = []
errors = []
accuracy = 0
for i in range(test_images.shape[0]):
        label = classfy(train_images, train_labels, test_images[i], k=20)
        predictions.append(label)
        if label == test_labels[i]:
            accuracy += 1
        else:
            errors.append((i,label,test_labels[i]))
#%%
print("Accuracy: {:5.2f}%".format(100 * accuracy/test_images.shape[0]))
# print(predictions)

#%%
print(errors)

#%%
