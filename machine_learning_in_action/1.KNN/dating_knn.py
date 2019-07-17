#%%
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

#%%
def load_data(filename):
    """导入训练数据
    
    Arguments:
        filename {str} -- 数据文件路径
    
    Return:
        train_data:
        train_labels:
        test_data:
        test_labels:
    """
    with open(filename) as f:
        numberofLines = len(f.readlines())
        returnMat = np.zeros((numberofLines, 3))
        classLabelVector = []
        f.close()

    with open(filename) as f:
        index = 0
        for line in f.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1

    rows,columns = returnMat.shape
    train_data = returnMat[:int(rows*0.8), :]
    train_labels = classLabelVector[:int(rows*0.8)]
    test_data = returnMat[int(rows*0.8):, :]
    test_labels = classLabelVector[int(rows*0.8):]
    # print(test_data)

    return (np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))


def visualizeOriginalData(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:,0], data[:,1], 15.0*np.array(labels), 15.0*np.array(labels))
    plt.show()


def normlize(dataset):
    """[summary]
    
    Arguments:
        dataset {numpy.array} -- 数据集
    
    Returns:
        [type] -- [description]
    """
    minVal = dataset.min(0)
    maxVal = dataset.max(0)
    ranges = maxVal - minVal
    normDataset = np.zeros(dataset.shape)
    m = dataset.shape[0]
    normDataset = dataset - np.tile(minVal, (m, 1))
    normDataset = normDataset / np.tile(ranges,(m,1))
    return normDataset


def classfy(train_data, train_labels, test_data, k=20):
    test_data = np.tile(test_data, (train_data.shape[0], 1))
    distance = np.sum((train_data - test_data) ** 2, axis=1)**0.5
    # print(distance.shape)
    indices = distance.argsort()
    # distance = distance[indices[:20]]
    # print(distance)
    labels = train_labels[indices[:20]]
    return np.argmax(np.bincount(labels))
    


    


def classfier():
    (train_data, train_labels), (test_data, test_labels) = load_data("machine_learning_in_action/1.KNN/datingTestSet2.txt")
    # print(test_labels)
    # visualizeOriginalData(train_data,train_labels)
    train_data = normlize(train_data)
    test_data = normlize(test_data)
    # print(test_data[:,0])
    predictions = []
    accuracy = 0
    for i in range(test_data.shape[0]):
        label = classfy(train_data, train_labels, test_data[i], k=1)
        predictions.append(label)
        if label == test_labels[i]:
            accuracy += 1
    print("Accuracy: {:5.2f}%".format(accuracy/test_data.shape[0]))
    print(predictions)
    



#%%
if __name__ == "__main__":
    classfier()

#%%
