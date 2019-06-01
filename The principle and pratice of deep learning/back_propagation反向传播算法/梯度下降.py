#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#%%
# 定义参数
input_dim = 2 # 输入的维度
output_dim = 2 # 输出的维度， 分类数

epsilon = 0.01 # 梯度下降的学习率
reg_lambda = 0.01 # 正则化强度

#%%
# 损失函数
def calculate_loss(model, X, y):
    '''损失函数'''

    num_examples = len(X) # 训练集大小
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # 使用正向传播计算预测值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算损失值
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # 对损失值进行归一化
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss

#%%
# 预测函数
def predict(model, x):
    '''预测函数'''

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 向前传播
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

#%%
# 反向传播算法计算梯度下降
def ANN_model(X, y, nn_hdim):
    '''
    人工神经网络模型函数
    - nn_hdim: 隐层的神经元节点（隐层的数目）
    '''

    num_indim = len(X) # 用于训练网络的输入数据
    model = {} # 模型存储定义
    
    # 随机初始化网络中的权重参数w1, w2 和偏置 b1, b2
    np.random.seed(0)
    W1 = np.random.randn(input_dim, nn_hdim) / np.sqrt(input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, output_dim) / np.sqrt(input_dim)
    b2 = np.zeros((1, output_dim))

    # 批量梯度下降算法BSGD
    num_passes = 20000 # 梯度下降迭代次数
    for i in range(0, num_passes):
        # 向前传播
        z1 = X.dot(W1) + b1 # M[200*2] * M[2*3] --> M[200*3]
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2 # M[200*3] * M[3*2] --> M[200*2]
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 向后传播算法
        delta3 = probs # 得到的预测值  [200*2]
        delta3[range(num_indim), y] -= 1
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) # [200*3]
        dW2 = (a1.T).dot(delta3) # [3*2]
        db2 = np.sum(delta3, axis=0, keepdims=True) # b2的导数 [200*1]
        dW1 = np.dot(X.T, delta2) # W1的导数 [2*3]
        db1 = np.sum(delta2, axis=0) # b1的导数 

        # 添加正则化项
        dW1 += reg_lambda * W1
        dW2 += reg_lambda * W2

        # 根据梯度下降算法更新权重
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # 把新写的参数写入model 字典中进行记录
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if i % 1000 == 0:
            print("Loss after iteration %i: %f", i, calculate_loss(model, X, y))

    return model


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def generate_data():
    np.random.seed(0)
    # X : array of shape [n_samples, 2] The generated samples.
    # y : array of shape [n_samples] The integer labels (0 or 1) 
    # for class membership of each sample.
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


# 执行该人工神经网络模型代码
#%%
X, y = generate_data()
hidden_3_model = ANN_model(X, y, 3)
plt.title("Hidden Layer size 3")
plot_decision_boundary(lambda x: predict(hidden_3_model, x), X, y)


#%%
# 隐层节点数目对人工神经网络模型的影响
# 待输入隐层节点数目
hidden_layer_dimensions = [1,2,3,4,30,50]

for i,nn_hdim in enumerate(hidden_layer_dimensions):
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = ANN_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)

#%%
# 不同学习速率对人工神经网络的影响
epsilon = 0.01 # 梯度下降的学习率
epsilons = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.5, 1]
for i in epsilons:
    epsilon = i
    plt.title("Epsilon %f" % epsilon)
    model = ANN_model(X, y, nn_hdim = 3)
    plot_decision_boundary(lambda x: predict(model, x), X, y)

#%%
