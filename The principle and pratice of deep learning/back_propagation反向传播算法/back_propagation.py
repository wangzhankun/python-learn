#%% [markdown]
# # 反向传播算法初始化
# 这里的神经网络是有3层（包括输入层），输入层有3个神经元，隐层有4个神经元，输出层有2个神经元<br/>
# 根据网络模型的大小，利用高斯分布函数的权重参数 weights 和偏置参数 biases 产生均值为0、方差为1的随机值

#%%
# 定义神经网络模型架构[input, hidden layer, output],定义每层的神经元数量
import numpy as np 
network_sizes = [3,4,2]

# %%
# 初始化神经网络的参数
sizes = network_sizes
# 网络层数
num_layers = len(sizes)
#%%
# 生成 h*1 的矩阵
biases = [np.random.randn(h,1) for h in sizes[1:]]
print(type(biases))
print(biases)
#%%
# 生成 3*4 和 4*2 的矩阵
weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
print(type(weights))
print(weights)


#%%[markdown]
# # 反向传播算法定义损失函数和激活函数
#%% 
# 损失函数的偏导
def loss_der(network_y, real_y):
    """
    返回损失函数的偏导，损失函数使用 MSE
    L = 1 / 2 * (network_y - real_y) ^ 2
    delta_L = network_y - real_y
    """
    return (network_y - real_y)

#%%
# 激活函数
def sigmoid(z):
    """激活函数使用 sigmoid """
    return 1.0 / (1.0 + np.exp(-z))

# 激活函数的偏导
def sigmoid_der(z):
    """sigmoid 函数的导数 """
    return sigmoid(z) * (1 - sigmoid(z))

#%% [markdown]
# # 反向传播算法的具体实现
# backprop() 函数的输入为 x,y ，根据反向传播的四个基本公式的计算中需要知道每一层神经元的激活值
# 和加权输入值，因此在进行向前传播时，分别使用activations记录每一层的激活值和zs记录每一层
# 的加权输入值

#%%
def backprop(x,y):
    """反向传播算法的实现"""

    # 1) 初始化网络参数的导数  权重w的偏导和偏置b的偏导
    delta_w = [np.zeros(w.shape) for w in weights]
    delta_b = [np.zeros(b.shape) for b in biases]

    # 2) 前向传播 feed forward
    activation = x # 把输入的数据作为第一次激活值
    activations = [x] # 存储网络的激活值
    zs = [] # 存储网络的加权输入值 (z = wx + b)， 注意没有记录input

    for w,b in zip(weights,biases):
        z = np.dot(w,activation) + b
        activation = sigmoid(z)
        activations.append(activation) # 记录激活值
        zs.append(z) # 记录加权输入


    # 3）反向传播
    # bp1 计算传输层误差
    delta_L = loss_der(activations[-1],y) * sigmoid_der(zs[-1])
    
    # bp3 计算输出层关于偏置的误差
    delta_b[-1] = delta_L

    # bp4 损失函数在输出层关于权值的偏导
    delta_w[-1] = np.dot(delta_L,activations[-2].transpose())

    delta_l = delta_L
    for l in range(2,num_layers):
        # bp2 计算第一层误差
        z = zs[-l]
        sp = sigmoid_der(z)
        delta_l = np.dot(weights[-l+1].transpose(), delta_l) * sp
        # bp3 损失函数在l层关于偏置的偏导
        delta_b[-l] = delta_l
        # bp4 损失函数在l层关于权值的偏导
        delta_w[-l] = np.dot(delta_l,activations[-l-1].transpose())

    return (delta_w,delta_b)




#%%
# 产生训练的数据
training_x = np.random.rand(3).reshape(3,1)
print(type(training_x))
print(training_x)
training_y = np.array([0,1]).reshape(2,1)
print(type(training_y))
print(training_y)

#%%
backprop(training_x,training_y)

print("weights:\n{}".format(weights))
print("biases:\n{}".format(biases))