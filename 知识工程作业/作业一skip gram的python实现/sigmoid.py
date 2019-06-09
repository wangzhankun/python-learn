import numpy as np

def sigmoid(x):
    """
    计算输入x的sigmoid。

    参数:
    x -- 常量或者numpy array.

    返回:
    s -- sigmoid(x)
    """
    s = np.true_divide(1, 1 + np.exp(-x)) # 使用np.true_divide进行加法运算
    return s


def sigmoid_grad(s):
    """
    计算sigmoid的梯度，这里的参数s应该是x作为输入的sigmoid的返回值。

    参数:
    s -- 常数或者numpy array。

    返回:
    ds -- 梯度。
    """
    ds = s * (1 - s) # 可以证明：sigmoid函数关于输入x的导数等于`sigmoid(x)(1-sigmoid(x))`
    return ds