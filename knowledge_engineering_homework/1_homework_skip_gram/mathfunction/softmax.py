import numpy as np

def softmax(x):
    """
    对输入x的每一行计算softmax。
     参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理x
    """
    origin_shape = x.shape

    if len(origin_shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))

        x /= tmp.reshape((x.shape[0], 1))

    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    return x