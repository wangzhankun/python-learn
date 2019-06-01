#%%
from sklearn import linear_model
from sklearn import datasets
import sklearn
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline

#%%
def plot_boundary(pred_func, data, labels):
    """绘制分类边界函数"""

    # 设置最大值和最小值并增加0.5的边界（0.5 padding）
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = x_min, x_max
    h = 0.01 # 点阵间距

    # 生成一个点阵网格，点阵间距为h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 计算分类结果
    z = pred_func(np.c_[xx.ralue(), yy.ravel()])
    z = z.reshape(xx.shape)

    # 绘制轮廓和训练样本，轮廓颜色使用blue，透明度0.2
    plt.contourf(xx,yy,z,cmap=plt.cm.Blues, aplha=0.2)
    plt.scatter(data[:, 0], data[:, 1], s = 40, c = labels, cmap=plt.cm.Vega20c, edgecolors="Black")

np.random.seed(0)
X, y = datasets.make_moons(300, noise=0.25) # 300个数据点，噪声设定为0.25

# 显示产生的医疗数据
plt.scatter(X[:,0], X[:,1], s=50, c=y, cmap=plt.cm.Vega20c, edgecolors="Black")
plt.title('Medical data')
plt.show()