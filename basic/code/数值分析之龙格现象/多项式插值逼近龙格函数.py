'''
龙格函数：f(x)=1/(1+25*x**2)

在这个程序中我们有以下任务：
1.绘制龙格函数图像
2.多项式插值逼近龙格函数
    * 可以自动设置不同的阶数
    * 可以自动设置不同的节点数
    * 绘制逼近得到的多项式的函数，和龙格函数绘制在同一幅图当中
    * 输出多项式，节点和方差结果到csv表格当中
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1,1,1000)
y1 = x ** 2
y2 = 2*x + 1
plt.figure(num='I love you',figsize=(20,10))
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.xlim(-2,2)
plt.xlabel("I like wxf")
plt.ylabel("I like wzk")
new_tricks = np.linspace(-1,1,10)
print(new_tricks)
plt.xticks(new_tricks)
plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
ax = plt.gca()#获取当前坐标轴信息
ax.spines['right'].set_color('none')#设置右边框为白色（默认颜色）
ax.spines['top'].set_color('none')#设置上边框为白色
#使用.xaxis.set_ticks_position设置x坐标刻度数字或名称的位置
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
plt.show()