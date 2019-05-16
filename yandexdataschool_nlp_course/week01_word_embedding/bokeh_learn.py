# 声明：一下内容非本人原创，所有权利归原作者所有<br/>
# 内容来源：
# https://cloud.tencent.com/developer/article/1376859
#%% [markdown]
# # bokeh使用步骤
# * 准备数据
# * 确定可视化的呈现位置
# * 配置图形界面
# * 连接并绘制数据
# * 组织布局
# * 预览并保存数据创建
# ```python
# # 数据处理库
# import pandas as pd
# import numpy as np
# # bokeh库
# from bokeh.io import output_file, output_notebook
# from bokeh.plotting import figure, show
# from bokeh.models import ColumnarDataSource
# from bokeh.layouts import row, column, gridplot
# from bokeh.models.widgets import Tabs, Panel
# # 步骤一：准备数据
# # 步骤二：决定可视化的呈现位置
# output_file('filename.html') # 生成一个静态html文件
# output_notebook() # 在jupyter notebook中内联呈现
# # 步骤三：设置图形
# fig = figure() # 实例化一个figure()对象
# # 步骤四：连接并绘制数据
# # 步骤五：组织布局
# # 步骤六：预览和保存
# show(fig)
# ```
# ## 步骤一：准备数据
# 此步骤通常涉及Pandas和Numpy等数据处理库的使用，并且会采取必要的步骤将其转换为最适合我们预期可视化的形式。<br/>
# ## 步骤二：确定可视化的位置
# 在此步骤中，你将确定如何生成并最终查看可视化。Bokeh提供了两个常见选项：(1) 生成静态的HTML文件，(2) 在Jupyter Notebook中内联呈现可视化。<br/>
# ## 步骤三：配置图形化界面
# 你将配置图形，为可视化准备画布。在此步骤中，你可以自定义比如标题，刻度线等的所有内容，你还可以设置一组工具，以便与你的可视化进行各种用户交互。<br/>
# ## 步骤四：连接并绘制数据
# 接下来，你将使用Bokeh的渲染器（可视化图）来塑造数据。在这里，你可以灵活地使用许多可用的标记和形状选项从头开始绘制数据，所有这些都可以轻松定制，有极高的创作自由。<br/>
# 此外，Bokeh还具有一些内置功能，可用于构建堆积条形图等大量示例，以及用于创建网络图和地图等更高级可视化的大量示例。<br/>
# ## 步骤五：组织布局
# 此外，Bokeh还具有一些内置功能，可用于构建堆积条形图等大量示例，以及用于创建网络图和地图等更高级可视化的大量示例。<br/>
# 此外，所有绘图可以快速连接在一起，如果手动选择其中一个，也必将会反映在与已连接的其它任何组合上。<br/>
# ## 步骤六：预览并保存数据创建
# 无论是在浏览器还是notebook中查看可视化，都可以浏览可视化，检查自定义，以及使用添加的任何交互。如果对其中的某个很喜欢，还可以将可视化文件保存到图像文件中。

#%%
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from bokeh.io import output_notebook
import bokeh.plotting as pl
import bokeh.models as bm
#%%
# 获取数据
model = KeyedVectors.load_word2vec_format('yandexdataschool_nlp_course/week01_word_embedding/glove-wiki-gigaword-50')
words = sorted(model.vocab.keys(),
               key=lambda word: model.vocab[word].count,
               reverse=True)[:1000]

#%%
print(words[::100])
# %%
# for each words, compute it's vector
word_vectors = np.array([model.get_vector(i) for i in words])
# 降维处理
word_vectors_pca = PCA(n_components=2).fit_transform(word_vectors)
#%%
output_notebook()
def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


# %%
draw_vectors(word_vectors_pca[:, 0], word_vectors_pca[:, 1], token=words)

# %%
word_tsne = TSNE(verbose=True).fit_transform(word_vectors)
# %%
draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='green', token=words)