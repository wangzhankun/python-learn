# %% [markdown]
# 主要内容
# * 语言模型
# * 隐马可夫模型
# * veterbi算法
# * 中文分词工具
# 基于统计的分词：
# 1. 建立统计语言模型
# 2. 对句子进行单词划分，对划分结果进行概率计算，获得概率最大的分词方式。
# # 语言模型
# 语言模型指：长度为m的字符串，目的是确定其概率分布P(w1,w2,...,wm)

# %%
import jieba
string = '我来到北京清华大学'
seg_list = jieba.cut(string, cut_all=True)
print(seg_list)
# %%
# 用'/'把生成器中的词串起来显示
print('/'.join(seg_list))
# %%
# 精确模式
seg_list = jieba.cut(string, cut_all=False)
print('/'.join(seg_list))
#%%
# 搜索引擎模式
seg_list = jieba.cut_for_search(string)
print('/'.join(seg_list))

#%%
text = '市场有很多机遇但同时也充满杀机，野蛮生长和快速发展中如何慢慢稳住底盘，驾驭风险，保持起伏冲撞在合理的范围，特别是新兴行业，领军企业更得有胸怀和大局，需要在竞争中保持张弛有度，促成行业建立同盟和百花争艳的健康持续的多赢局面，而非最后比的是谁狠，比的是谁更有底线，劣币驱逐良币，最终谁都逃不了要还的。'
print(text)
#%%
a = jieba.cut(text,cut_all=False)
print('/'.join(a))

#%% [markdown]
# jieba在某些特殊情况下表现不好。但是允许用户自定义添加词到字典中<br/>
# 调用方法是：jieba.load_userdic()<br/>
# 动态修改字典：<br/>
# add_word(word,freq=None,tag=None)<br/>
# del_word(word)<br/>
# suggest_freq(segment,tune=Ture), 可以调节单个词语的词频<br/>