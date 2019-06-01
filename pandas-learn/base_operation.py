#%%
from __future__ import print_function
import pandas as pd 
pd.__version__
#%%
city_name = pd.Series(['San Franciso','San Jose','Sacramento'])
city_name
#%%
population = pd.Series([2345,4567,87654])

pd.DataFrame({'City name':city_name,'Population':population})

#%%
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

#%%
# 绘制图表
california_housing_dataframe.hist('housing_median_age')

#%%
cities = pd.DataFrame({'City name':city_name,'Population':population})
print(type(cities['Population']))
cities['City name']

#%%
print(type(cities['City name'][1]))
cities[0:2]
#%%
# 操控数据
population / 1000
#%%
import numpy as np 
np.log(population)

#%%
population.apply(lambda val:val > 1000)

#%%
cities['Area square miles'] = pd.Series([46.87,176.53,97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities 

#%%
cities['Is wide and has saint name'] = (cities['Area square miles'] >50) & cities['City name'].apply(lambda name: name.startswith('San'))
cities