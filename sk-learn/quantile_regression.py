""" 
# @File  : quantile_regression.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/12/7
# @Desc  :  Quantile regression by statsmodels

"""

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [36, 29, 24, 21, 20, 21, 24, 39, 36]

data = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])

# mod = smf.quantreg('y~x', data)
# res = mod.fit(q=0.9)
# print(res.summary())

# plt.scatter(x, y)
# xx = np.arange(min(x), max(x), 0.01)
# yy = [i*res.params['x'] + res.params['Intercept'] for i in xx]
# plt.plot(xx, yy, color='red')
# plt.show()
# plt.close()

data = sm.datasets.engel.load_pandas().data
mod = smf.quantreg('foodexp ~ income', data)
res = mod.fit(q=0.6)
print(res.summary())

plt.scatter(data['income'], data['foodexp'])
xx = np.arange(min(data['income']), max(data['income']))
yy = [i*res.params['income'] + res.params['Intercept'] for i in xx]
plt.plot(xx, yy, color='red')
plt.show()

