# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:41:47 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: forecast based on the last 12 months
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # pacf 是偏相关系数


sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ini_data = pd.read_excel(r'sumCleanData.xlsx')
ini_data['日期'] = pd.to_datetime(ini_data['日期']).dt.strftime('%y-%m-%d') 
group_data =  ini_data[['日期', 'sum']].groupby('日期').sum()
group_data.plot(kind = 'line', title = '所有用户用电量之和')
result = adfuller(group_data.values)
print('ADF Statistic: %f' % result[0]) # ADF 检验稳定性的，若不稳定则做差分，直到稳定为止
print('p-value: %f' % result[1])

# 也可以通过看差分的自相关系数确定差分阶数
# 确定差分阶数 d, d = 1, 确定移动平均阶数，q = 0
# Original Series
fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(group_data.values); 
axes[0, 0].set_title('原始数据')
plot_acf(group_data.values, ax=axes[0, 1], title = '自相关系数')

# 1st Differencing
axes[1, 0].plot(group_data.diff().dropna().values); 
axes[1, 0].set_title('一阶差分')
plot_acf(group_data.diff().dropna().values, ax=axes[1, 1], title = '自相关系数')

# 2nd Differencing
axes[2, 0].plot(group_data.diff(2).dropna().values); 
axes[2, 0].set_title('二阶差分')
plot_acf(group_data.diff(2).dropna().values, ax=axes[2, 1], title = '自相关系数')

plt.setp(plt.gcf().get_axes(), xticks=[]); # gcf: get current figure
plt.show()

# print()
# result = adfuller(group_data.diff().dropna().values)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

# 确定自回归阶数, p=1
fig, axes = plt.subplots(1, 2)
axes[0].plot(group_data.diff().dropna().values); axes[0].set_title('1st Differencing')
plot_pacf(group_data.diff().dropna().values, ax=axes[1])
plt.show()


# build ARIMA Model
model = sm.tsa.ARIMA(group_data.values, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# # Plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1]) # 密度图 KDE:Kernel Density Estimate
# plt.show()

# Actual vs Fitted
plt.figure()
predict = model_fit.predict(start=1, end = 380)
plt.plot(range(366), group_data['sum'], 'm', label = '原始数据')
plt.plot(range(380), predict, 'r:', label = '预测数据')
plt.legend()
plt.show()
plt.savefig('forecast.png', dpi=1000)


# df1 = ini_data[ini_data.户号对应 == 1]
# df1.plot(x = '日期', y = 'sum', kind = 'line', title = '用户1的历史用电量')
# # df2 = ini_data[ini_data.户号对应 == 2]
# # df2.plot(x = '日期', y = 'sum', kind = 'line', title = '用户2的历史用电量')
# df3 = ini_data[ini_data.户号对应 == 3]
# df3.plot(x = '日期', y = 'sum', kind = 'line', title = '用户3的历史用电量')
# df4 = ini_data[ini_data.户号对应 == 4]
# df4.plot(x = '日期', y = 'sum', kind = 'line', title = '用户4的历史用电量')
# df5 = ini_data[ini_data.户号对应 == 5]
# df5.plot(x = '日期', y = 'sum', kind = 'line', title = '用户5的历史用电量')
# df6 = ini_data[ini_data.户号对应 == 6]
# df6.plot(x = '日期', y = 'sum', kind = 'line', title = '用户6的历史用电量')
# df8 = ini_data[ini_data.户号对应 == 8]
# df8.plot(x = '日期', y = 'sum', kind = 'line', title = '用户8的历史用电量')




