# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:41:47 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm


sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']

ini_data = pd.read_excel(r'sumCleanData.xlsx')
ini_data['日期'] = pd.to_datetime(ini_data['日期']).dt.strftime('%y-%m-%d') 
group_data =  ini_data[['日期', 'sum']].groupby('日期').sum()
group_data.plot(kind = 'line', title = '所有用户用电量之和')
df1 = ini_data[ini_data.户号对应 == 1]
df1.plot(x = '日期', y = 'sum', kind = 'line', title = '用户1的历史用电量')
# df2 = ini_data[ini_data.户号对应 == 2]
# df2.plot(x = '日期', y = 'sum', kind = 'line', title = '用户2的历史用电量')
df3 = ini_data[ini_data.户号对应 == 3]
df3.plot(x = '日期', y = 'sum', kind = 'line', title = '用户3的历史用电量')
df4 = ini_data[ini_data.户号对应 == 4]
df4.plot(x = '日期', y = 'sum', kind = 'line', title = '用户4的历史用电量')
df5 = ini_data[ini_data.户号对应 == 5]
df5.plot(x = '日期', y = 'sum', kind = 'line', title = '用户5的历史用电量')
df6 = ini_data[ini_data.户号对应 == 6]
df6.plot(x = '日期', y = 'sum', kind = 'line', title = '用户6的历史用电量')
df8 = ini_data[ini_data.户号对应 == 8]
df8.plot(x = '日期', y = 'sum', kind = 'line', title = '用户8的历史用电量')




