# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:11:48 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 对2021年第二批次的数据按用户切割
    
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

ini_data1 = pd.read_excel(r'sumCleanData.xlsx')
ini_data1['数据日期'] = pd.to_datetime(ini_data1['数据日期']).dt.strftime('%Y-%m-%d') 
ini_data1['week'] = pd.to_datetime(ini_data1['数据日期']).dt.strftime('%Y-%W')
ini_data = ini_data1[ini_data1['sum'] != 0] # 删除零值所在的行
ini_data = ini_data[['编号', '数据日期', 'week', '一级行业', '二级行业', '三级行业', '四级行业', '五级行业', 'sum']]

# 所有用户的用电量之和
# group_data =  ini_data[['数据日期', 'sum']].groupby('数据日期').sum()
# group_data.plot(kind = 'line', title = '所有用户用电量之和')
# group_data.to_excel('groupData.xlsx')

# df1 = ini_data[ini_data.编号 == 1]
# df1.plot(x = '数据日期', y = 'sum', kind = 'line', title = '用户1的历史用电量')

# df2 = ini_data[ini_data.编号 == 2]
# df2.plot(x = '数据日期', y = 'sum', kind = 'line', title = '用户2的历史用电量')

# df3 = ini_data[ini_data.编号 == 3]
# df3.plot(x = '数据日期', y = 'sum', kind = 'line', title = '用户3的历史用电量')

df4 = ini_data[ini_data.编号 == 4]
df4.plot(x = '数据日期', y = 'sum', kind = 'line', title = '用户4的历史用电量')

# 将所有数据输出到单独的 excel 表
user_num = ini_data['编号'].max()
for i in range(1, user_num+1):
    df = ini_data[ini_data.编号 == i]
    file_address = "D:\\Users\\chen_\\git\\Python-Practices\\Elec Power Chongqing\\2021\\new\\各用户用电量表-天\\"
    xls_name = file_address + '用户' + str(i) + '.xlsx'
    df.to_excel(xls_name, index = False)
    df2 = df.groupby(['编号', 'week', '一级行业', '二级行业', '三级行业', '四级行业', '五级行业'], as_index = False).sum()
    file_address = "D:\\Users\\chen_\\git\\Python-Practices\\Elec Power Chongqing\\2021\\new\\各用户用电量表-周\\"
    xls_name = file_address + '用户' + str(i) + '.xlsx'
    df2.to_excel(xls_name, index = False)

df2.plot(x = 'week', y = 'sum', kind = 'line', title = '用户周用电量')