# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:51:03 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # pacf 是偏相关系数
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm

sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.options.mode.chained_assignment = None  # default='warn'


final_result = pd.DataFrame()
files = os.listdir(r'D:\项目\电网\2022\各用户用电量\\')
user_num= len(files)
for m in range(user_num):
    file_name = 'D:\\项目\\电网\\2022\\各用户用电量\\' + files[m]
    old_data = pd.read_excel(file_name)
    old_data['data_date'] = pd.to_datetime(old_data['data_date']).dt.strftime('%Y-%m-%d') 
    
    # detect outliers and drop
    old_data= old_data[old_data['day_consuming'] != 0]
    data = old_data[(np.abs(stats.zscore(old_data['day_consuming'])) < 2)]  # 2 sigma as outlier
    data = data.reset_index(drop=True)
    
    # decide the number of order difference d 确定差分阶数
    # ADF test
    day_datas = data['day_consuming']
    if len(day_datas) < 10:       
        continue
    result = adfuller(day_datas)
    p_value = result[1]
    d = 0
    diff_values = day_datas
    while p_value > 0.5:
        d = d + 1
        result = adfuller(diff_values.diff().dropna())
        p_value = result[1]    
        diff_values = diff_values.diff().dropna()
        
    # decide the value of p 
    try: # arima can't fit for pure linear data  
        pacf_values = pacf(diff_values)
        num = len(pacf_values)
        bound = 1.96 / math.sqrt(len(diff_values))
        p = 0
        for i in range(num):
            if abs(pacf_values[i]) < bound:
                p = i-1
                break
            if i > 5:
                p = 5
                break
    except:
        n = len(day_datas)
        x = np.arange(n)
        y = np.array(day_datas)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        predict = model.predict(x)
        predict_df = pd.Series(predict, name = '预测值')
        predict_df = predict_df.to_frame()
        result = pd.concat([predict_df, data], axis=1)
        file_name = 'D:\\项目\\电网\\2022\\各用户用电量-预测\\' + files[m]
        result.to_excel(file_name)
        continue
        
    
    # decide the value of q 
    acf_values = acf(diff_values)
    num = len(acf_values)
    bound = 1.96 / math.sqrt(len(diff_values))
    q = 0
    for i in range(num):
        if abs(acf_values[i]) < bound:
            q = i-1
            break
        if i > 10:
            q = 10
            break
               
    # build ARIMA Model  
    # arima can't fit for pure linear data  
    model = sm.tsa.ARIMA(day_datas, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    print(model_fit.summary())
    
    # predict for one more week
    num_obvservation = len(day_datas)
    predict = model_fit.predict(start=1, end = num_obvservation + 7)
    predict = predict.reset_index(drop=True)                    
    predict_df = pd.Series(predict, name = '预测值')
    predict_df = predict_df.to_frame()
    result = pd.concat([predict_df, data], axis=1)
    result['cons_no'].fillna(result['cons_no'][0], inplace = True)
    datelist = pd.date_range(data['data_date'].iloc[-1], periods=8).strftime('%Y-%m-%d').to_series()
    datelist = datelist[1:]
    result['data_date'][result['data_date'].isnull()] = datelist.values
    file_name1 = 'D:\\项目\\电网\\2022\\各用户用电量-预测\\' + files[m]
    result.to_excel(file_name1)
    final_result = pd.concat([final_result, result])
        


# plt.figure()
# plt.plot(range(25), day_datas, 'm', label = '原始数据')
# plt.plot(range(33), predict, 'r:', label = '预测数据')
# plt.legend()
# plt.show()
# plt.savefig('forecast.png', dpi=1000)
