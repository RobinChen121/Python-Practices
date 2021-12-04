# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:46:40 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import pmdarima as pm
import numpy as np


sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

user_num= len(os.listdir(r'D:\Users\chen_\git\Python-Practices\Elec Power Chongqing\2021\new\各用户用电量表-周\\'))
for i in range(195, user_num + 1):   
    file_name = '用户' + str(i) + '.xlsx'
    df2 = pd.read_excel(r'D:\Users\chen_\git\Python-Practices\Elec Power Chongqing\2021\new\各用户用电量表-周\\' + file_name)
    df = df2[['sum']]
    # fitting a stepwise model:

    
    try:
        stepwise_fit = pm.auto_arima(df, d=1, start_p=1, start_q=1, max_d=2, max_p=2, max_q=2, m=53,
                            start_P=0, seasonal=True, D=1, trace=True,
                            error_action='ignore',  # don't want to know if an order does not work
                            suppress_warnings=True,  # don't want convergence warnings
                            stepwise=True)  # set to stepwise

        stepwise_fit.summary()
        predict_historyData = stepwise_fit.predict_in_sample()
        next_8 = stepwise_fit.predict(n_periods = 8)
    except Exception:
        stepwise_fit = pm.auto_arima(df, d=1, start_p=1, start_q=1, max_d=2, max_p=2, max_q=2, m=53,
                             start_P=0, seasonal=False, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise
        stepwise_fit.summary()
        predict_historyData = stepwise_fit.predict_in_sample()
        next_8 = stepwise_fit.predict(n_periods = 8)
    
    all_predict = np.concatenate((predict_historyData, next_8))
    all_predict = pd.DataFrame(all_predict)
    df_final = pd.concat([df2, all_predict], axis=1)
    df_final = df_final.rename(columns={0: '预测值'})
    df_final.to_excel(r'D:\Users\chen_\git\Python-Practices\Elec Power Chongqing\2021\new\各用户用电量表-周-预测\\' + file_name)
    print()
    


