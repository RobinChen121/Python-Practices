# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:05:01 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: craw the demand from comments of online tmall store
    
"""
#import openturns as ot
import pandas as pd
from datetime import timedelta
from scipy.stats import norm, gamma, exponweib, kstest, expon, lognorm, rv_continuous
from fitter import Fitter

item_id = 587914055973 # 键盘
#item_id = 588788936455 #鼠标
#item_id = 588953969527 # 耳机
address = 'E:\爬虫练习\天猫评论\商品' + str(item_id) + '-2020-09-19' +'.csv'
comments = pd.read_csv(address, error_bad_lines=False, warn_bad_lines=False) # error_bad_lines=False 防止出现错误
comment_date = comments['rateDate']
# comments['rateDate'] = comments['rateDate'].apply(lambda x: x[0:10]) # 只保留日期
comments['rateDate'] = pd.to_datetime(comments['rateDate'])
comments = comments.sort_values(by = 'rateDate').reset_index() # 默认升序排列， by 后面跟列名
delay_week = 2
delay_days = delay_week * 7

comments['rateDate'] = comments['rateDate'].apply(lambda x : x - timedelta(days=delay_days))
comments['week'] = comments['rateDate'].apply(lambda x : x.strftime('%Y-%m-%W'))
comments['month'] = comments['rateDate'].apply(lambda x : x.strftime('%Y-%m'))

week_demand = pd.DataFrame(comments['week'])
week_demand['demand'] = 1
month_demand = pd.DataFrame(comments['month'])
month_demand['demand'] = 1

week_demand = pd.DataFrame(week_demand.groupby('week', as_index=False).sum())
Biweek_demand = pd.DataFrame(week_demand.groupby(week_demand.index // 2).sum())
Biweek_demand['demand'] = Biweek_demand['demand'] * 10
Biweek_demand['week'] = week_demand.iloc[0::2, 0].values
month_demand = pd.DataFrame(month_demand.groupby('month', as_index=False).sum())
week_demand['demand'] = week_demand['demand'] * 10
month_demand['demand'] = month_demand['demand'] * 10
#month_demand.drop(month_demand.tail(1).index, axis = 0, inplace = True)
#month_demand.drop(month_demand.head(1).index, axis = 0, inplace = True)

history_data = Biweek_demand['demand'].values #[140,130,120,110,100,100,90,90,90,70,60,60,60,60,60,40,40,40,40] 

#history_data = [10,30,150,90,120,320,40,410,170,170,60,60,150,90,150,330,100,60,70,60,100,90,210,130,140,40,40,110,60,40,20,20]
#history_data = [i for i in history_data if i >=300]
history_data = history_data[history_data>100]
#history_data = history_data[history_data>200]

params = gamma.fit(history_data)
params = rv_continuous(gamma, history_data)
statistic, pvalue = kstest(history_data , gamma.cdf, params)
print('gamma: %f %f' %(statistic, pvalue))
print(params)

params = norm.fit(history_data)
statistic, pvalue = kstest(history_data , "norm", params)
print('norm: %f %f' %(statistic, pvalue))

params = exponweib.fit(history_data)
statistic, pvalue = kstest(history_data , "exponweib", params)
print('weibull: %f %f' %(statistic, pvalue))

params = expon.fit(history_data)
statistic, pvalue = kstest(history_data , "expon", params)
print('expon: %f %f' %(statistic, pvalue))

params = lognorm.fit(history_data)
statistic, pvalue = kstest(history_data , "lognorm", params)
print('lognormal: %f %f' %(statistic, pvalue))
print(params)


f = Fitter(history_data, distributions=['gamma', 'norm', 'exponweib', 'expon', 'lognorm'])
f.fit()
f.summary()