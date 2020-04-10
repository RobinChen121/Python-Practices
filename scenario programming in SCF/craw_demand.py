# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:05:01 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: craw the demand from comments of online tmall store
    
"""

import pandas as pd

item_id = 591120591607 
address = 'E:\爬虫练习\天猫评论\商品' + str(item_id) + '.csv'
comments = pd.read_csv(address, error_bad_lines=False, warn_bad_lines=False) # error_bad_lines=False 防止出现错误
comment_date = comments['ratedate']
# comments['ratedate'] = comments['ratedate'].apply(lambda x: x[0:10]) # 只保留日期
comments['ratedate'] = pd.to_datetime(comments['ratedate'])
comments = comments.sort_values(by = 'ratedate').reset_index() # 默认升序排列， by 后面跟列名
comments['week'] = comments['ratedate'].apply(lambda x : x.strftime('%Y-%m-%W'))
comments['month'] = comments['ratedate'].apply(lambda x : x.strftime('%Y-%m'))

week_demand = pd.DataFrame(comments['week'])
week_demand['demand'] = 1
month_demand = pd.DataFrame(comments['month'])
month_demand['demand'] = 1

week_demand = pd.DataFrame(week_demand.groupby('week', as_index=False).sum())
month_demand = pd.DataFrame(month_demand.groupby('month', as_index=False).sum())
week_demand['demand'] = week_demand['demand'] * 10
month_demand['demand'] = month_demand['demand'] * 10
month_demand.drop(12, axis = 0, inplace = True)