# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:05:01 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: clean the crawling data of online tmall store
    
"""

import pandas as pd
import datetime

item_id = 587914055973 # 4571301 #
address = 'E:\爬虫练习\电商评论\商品' + str(item_id) + '.csv'
comments = pd.read_csv(address, error_bad_lines=False, warn_bad_lines=False)  # error_bad_lines=False 防止出现错误
comment_date = comments['rateDate']
# comments['ratedate'] = comments['ratedate'].apply(lambda x: x[0:10]) # 只保留日期
comments['rateDate'] = pd.to_datetime(comments['rateDate'])
#comments['rateDate'] = comments['rateDate'].apply(lambda x: x - timedelta(14))
comments.sort_values(by='rateDate', inplace = True)
comments.reset_index()  # 默认升序排列， by 后面跟列名
comments['week'] = comments['rateDate'].apply(lambda x: x.strftime('%Y-%m-%W'))
comments['month'] = comments['rateDate'].apply(lambda x: x.strftime('%Y-%m'))



week_demand = pd.DataFrame(comments['week'])
week_demand['demand'] = 1
month_demand = pd.DataFrame(comments['month'])
month_demand['demand'] = 1

week_demand = pd.DataFrame(week_demand.groupby('week', as_index=False).sum())
month_demand = pd.DataFrame(month_demand.groupby('month', as_index=False).sum())
week_demand['demand'] = week_demand['demand'] * 10
month_demand['demand'] = month_demand['demand'] * 10
# month_demand.drop(12, axis=0, inplace=True) # 去掉最后一行
address = 'E:\爬虫练习\电商评论\月需求-商品' + str(item_id) + '-' + str(datetime.date.today()) + '.xlsx'
month_demand.to_excel(address)
