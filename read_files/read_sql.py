# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:04:04 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

# 代码 4-1
from sqlalchemy import create_engine
## 创建一个mysql连接器，用户名为root，密码为1234
## 地址为127.0.0.1，数据库名称为testdb，编码为utf-8
engine = create_engine('mysql+pymysql://root:1234@127.0.0.1:\
3306/testdb?charset=utf8')
print(engine)



# 代码 4-2
import pandas as pd
## 使用read_sql_query查看tesdb中的数据表数目
formlist = pd.read_sql_query('show tables', con = engine)
print('testdb数据库数据表清单为:','\n',formlist)
