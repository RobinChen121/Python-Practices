""" 
# @File  : macro_eco_data.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/3/28
# @Desc  : use tushare to crawl maro economics data

"""


import tushare as ts

print(ts.get_cpi()) # 得到cpi数据
print(ts.get_hist_data('600519')) # 得到茅台的股票交易数据
