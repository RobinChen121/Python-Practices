""" 
# @File  : test.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/3/28
# @Desc  : 

"""

import yfinance as yf

msft = yf.Ticker("MSFT")

# get stock info
msft.info