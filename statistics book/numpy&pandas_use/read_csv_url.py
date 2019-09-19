""" 
3 # @File  : read_csv_url.py
4 # @Author: Chen Zhen
5 # @Date  : 2019/9/18
6 # @Desc  : read csv from an url

"""


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv')
print(df)