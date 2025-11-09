"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 09/11/2025, 18:36
@Desc    : 

"""
import pandas as pd
import sys
import matplotlib.pyplot as plt

file_name = 'LD2011_2014.txt'
file_path = ''
if sys.platform == 'win32':
    file_path = 'D:/chenzhen/data/' + file_name
data = pd.read_csv(file_path, sep=';', index_col=0, decimal=',', parse_dates=[0])
data_day = data.resample('D').sum()
data_day[['MT_001', 'MT_002']].plot()
plt.show()
pass
