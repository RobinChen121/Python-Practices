# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:48:03 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: the impact of required service level
    
"""


import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'iniCash', 'sigmaCoe', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6) | (df.serviceRate == 0.5) | (df.serviceRate == 0.7)]
df1 = df1[(df.sigmaCoe == 0.25) & (df.iniCash == 40) & (df['scenario number']==3125)]
df2 = df1.groupby('demand mode').mean()
plt.close('all')