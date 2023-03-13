# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:43:54 2022

@author: chen
"""

import numpy as np
import pandas as pd


datas = pd.read_excel('rollingtests.xlsx')
df = datas[['demand mode', 'rolling length', 'sample one period', 'rolling obj', 'rolling service rate']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['rolling obj'] = df['rolling obj']
df['rolling service rate'] = df['rolling service rate']

df1 = df.groupby(['demand mode', 'rolling length', 'sample one period']).mean()
df1 = df1.reset_index()

df1 = df1[df1['sample one period'] == 15]
rolling_values1 = df1[df1['rolling length'] == 1].drop(columns = ['sample one period', 'rolling length'])
rolling_values2 = df1[df1['rolling length'] == 2].drop(columns = ['sample one period', 'rolling length'])
rolling_values3 = df1[df1['rolling length'] == 3].drop(columns = ['sample one period', 'rolling length'])

header_txt = 'demand-mode rolling-obj1 rolling-service1 rolling-obj2 rolling-service2 rolling-obj3 rolling-service3'
output = rolling_values1.merge(rolling_values2, on = 'demand mode')
output = output.merge(rolling_values3, on = 'demand mode')
## revise file name below
np.savetxt('rolling_output4.txt', output.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')
