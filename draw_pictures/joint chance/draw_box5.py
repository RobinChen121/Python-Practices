# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:07:24 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'price', 'iniCash', 'sigmaCoe', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service', 'holdCost']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6) & (df.holdCost == 0) & (df.sigmaCoe == 0.25) & (df['scenario number']==3125)]
plt.close('all')
plt.figure()
fig1 = sns.boxplot(x="demand mode", y="sim extend SAA service", hue="price",
                 data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.legend()

plt.figure()
fig2 = sns.boxplot(x="demand mode", y="sim SAA service rate", hue="price",
                 data=df1, palette="Set3")
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
fig2.legend()

plt.figure()
fig3 = sns.boxplot(x="demand mode", y="sim extend SAA obj", hue="price", data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig3.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_extendObj.eps', dpi=3000)


plt.figure()
fig4 = sns.boxplot(x="demand mode", y="sim SAA obj", hue="price",
                 data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig4.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_SAAObj.eps', dpi=3000)

df2 = df1[['demand mode', 'price', 'sim extend SAA service', "sim SAA service rate", "sim extend SAA obj", "sim SAA obj"]].round(2)
service_values1 = df2.groupby(['demand mode', 'price'])["sim extend SAA service"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
service_values2 = df2.groupby(['demand mode', 'price'])["sim SAA service rate"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
service_values3 = df2.groupby(['demand mode', 'price'])["sim extend SAA obj"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
service_values4 = df2.groupby(['demand mode', 'price'])["sim SAA obj"].apply(lambda df1: df1.reset_index(drop=True)).unstack()


names = ['price_extendService','price_SAAService', 'price_extendObj', 'price_SAAObj']
row_num = service_values1.shape[0]
col_num = service_values1.shape[1]
for k in range(1, 5):
    txt_name = r"D:\Users\chen_\git\Numerical-tests\joint chance datas" + '\\' +names[k-1] + '.txt'  
    df_txt = eval('service_values'+str(k))
    np.savetxt(txt_name, df_txt, fmt='%d', delimiter='', newline = '\\\\\n')
    txt_name2 = r"D:\Users\chen_\git\Numerical-tests\joint chance datas" + '\\' +names[k-1] + '_latexCode.txt'
    with open(txt_name, 'w') as f:
        for i in range(row_num):
            position_num = 0.1 + math.floor(i/3)*0.9 + i%3*0.25
            some_codes = r'\addplot+[draw=black!50,boxplot={draw position='+ str(round(position_num,2)) + r',box extend=0.2}] table[row sep=\\,y index=0] {' + '\n'
            f.write(some_codes)
            for j in range(col_num):
                f.write(str(df_txt.iloc[i, j])+r'\\')
            f.write('};\n')