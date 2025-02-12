# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:49:35 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: draw bars for the joint chance results with confidence intervals for different scenario size
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'sigmaCoe', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6) & (df.sigmaCoe == 0.25) & (df.iniCash == 40)]
plt.close('all')
plt.figure()
fig1 = sns.boxplot(x="demand mode", y="sim extend SAA service", hue="scenario number",
                 data=df1, palette="Set3")
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
plt.legend()
# figure = fig1.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_extendService.eps', dpi=3000)

df2 = df1[['demand mode', 'scenario number', 'sim extend SAA service', "sim SAA service rate", "sim extend SAA obj", "sim SAA obj"]].round(1)
# df_csv.to_csv(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\df.csv', index=False)

statistic_values1 = df2.groupby(['demand mode', 'scenario number'])["sim extend SAA service"].describe()
statistic_values2 = df2.groupby(['demand mode', 'scenario number'])["sim SAA service rate"].describe()
statistic_values3 = df2.groupby(['demand mode', 'scenario number'])["sim extend SAA obj"].describe()
statistic_values4 = df2.groupby(['demand mode', 'scenario number'])["sim SAA obj"].describe()

sampleNum_values1 = df2.groupby(['demand mode', 'scenario number'])["sim extend SAA service"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
sampleNum_values2 = df2.groupby(['demand mode', 'scenario number'])["sim SAA service rate"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
sampleNum_values3 = df2.groupby(['demand mode', 'scenario number'])["sim extend SAA obj"].apply(lambda df1: df1.reset_index(drop=True)).unstack()
sampleNum_values4 = df2.groupby(['demand mode', 'scenario number'])["sim SAA obj"].apply(lambda df1: df1.reset_index(drop=True)).unstack()

names = ['sampleNum_extendService','sampleNum_SAAService', 'sampleNum_extendObj', 'sampleNum_SAAObj']
row_num = sampleNum_values1.shape[0]
col_num = sampleNum_values1.shape[1]
for k in range(1, 5):
    txt_name = r"D:\Users\chen_\git\Numerical-tests\joint chance datas" + '\\' +names[k-1] + '.txt'  
    df_txt = eval('sampleNum_values'+str(k))
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
        # f.write('\n xtick = {') #  xtick = {0.35,1.25,2.15,3.05,3.95,4.85,5.75,6.65,7.55,8.45},
        
        # for index, i in enumerate(range(col_num)):
        #     position_num = 0.35 + i*0.9
        #     if index > 0:
        #         f.write(',')
        #     f.write(str(round(position_num,2)))
        # f.write('},')
        


# for k in range(1, 5):
#     df_txt = eval('sampleNum_values'+str(k))
#     txt_name = r"D:\Users\chen_\git\Numerical-tests\joint chance datas" + '\\' +names[k-1] + '.txt'
#     with open(txt_name, 'w') as f:
#         for i in range(row_num):
#             f.write(str(df_txt.index[i][0])+r'\\'+str(df_txt.index[i][1])+r'\\  ')
#             for j in range(col_num):
#                 f.write(str(df_txt.iloc[i][j])+r'\\')
#             f.write('\n')

# sampleNum_values1.to_csv(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\sampleNum_extendService.csv',header=False, index=False)
# sampleNum_values2.to_csv(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\sampleNum_SAAService.csv',header=False, index=False)
# sampleNum_values3.to_csv(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\sampleNum_extendObj.csv',header=False, index=False)
# sampleNum_values4.to_csv(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\sampleNum_SAAObj.csv',header=False, index=False)

plt.figure()
fig2 = sns.boxplot(x="demand mode", y="sim SAA service rate", hue="scenario number",
                 data=df1, palette="Set3")
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
fig2.legend() # loc='upper left', ncol=3
# figure = fig2.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_SAAService.eps', dpi=3000)

plt.figure()
fig3 = sns.boxplot(x="demand mode", y="sim extend SAA obj", hue="scenario number", data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig3.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_extendObj.eps', dpi=3000)


plt.figure()
fig4 = sns.boxplot(x="demand mode", y="sim SAA obj", hue="scenario number",
                 data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig4.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_SAAObj.eps', dpi=3000)












