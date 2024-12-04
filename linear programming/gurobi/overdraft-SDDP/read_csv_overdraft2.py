#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:54:47 2024

@author: zhenchen

@disp:  for the comparison results of different overdraft parameter values
    
    
"""

import pandas as pd


# df1 = pd.read_csv('/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/multiproduct_noenhance_tests.csv')
# df1 = df1.drop_duplicates(keep=False, ignore_index=True)
# df1 = df1[['demand_pattern', 'time', 'lower bound']]
# df1[['time', 'lower bound']] = df1[['time', 'lower bound']].astype(float)
# df1['demand_pattern'] = df1['demand_pattern'].astype(int)
# df_out1 = df1.groupby('demand_pattern').mean(numeric_only = True) 

df2 = pd.read_csv('/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/multiproduct_cutselection_tests.csv')
df2 = df2.drop_duplicates(keep=False, ignore_index=True)
df2['overdraft_interest_rate'] = df2['overdraft_interest_rate'].astype(float)
df2['overdraft_limit'] = df2['overdraft_limit'].astype(int)

df1 = df2[(df2['overdraft_interest_rate'] == 0.1) & (df2['overdraft_limit'] == 500)]
df1 = df1[['demand_pattern', 'time', 'lower bound']]
df1[['time', 'lower bound']] = df1[['time', 'lower bound']].astype(float)
df1['demand_pattern'] = df1['demand_pattern'].astype(int)
df_out1 = df1.groupby('demand_pattern').mean(numeric_only = True) 

df2_1 = df2[(df2['overdraft_interest_rate'] == 0.1) & (df2['overdraft_limit'] == 300)]
df2_1['demand_pattern'] = df2_1['demand_pattern'].astype(int)
df2_1 = df2_1[['demand_pattern', 'time', 'lower bound']]
df2_1[['time', 'lower bound']] = df2_1[['time', 'lower bound']].astype(float)
df_out2_1 = df2_1.groupby('demand_pattern').mean(numeric_only = True) 

df = df_out2_1['lower bound']
df = df.to_frame()
df.rename(columns = {'lower bound': 'lower bound0'}, inplace=True)

df2_2 = df2[(df2['overdraft_interest_rate'] == 0.1) & (df2['overdraft_limit'] == 400)]
df2_2['demand_pattern'] = df2_2['demand_pattern'].astype(int)
df2_2 = df2_2[['demand_pattern', 'time', 'lower bound']]
df2_2[['time', 'lower bound']] = df2_2[['time', 'lower bound']].astype(float)
df_out2_2 = df2_2.groupby('demand_pattern').mean(numeric_only = True) 

df['lower bound1'] = df_out2_2['lower bound']
df['lower bound2'] = df_out1['lower bound']

df2_3 = df2[(df2['overdraft_interest_rate'] == 0.1) & (df2['overdraft_limit'] == 600)]
df2_3['demand_pattern'] = df2_3['demand_pattern'].astype(int)
df2_3 = df2_3[['demand_pattern', 'time', 'lower bound']]
df2_3[['time', 'lower bound']] = df2_3[['time', 'lower bound']].astype(float)
df_out2_3 = df2_3.groupby('demand_pattern').mean(numeric_only = True) 

df['lower bound3'] = df_out2_3['lower bound']

df2_4 = df2[(df2['overdraft_interest_rate'] == 0.1) & (df2['overdraft_limit'] == 700)]
df2_4['demand_pattern'] = df2_4['demand_pattern'].astype(int)
df2_4 = df2_4[['demand_pattern', 'time', 'lower bound']]
df2_4[['time', 'lower bound']] = df2_4[['time', 'lower bound']].astype(float)
df_out2_4 = df2_4.groupby('demand_pattern').mean(numeric_only = True) 

df['lower bound4'] = df_out2_4['lower bound']


df3_1 = df2[(df2['overdraft_interest_rate'] == 0) & (df2['overdraft_limit'] == 500)]
df3_1['demand_pattern'] = df3_1['demand_pattern'].astype(int)
df3_1 = df3_1[['demand_pattern', 'time', 'lower bound']]
df3_1[['time', 'lower bound']] = df3_1[['time', 'lower bound']].astype(float)
df_out3_1 = df3_1.groupby('demand_pattern').mean(numeric_only = True) 

dff = df_out3_1['lower bound']
dff = dff.to_frame()
dff.rename(columns = {'lower bound': 'lower bound0'}, inplace=True)

df3_2 = df2[(df2['overdraft_interest_rate'] == 0.05) & (df2['overdraft_limit'] == 500)]
df3_2['demand_pattern'] = df3_2['demand_pattern'].astype(int)
df3_2 = df3_2[['demand_pattern', 'time', 'lower bound']]
df3_2[['time', 'lower bound']] = df3_2[['time', 'lower bound']].astype(float)
df_out3_2 = df3_2.groupby('demand_pattern').mean(numeric_only = True) 

dff['lower bound1'] = df_out3_2['lower bound']
dff['lower bound2'] = df_out1['lower bound']

df3_3 = df2[(df2['overdraft_interest_rate'] == 0.15) & (df2['overdraft_limit'] == 500)]
df3_3['demand_pattern'] = df3_3['demand_pattern'].astype(int)
df3_3 = df3_3[['demand_pattern', 'time', 'lower bound']]
df3_3[['time', 'lower bound']] = df3_3[['time', 'lower bound']].astype(float)
df_out3_3 = df3_3.groupby('demand_pattern').mean(numeric_only = True) 

dff['lower bound3'] = df_out3_3['lower bound']

df3_4 = df2[(df2['overdraft_interest_rate'] == 0.2) & (df2['overdraft_limit'] == 500)]
df3_4['demand_pattern'] = df3_4['demand_pattern'].astype(int)
df3_4 = df3_4[['demand_pattern', 'time', 'lower bound']]
df3_4[['time', 'lower bound']] = df3_4[['time', 'lower bound']].astype(float)
df_out3_4 = df3_4.groupby('demand_pattern').mean(numeric_only = True) 

dff['lower bound4'] = df_out3_4['lower bound']

# df_p1 = dff['lower bound2']
# df_p1 = df_p1.to_frame()

df_p1 = pd.DataFrame()
df_p1['gap1'] = (dff['lower bound0'] - dff['lower bound2'])/abs(dff['lower bound2'])
df_p1['gap2'] = (dff['lower bound1'] - dff['lower bound2'])/abs(dff['lower bound2'])
df_p1['gap3'] = (dff['lower bound2'] - dff['lower bound2'])/abs(dff['lower bound2'])
df_p1['gap4'] = (dff['lower bound3'] - dff['lower bound2'])/abs(dff['lower bound2'])
df_p1['gap5'] = (dff['lower bound4'] - dff['lower bound2'])/abs(dff['lower bound2'])

df_p2 = pd.DataFrame()
df_p2['gap1'] = (df['lower bound0'] - df['lower bound2'])/abs(df['lower bound2'])
df_p2['gap2'] = (df['lower bound1'] - df['lower bound2'])/abs(df['lower bound2'])
df_p2['gap3'] = (df['lower bound2'] - df['lower bound2'])/abs(df['lower bound2'])
df_p2['gap4'] = (df['lower bound3'] - df['lower bound2'])/abs(df['lower bound2'])
df_p2['gap5'] = (df['lower bound4'] - df['lower bound2'])/abs(df['lower bound2'])

df_final = df_p1.merge(df_p2, on = 'demand_pattern')
