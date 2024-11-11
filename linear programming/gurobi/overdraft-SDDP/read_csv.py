#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:28:48 2024

@author: zhenchen

@disp:  read the output csv files
    
    
"""

import pandas as pd


df1 = pd.read_csv('/Users/zhenchen/Documents/Numerical-tests/overdraft/multiproduct_noenhance_tests.csv')
df1 = df1.drop_duplicates(keep=False, ignore_index=True)
df1 = df1[['demand_pattern', 'time', 'final_value']]
df1.rename(columns = {'time': 'time0', 'final_value': 'final_value0'}, inplace=True)
df1[['time0', 'final_value0']] = df1[['time0', 'final_value0']].astype(float)
df1['demand_pattern'] = df1['demand_pattern'].astype(int)
df_out1 = df1.groupby('demand_pattern').mean(numeric_only = True) 

df2 = pd.read_csv('/Users/zhenchen/Documents/Numerical-tests/overdraft/multiproduct_similarity_tests.csv')
df2 = df2.drop_duplicates(keep=False, ignore_index=True)
df2 = df2[['demand_pattern', 'time', 'final_value']]
df2.rename(columns = {'time': 'time1', 'final_value': 'final_value1'}, inplace=True)
df2[['time1', 'final_value1']] = df2[['time1', 'final_value1']].astype(float)
df2['demand_pattern'] = df2['demand_pattern'].astype(int)
df_out2 = df2.groupby('demand_pattern').mean(numeric_only = True) 

df3 = pd.read_csv('/Users/zhenchen/Documents/Numerical-tests/overdraft/multiproduct_cutselection_tests.csv')
df3 = df3.drop_duplicates(keep=False, ignore_index=True)
df3 = df3[['demand_pattern', 'time', 'final_value']]
df3.rename(columns = {'time': 'time2', 'final_value': 'final_value2'}, inplace=True)
df3[['time2', 'final_value2']] = df3[['time2', 'final_value2']].astype(float)
df3['demand_pattern'] = df3['demand_pattern'].astype(int)
df_out3 = df3.groupby('demand_pattern').mean(numeric_only = True) 

df4 = pd.read_csv('/Users/zhenchen/Documents/Numerical-tests/overdraft/multiproduct_hybrid_tests.csv')
df4 = df4.drop_duplicates(keep=False, ignore_index=True)
df4 = df4[['demand_pattern', 'time', 'final_value']]
df4.rename(columns = {'time': 'time3', 'final_value': 'final_value3'}, inplace=True)
df4[['time3', 'final_value3']] = df4[['time3', 'final_value3']].astype(float)
df4['demand_pattern'] = df4['demand_pattern'].astype(int)
df_out4 = df4.groupby('demand_pattern').mean(numeric_only = True) 

df_final = df_out1.merge(df_out2, on = 'demand_pattern')
df_final = df_final.merge(df_out3, on = 'demand_pattern')
df_final = df_final.merge(df_out4, on = 'demand_pattern')

df_final['gap1'] = abs((df_final['final_value1'] - df_final['final_value0'])/df_final['final_value0'])
df_final['gap2'] = abs((df_final['final_value2'] - df_final['final_value0'])/df_final['final_value0'])
df_final['gap3'] = abs((df_final['final_value3'] - df_final['final_value0'])/df_final['final_value0'])

df_group = df_final.describe()

df_final.to_csv('/Users/zhenchen/Documents/Numerical-tests/overdraft/compare_result.csv')