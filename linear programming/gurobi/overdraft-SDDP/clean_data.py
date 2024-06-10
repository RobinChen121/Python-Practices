#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:42:11 2024

@author: zhenchen

@disp:  clean the data of the testing results
    
    
"""

import pandas as pd


file_name = 'tests_single_product_leadtime_testing.csv'
df = pd.read_csv(file_name)
drop_index = list(range(10, len(df), 11))
df1 = df.drop(drop_index).reset_index(drop = True)

df_part1 = df1[['ini_cash', 'ini_inventory', 'price', 'unit_order_cost', 'unit_salvage_value', 'deposit_interest_rate', 'overdraft_interest_rate', 'penalty_interest_rate', 'overdraft_limit', 'overhead_costs', \
            'cut_select_number', 'time_limit',  'iter', 'stop_condition']]
df_part2 = df1[['stationary', 'mean_demands', 'realization_num', 'scenario_forward_num', 'iter_limit', 'time', 'final_value', 'Q1', 'opt', 'gap']]
# change string to float numbers
df_part2[['time', 'final_value', 'Q1', 'opt', 'gap']] = df_part2[['time', 'final_value', 'Q1', 'opt', 'gap']].astype(float)
df_mean = df_part2.groupby(['stationary', 'mean_demands', 'realization_num', 'scenario_forward_num', 'iter_limit']).mean().reset_index()

df_mean_part1 = df_mean.groupby(['stationary', 'iter_limit']).mean().reset_index()
df_mean_part2 = df_mean.groupby(['stationary', 'scenario_forward_num']).mean().reset_index()
df_mean_part3 = df_mean.groupby(['stationary', 'realization_num']).mean().reset_index()