# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:04:01 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

from gurobipy import *
import time
from scenario_mip import mip, mip_allS, mip_pairS, mip_fixQ0
import product
import csv


# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 20000

T = 6
overhead_cost = [2000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
N = len(ini_I)
delay_length = 2
discount_rate = 0.01
B = 10000  # total quantity of order loan
r0 = 0.015  # loan rate
M = 10000

# tree 1
demand_scenarios = [[[133, 30, 49], [246, 58, 57], [87, 39, 20]], [[291, 468, 268], [597, 322, 293], [123, 124, 177]]]
demand_possibility = [[0.1, 0.598, 0.302], [0.286, 0.318, 0.396]]


K = len(demand_possibility[0])  # scenario number in a period
S = K ** T  # total scenario number

# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
for t in range(T):
    slices = round(S * (1 / K) ** (t + 1))  # number of scenario in a slice
    slice_num = round(K ** (t + 1))  # total number of slices
    for i in range(slice_num):
        for j in range(slices * i, slices * (i + 1)):
            for k in range(slices * i, slices * (i + 1)):
                scenarioLink[t][j][k] = 1

# set values for scenario probabilities
scenario_permulations = product.product(range(K), T)
scenario_probs = [0 for s in range(S)]
for s in range(S):
    index = scenario_permulations[s][0]
    scenario_probs[s] = demand_possibility[0][index]
    for i in range(1, len(scenario_permulations[s])):
        index = scenario_permulations[s][i]
        index2 = booming_demand[i]
        scenario_probs[s] = scenario_probs[s] * demand_possibility[index2][index]

#result1 = mip_allS(demand_scenarios, scenarioLink, scenario_probs, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
#print(result1)

values = [[0 for i in range(5)] for s in range(S)]
for sk in range(S):   
    result =  mip(sk, demand_scenarios, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
    values[sk] = [sk, result[0], result[1], result[2], result[3]]    
values.sort(key = lambda x: x[1])  

#headers = ['run','Final Value','Q1_0', 'Q2_0', 'Q3_0']
#with open('results-allScenarios.csv','w', newline='') as f: # newline = '' is to remove the blank line
#    f_csv = csv.writer(f)
#    f_csv.writerow(headers)
#    f_csv.writerows(values)
    
ref_index = values[-1][0]  
# choose another scenario and solve the pair scenario model:
pair_values = [[0 for i in range(4)] for s in range(S-1)]
ref_prob = scenario_probs[ref_index]
index = 0
last_Q_range = [0, 0, 0]
varlue_pair_fix = 0
time_pass = 0
upper_bound = 0
for sk in range(S):
    if abs(sk - ref_index) < 0.5:
        continue
    ref_range = [ref_index, sk]
    value_pair, Q1, Q2, Q3 =  mip_pairS(ref_range, ref_prob, demand_scenarios, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
    Q_range = [Q1, Q2, Q3]
    if index > 0 and abs(Q_range[0] - last_Q_range[0]) < 3 \
                and abs(Q_range[1] - last_Q_range[1]) < 3 and abs(Q_range[2] - last_Q_range[2]) < 3:
        pair_values[index] = [value_pair, value_pair_fix, sk, ref_index, time_pass]
    else:
        tic = time.time()
        value_pair_fix = mip_fixQ0(Q_range, demand_scenarios, scenarioLink, scenario_probs, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
        toc = time.time()
        time_pass = toc - tic
        print('running time is %.2f' % time_pass)
        pair_values[index] = [value_pair, value_pair_fix, sk, ref_index, time_pass]
    with open('results-pairValues.csv','a', newline='') as f: # newline = '' is to remove the blank line
        f_csv = csv.writer(f)
        #f_csv.writerow(headers2)
        temp = pair_values[index]
        f_csv.writerow(temp)
    upper_bound = upper_bound + value_pair * scenario_probs[sk]
    last_Q_range = Q_range
    index = index + 1
upper_bound = upper_bound / (1 - ref_prob)
print('upper_bound: %.4f' % upper_bound)


