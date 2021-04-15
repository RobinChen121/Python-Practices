# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:20:49 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: compute the lower bounds for saa in the maximization problem
    
"""

from gurobipy import *
from gurobipy import LinExpr
from gurobipy import GRB
from gurobipy import Model
import time
import math
import numpy as np
import scipy.stats as st
from math import exp
import itertools
import csv
from saa_mip_scenarios import mip_fixQ0, mip_pairS # very slow if saa_mip_scenarios have implementable codes
import _pickle as cPickle # save python list to files


def lognorm_ppf(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = exp(mu)
    return st.lognorm.ppf(x, shape, loc, scale)

def generate_sample(sample_num, trunQuantile, mus, sigmas, booming_demand):
    T = len(booming_demand)
    N = len(mus)
    samples = [[[0 for i in range(sample_num[t])] for n in range(N)] for t in range(T)]
    for t in range(T):
        for i in range(sample_num[t]):
            rand_p = np.random.uniform(trunQuantile*i/sample_num[t], trunQuantile*(i+1)/sample_num[t])
            for n in range(N):
                samples[t][n][i] = lognorm_ppf(rand_p, mus[n][booming_demand[t]], sigmas[n][booming_demand[t]])
    return samples


# make s as first index
def get_sample2(samples, scenario_permulations):
    S = len(scenario_permulations)
    T = len(samples)
    samples2 = [[[0 for n in range(N)] for t in range(T)] for s in range(S)]
    for s in range(S):
        index = scenario_permulations[s]
        for t in range(T):
            samples2[s][t] = [samples[t][0][index[t]], samples[t][1][index[t]], samples[t][2][index[t]]]
    return samples2

  
    
# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 20000 # should be same with original

T = 6
overhead_cost = [2000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
delay_length = 2
discount_rate = 0.01
B = 10000  # total quantity of order loan
ro = 0.015  # loan rate
M = 10000

mus = [[3.66, 5.79], [4.13, 5.91], [3.54, 4.96]]
sigmas = [[0.6, 0.26], [0.66, 0.33], [0.46, 0.18]]
#mus = [[3.66, 5.79], [4.13, 5.91]]
#sigmas = [[0.6, 0.26], [0.66, 0.33]]
N = len(mus)
sample_nums = [5, 5, 5, 3, 3, 3]
trunQuantile = 1

#samples = generate_sample(sample_nums, trunQuantile, mus, sigmas, booming_demand[0:T])
#cPickle.dump(samples, open("data.pkl", "wb"))

for kk in range(1):
    file_name = 'data' + str(kk+1) + '.pkl'
    samples = cPickle.load(open(file_name, "rb"))
    SS = np.prod(sample_nums[0:T]) # number of samples
    arr = []
    for t in range(T):
        arr.append(range(sample_nums[t]))
    scenario_permulations = list(itertools.product(*arr))
    
    # samples2 = get_sample2(samples, scenario_permulations)
    
    #headers = ['run','Final Value','Q1_0', 'Q2_0', 'Q3_0']
    #values = [[0 for i in range(5)] for s in range(SS)]
    #for sk in range(SS):   
    #    result =  mip(samples2[sk], B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
    #    values[sk] = [sk, result[0], result[1], result[2], result[3]]    
    #values.sort(key = lambda x: x[1])  
    #
    #with open('results-allScenarios.csv','w', newline='') as f: # newline = '' is to remove the blank line
    #    f_csv = csv.writer(f)
    #    f_csv.writerow(headers)
    #    f_csv.writerows(values)
            
    ## choosing the reference scenario:the one with lowest value        
         
    ref_index = 3374 # index of each period for the reference scenario
    
    # choose another scenario and solve the pair scenario model:
    pair_values = [[0 for i in range(3)] for s in range(SS-1)]
    last_Q_range = [0, 0, 0]
    this_value = 0
    time_pass = 0
    index = 0
    for sk in range(SS):
        if abs(sk - ref_index) < 0.5:
            continue
        ref_range = [ref_index, sk]
        value, Q1, Q2, Q3 = mip_pairS(ref_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
        Q_range = [Q1, Q2, Q3]
        if index > 0 and abs(Q_range[0] - last_Q_range[0]) < 3 \
                    and abs(Q_range[1] - last_Q_range[1]) < 3 and abs(Q_range[2] - last_Q_range[2]) < 3:
            pair_values[index] = [value, this_value, sk, ref_index, time_pass]
        else:       
            tic = time.time()
            this_value = mip_fixQ0(Q_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
            toc = time.time()
            time_pass = toc - tic
            print('running time is %.2f' % time_pass)
            pair_values[index] = [value, this_value, sk, ref_index, time_pass]
        with open('results-pairValues.csv','a', newline='') as f: # newline = '' is to remove the blank line
            f_csv = csv.writer(f)
            #f_csv.writerow(headers2)
            temp = pair_values[index]
            f_csv.writerow(temp)
        last_Q_range = Q_range
        index = index + 1


#headers2 =  ['value','S1', 'S2']
#with open('results-pairValues.csv','a', newline='') as f: # newline = '' is to remove the blank line
#    f_csv = csv.writer(f)
#    #f_csv.writerow(headers2)
#    f_csv.writerows(pair_values)
    
#Q_range =[0, 0, 0]
#this_value_refS, Q_range[0], Q_range[1], Q_range[2] = mip(samples2[ref_index], B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
#
#this_value_opt, _, _, _ = mip_allS(samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
#
#this_value_fixQ = mip_fixQ0(Q_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)




    

    