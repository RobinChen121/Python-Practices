# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:12:20 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: compute the stochastic value of SAA model versus the dertiministic model
    
"""

import math
import numpy as np
from saa_mip_scenarios import mip_determin, mip_fixQ0, mip_allS
import time
import itertools
import scipy.stats as st
import _pickle as cPickle # save python list to files


def lognorm_ppf(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = math.exp(mu)
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
delay_length = 2
discount_rate = 0.01
B = 10000  # total quantity of order loan
r0 = 0.1  # loan rate
M = 10000

mus = [[3.66, 5.79], [4.13, 5.91], [3.54, 4.96]]
sigmas = [[0.6, 0.26], [0.66, 0.33], [0.46, 0.18]]
#mus = [[3.66, 5.79], [4.13, 5.91]]
#sigmas = [[0.6, 0.26], [0.66, 0.33]]
N = len(mus)
sample_nums = [5, 5, 5, 3, 3, 3]
trunQuantile = 0.9999

#samples = generate_sample(sample_nums, trunQuantile, mus, sigmas, booming_demand[0:T])
samples = cPickle.load(open("data1.pkl", "rb"))

S = np.prod(sample_nums[0:T])
arr = []
for t in range(T):
    arr.append(range(sample_nums[t]))
scenario_permulations = list(itertools.product(*arr))

means = [[0 for i in range(N)] for t in range(T)]
for t in range(T):
    for i in range(N):
        means[t][i] = math.exp(mus[i][booming_demand[t]] + sigmas[i][booming_demand[t]] ** 2 / 2)

value, Q1, Q2, Q3 = mip_determin(means, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
Q_range = [Q1, 50, Q3]
tic = time.time()
this_value = mip_fixQ0(Q_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
toc = time.time()
time_pass = toc - tic
print('running time is %.2f' % time_pass)

this_value_opt, _, _, _ = mip_allS(samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
print('value of deterministic model is %.2f' % this_value)
print('value of stochastic model is %.2f' % this_value_opt)
