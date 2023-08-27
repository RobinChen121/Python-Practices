#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:23:19 2023

@author: zhenchen

@disp:  
    
    
"""

import numpy as np
import scipy.stats as st
import itertools
import random

def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples

# get the number of elements in a list of lists
def getSizeOfNestedList(listOfElem):
    ''' Get number of elements in a nested list'''
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:  
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1    
    return count

def generate_scenario_samples(sample_num, trunQuantile, mus):
    T = len(mus)
    samples = [[0 for t in range(T)] for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
            samples[i][t] = st.poisson.ppf(rand_p, mus[t])
            
    return samples


def get_tree_strcture(samples):
    T = len(samples[0])
    N = len(samples)
    node_values = [[] for t in range(T)]
    node_index = [[] for t in range(T)] # this is the wanted value, more important than node_values
    
    for n in range(N):
        for t in range(T):
            if n == 0:
                node_values[t].append(samples[n][t])
                index = []
                index.append(n)
                node_index[t].append(index)
            else:
                if (samples[n][:t+1] != samples[n-1][:t+1]):
                    node_values[t].append(samples[n][t])
                    index = []
                    index.append(n)
                    node_index[t].append(index)
                else:
                    index = node_index[t][-1]
                    index.append(n)
                    node_index[t][-1] = index                   
    return node_values, node_index



# mean_demands = [10, 20, 10]
# T = len(mean_demands)
# sample_nums = [10 for t in range(T)]
# trunQuantile = 0.9999 
# scenario_numTotal = 1
# for i in sample_nums:
#     scenario_numTotal *= i    

# # samples_detail is the detailed samples in each period
# samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
# for t in range(T):
#     samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

# scenarios_full = list(itertools.product(*samples_detail)) 
# sample_num = 30
# # random.seed(10000)
# sample_scenarios= generate_scenario_samples(sample_num, trunQuantile, mean_demands)
# sample_scenarios.sort() # sort to make same numbers together
# node_values, node_index = get_tree_strcture(sample_scenarios)