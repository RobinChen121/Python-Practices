#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:23:19 2023

@author: zhen chen

@disp:  
    
    
"""

import numpy as np
import scipy.stats as st
import itertools
import random


# generate poisson distribution
def generate_samples(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples


# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
def generate_samples_gamma(sample_num, trunQuantile, mean, beta):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.gamma.ppf(rand_p, mean*beta, loc=0, scale=1/beta)
    random.shuffle(samples)
    return samples

def generate_samples_normal(sample_num, trunQuantile, mean, sigma):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.norm.ppf(rand_p, loc = mean, scale = sigma)
    random.shuffle(samples)
    return samples

def generate_samples_discrete(self, sample_num, xk, pk):
    samples =  [[0.0 for _ in range(sample_num)] for t in self.T]
    for i in range(sample_num):
        # np.random.seed(10000)
        rand_p = np.random.uniform(i/sample_num, (i+1)/sample_num)
        dist = st.rv_discrete(values=(xk, pk))
        samples[t][i] = dist.ppf(rand_p)
    random.shuffle(samples)
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


# generate for any distribution from sample details
def generate_scenarios2(scenario_num, sample_num, sample_details):
    T = len(sample_details)
    scenarios = [[0 for t in range(T)] for i in range(scenario_num)]
    for i in range(scenario_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_index = np.random.randint(0, sample_num)
            scenarios[i][t] = sample_details[t][rand_index]
        # random.shuffle(samples[i])
            
    return scenarios

# generate for Possion distribution
def generate_scenarios(scenario_num, trunQuantile, mus):
    T = len(mus)
    samples = [[0 for t in range(T)] for i in range(scenario_num)]
    for i in range(scenario_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(trunQuantile*i/scenario_num, trunQuantile*(i+1)/scenario_num)
            samples[i][t] = st.poisson.ppf(rand_p, mus[t])
        # random.shuffle(samples[i])
            
    return samples

# for self defined discrete distribution
def generate_scenarios_discrete(sample_num, xk, pk, T):
    samples = [[0 for t in range(T)] for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(i/sample_num, (i+1)/sample_num)
            dist = st.rv_discrete(values=(xk, pk))
            samples[i][t] = dist.ppf(rand_p)
        random.shuffle(samples[i])
    return samples

def generate_scenarios_gamma(sample_num, trunQuantile, mean, beta, T):
    samples = [[0 for t in range(T)] for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
            samples[i][t] = st.gamma.ppf(rand_p, mean*beta, loc=0, scale=1/beta)
        random.shuffle(samples[i])
    return samples

def generate_scenarios_normal(sample_num, trunQuantile, means, sigmas):
    T = len(means)
    samples = [[0 for t in range(T)] for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
            samples[i][t] = round(st.norm.ppf(rand_p, loc=means[t], scale=sigmas[t]))
        random.shuffle(samples[i])
    return samples

def generate_scenarios_poisson(sample_num, trunQuantile, mean, T):
    samples = [[0 for t in range(T)] for i in range(sample_num)]
    for i in range(sample_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
            samples[i][t] = st.poisson.ppf(rand_p, mean)
        random.shuffle(samples[i])
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


def compute_ub(twoDArray):
    N = len(twoDArray)
    T = len(twoDArray[0])
    z_sub_values = [0 for n in range(N)]
    for n in range(N):
        for t in range(T):
            z_sub_values[n] += twoDArray[n][t]
    
    z_mean = np.mean(z_sub_values)    
    z_std = np.std(z_sub_values, ddof = 1)
    z_ub = z_mean + 1.96*z_std/np.sqrt(N)
    z_lb = z_mean - 1.96*z_std/np.sqrt(N)
    return [z_lb, z_ub, z_mean]

mean_demand = 10
beta = 1
N = 10
trunQuantile = 0.9999   

# samples_detail is the detailed samples in each period
sample_detail = [0 for i in range(N)] 
sample_detail = generate_samples_gamma(N, trunQuantile, mean_demand, beta)

# scenarios_full = list(itertools.product(*samples_detail)) 
# sample_num = 30
# # random.seed(10000)
# sample_scenarios= generate_scenario_samples(sample_num, trunQuantile, mean_demands)
# sample_scenarios.sort() # sort to make same numbers together
# node_values, node_index = get_tree_strcture(sample_scenarios)
