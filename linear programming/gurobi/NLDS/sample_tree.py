#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:33:30 2023

@author: zhenchen

@disp:  
    
    
"""

import numpy as np
import scipy.stats as st

def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
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

def get_tree_strcture(samples):
    T = len(samples[0])
    N = len(samples)
    node_values = [[] for t in range(T)]
    node_index = [[] for t in range(T)] # this is the wanted value
    for t in range(T):
        node_num = 0
        if t == 0:           
            for i in range(N):           
                if samples[i][t] not in node_values[t]:
                    node_values[t].append(samples[i][t]) 
                    node_index[t].append([])
                    node_index[t][node_num].append(i)
                    node_num = node_num + 1
                else:
                    temp_m = len(node_values[t])
                    for j in range(temp_m): # should revise
                        if samples[i][t] == node_values[t][j]:
                            node_index[t][j].append(i)
                            break
        else:
            lastNodeNum = len(node_index[t-1])
            for i in range(lastNodeNum):
                child_num = len(node_index[t-1][i])
                node_values[t].append([])
                for j in range(child_num):
                    index = node_index[t-1][i][j]
                    if samples[index][t] not in node_values[t][i]:
                        node_values[t][i].append(samples[index][t]) 
                        node_index[t].append([])
                        node_index[t][node_num].append(index)
                        node_num = node_num + 1
                    else:
                        temp_m = len(node_values[t][i]) #2
                        for k in range(temp_m): 
                            if samples[index][t] == node_values[t][i][k]:
                                node_index[t][k].append(index)
                                break
                    
    return node_values, node_index