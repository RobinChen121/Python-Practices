#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:16:34 2024

@author: zhenchen

@Python version: 3.10

@disp:  solve multi-period newsvendor by stochastic dual dynamic programming (SDDP);

build the sddp with object oriented classes, aiming to make it more flexible
for different problems;
    
    
"""

from gurobipy import Model
from enum import Enum
import scipy.stats as st
import numpy as np

class Direction(Enum):
    """
    objective direction of the model
    """
    
    MIN = "MIN"
    MAX = "MAX"


# first build single product, do not consider too complex at first
class GenerateSampleScenaro:
    """
    genearate samples or scenarios for a given distribution
    
    """
    
    def __init__(self, parameters, distribution_type: str, sample_nums: list[int], trunQuantile: float):
        self.parameters = parameters # parameter valus of the distribution at all stages
        self.distribution_type = distribution_type
        self.T = len(parameters)
        self.sample_nums = sample_nums
        self.trunQuantile = trunQuantile
        
        
    def generate_sample(self):
        """
        

        Returns
        -------
        sample details for each stage

        """
        arr = [[0 for i in range(self.sample_nums[t])] for t in range(self.T)]
        if self.distribution_type == 'poisson':
            for t in range(self.T):
                for i in range(sample_nums[t]):
                    # np.random.seed(10000)
                    rand_p = np.random.uniform(self.trunQuantile*i/self.sample_nums[t], self.trunQuantile*(i+1)/self.sample_nums[t])
                    arr[t][i] = st.poisson.ppf(rand_p, self.parameters[t])
            return arr
    
    

class MSP:
    """
    multi stage programming model
    """
    
    def __init__(self, T: int, obj_direction = 'min'):
        self.T = T # number of stages
        self.obj_direction = obj_direction # 1 means objecitve minimize, otherwise maximize
        self.models = self.set_models()
                
    def set_models(self):
        """
        set up the linear programing model for each stage

        Returns
        -------
        gurobi model for each period

        """
        models = [object for t in range(T)]
        for t in range(self.T):
            name = 'LP_model ' + 't = ' + str(t)
            models[t] = Model(name)
            models[t].ModelSense =  1 if self.obj_direction == 'min' else -1
        return models
            



ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20]
distribution_type = 'poisson'
T = len(mean_demands)
obj_direction = Direction.MIN  # 'minimize' or 'maximize'
model_type = 'LP' # ‘LP' or 'MIP'

trunQuantile = 0.9999 # truncated quantile when generating samples
sample_num = 10 # sample number in each stage
sample_nums = [sample_num for t in range(T)] 

gs = GenerateSampleScenaro(mean_demands, distribution_type, sample_nums, trunQuantile)
sample_detail = gs.generate_sample() 

pass


if __name__ == '__main__':
    newsvendor = MSP(T)
    for t in range(T):
        
        
        m = newsvendor.models[t]
        q = m.addVar(vtype = 'C', obj = vari_cost, name = 'q')
        I = m.addVar(vtype = 'C', obj = unit_hold_cost, name = 'I')
        B = m.addVar(vtype = 'C', obj = unit_hold_cost, name = 'B')
        name = 'LP_model ' + 't = ' + str(t)
        name = name + '.lp'
        m.write(name)
        