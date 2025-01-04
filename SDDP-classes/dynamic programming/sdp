#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:00:35 2024

@author: zhenchen

@Python version: 3.10

@disp:  stochastic dynamic programming to compute multi-period newsvendor problems;
    
    use @dataclass for ease of defining classes;

    parallel computing unsucessful, highly prone to make mistakes;
"""

import scipy.stats as sp
from dataclasses import dataclass
from functools import lru_cache
import time


@dataclass(frozen=True) 
class State:
    """
    state in  a period: initial inventory 
    """
    
    t: int
    iniInventory: float
    

@dataclass
class Pmf:
    """
    probability mass function for the demand distribution in each period
    """
    
    truncQuantile: float
    distribution_type: str 
    
    def get_pmf(self, distribution_parameters):
        """
        
        Parameters
        ----------
        distribution_parameters: list, may be multi dimensional
            DESCRIPTION. parameter values of the distribution
        Returns
        -------
        pmf : 3-D list
            DESCRIPTION. probability mass function for the demand in each period

        """
        if (self.distribution_type == 'poisson'):  
            mean_demands = distribution_parameters
            max_demands = [sp.poisson.ppf(self.truncQuantile, d).astype(int) for d in mean_demands]
            T = len(mean_demands)
            pmf = [[[k, sp.poisson.pmf(k, mean_demands[t])/self.truncQuantile] for k in range(max_demands[t])] for t in range(T)]
            return pmf
   
    
@dataclass(eq = False) 
class StochasticInventory:
    """
    multi period stochastic inventory model class
    
    """    
    T: int          
    capacity: float
    fixOrderCost: float
    variOrderCost: float
    holdCost: float
    penaCost: float
    truncationQ: float
    max_inventory: float
    min_inventory: float
    pmf: [[[]]]
    cache_actions = {}
       

    def get_feasible_action(self, state:State):
        """
        feasible actions for a certain state
        
        """      
        return range(self.capacity + 1)

    def state_tran(self, state:State, action, demand):
        """
        state transition function
        
        """       
        nextInventory = state.iniInventory + action - demand
        nextInventory = self.max_inventory if self.max_inventory < nextInventory else nextInventory
        nextInventory = self.min_inventory if self.min_inventory > nextInventory else nextInventory
        return State(state.t + 1, nextInventory)

    def imme_value(self, state:State, action, demand):
        """
        immediate value function
        
        """
        fixCost = self.fixOrderCost if action > 0 else 0
        variCost = self.variOrderCost * action
        nextInventory = state.iniInventory + action - demand
        nextInventory = self.max_inventory if nextInventory > self.max_inventory else nextInventory
        nextInventory = self.min_inventory if nextInventory < self.min_inventory else nextInventory
        holdingCost = self.holdCost * max(0, nextInventory)
        penaltyCost = self.penaCost * max(0, -nextInventory)
        return fixCost + variCost + holdingCost + penaltyCost
    
    # recursion
    @ lru_cache(maxsize = None)
    def f(self, state:State):
        """
        recursive function

        """
        bestQValue = float('inf')
        bestQ = 0
        for action in self.get_feasible_action(state):
            thisQValue = 0
            
            for randDandP in self.pmf[state.t - 1]:
                thisQValue += randDandP[1] * self.imme_value(state, action, randDandP[0])
                if state.t < T:
                    thisQValue += randDandP[1] * self.f(
                        self.state_tran(state, action, randDandP[0])
                        )
            if thisQValue < bestQValue:
                bestQValue = thisQValue
                bestQ = action
                    
        self.cache_actions[str(state)] = bestQ
        return bestQValue


demands = [10, 20, 10, 20]
distribution_type = 'poisson'
capacity = 100 # maximum ordering quantity
fixOrderCost = 0
variOderCost = 1
holdCost = 2
penaCost = 10
truncQuantile = 0.9999 # trancated quantile for the demand distribution
maxI = 500 # maximum possible inventory
minI = -300 # minimum possible inventory

pmf = Pmf(truncQuantile, distribution_type).get_pmf(demands)
T = len(demands)

if __name__ == '__main__': 
    start = time.process_time()
    model = StochasticInventory(
        T,
        capacity, fixOrderCost, variOderCost,
        holdCost, penaCost, truncQuantile,
        maxI, minI,
        pmf
        )
    
    ini_state = State(1, 0)
    expect_total_cost = model.f(ini_state)
    print('****************************************')
    print('final expected total cost is %.2f' % expect_total_cost)
    optQ = model.cache_actions[str(State(1, 0))]
    print('optimal Q_1 is %.2f' % optQ)
    end = time.process_time()
    cpu_time = end - start
    print('cpu time is %.4f s' % cpu_time)




    

        