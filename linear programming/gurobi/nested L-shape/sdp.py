#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:32:46 2023

@author: chen
@desp: stochastic dynamic programming code for multi period newsvendor problem without price
"""

import scipy.stats as sp
from functools import lru_cache
import time


class State:
    def __init__(self, t: int, iniInventory: float):
        self.t = t
        self.iniInventory = iniInventory

    def __str__(self):
        return "t = " + str(self.t) + " " + "，I = " + str(self.iniInventory)

    def __hash__(self):
        return hash(str(self.t) + str(self.iniInventory))

    def __eq__(self, other):
        return self.t == other.t and self.iniInventory == other.iniInventory

    
class StochasticInventory:
    def __init__(self, B: float, K: float, v: float, h: float, pai: float, meanD: list[float], tq: float):
        self.capacity = B
        self.fixOrderCost = K
        self.variOrderCost = v
        self.holdCost = h
        self.penaCost = pai
        self.demands = meanD
        self.truncationQ = tq
        self.pmf = self.get_pmf()
        self.max_inventory = 500
        self.min_inventory = -300
        self.cache_actions = {}

    def __str__(self):
        return 'B = %.2f\nK = %.2f\nv = %.2f\nh = %.2f\n pai = %.2f\n demands = %s' % \
               (self.capacity, self.fixOrderCost, self.variOrderCost, self.holdCost, self.penaCost, self.demands)

    def get_max_demands(self):
         max_demands = [sp.poisson.ppf(self.truncationQ, d).astype(int) for d in self.demands]
         return max_demands

    def get_pmf(self):
        max_demands = self.get_max_demands()
        T = len(self.demands)
        pmf = [[[k, sp.poisson.pmf(k,self.demands[t])/self.truncationQ] for k in range(max_demands[t])] for t in range(T)]
        return pmf

    def get_feasible_action(self, state:State):
        return range(self.capacity + 1)

    def state_tran(self, state:State, action, demand):
        nextInventory = state.iniInventory + action - demand
        nextInventory = self.max_inventory if self.max_inventory < nextInventory else nextInventory
        nextInventory = self.min_inventory if self.min_inventory > nextInventory else nextInventory
        return State(state.t + 1, nextInventory)

    def imme_value(self, state:State, action, demand):
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
    def f(self, state:State) -> float:
        bestQValue = float('inf')
        bestQ = 0
        for action in self.get_feasible_action(state):
            thisQValue = 0
            for randDandP in self.pmf[state.t - 1]:
                thisQValue += randDandP[1] * self.imme_value(state, action, randDandP[0])
                demand = randDandP[0]
                if state.t < len(self.demands):
                    if action > 0 and demand > 1:
                        pass
                    thisQValue += randDandP[1] * self.f(self.state_tran(state, action, randDandP[0]))
            if thisQValue < bestQValue:
                bestQValue = thisQValue
                bestQ = action

        self.cache_actions[str(state)] = bestQ
        return bestQValue


demands = [10, 10]
capacity = 100
fixOrderCost = 0
variOderCost = 1
holdCost = 2
penaCost = 10
truncationQ = 0.9999

start = time.process_time()
lot_sizing = StochasticInventory(capacity, fixOrderCost, variOderCost, holdCost, penaCost, demands, truncationQ)
ini_state = State(1, 0)
expect_total_cost = lot_sizing.f(ini_state)
print('****************************************')
print('final expected total cost is %.2f' % expect_total_cost)
optQ = lot_sizing.cache_actions[str(State(1, 0))]
print('optimal Q_1 is %.2f' % optQ)
end = time.process_time()
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)




    

        