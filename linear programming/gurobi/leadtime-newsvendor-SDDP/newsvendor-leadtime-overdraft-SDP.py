#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:11:39 2024

@author: zhenchen

@disp:  
    
for 3 periods [10, 20, 10], python running about 26s, while java 0s;
for 4 periods [10, 20, 10, 20], solution 215.48, python running more than 4 hours and can't get a solution, while java 31s
for 4 periods [10, 10, 10, 10], solution 26.67, python running more than 1249s, while java 29s    
"""

import scipy.stats as sp
from functools import lru_cache
import time


class State:
    def __init__(self, t: int, iniInventory: float, iniCash: float, preQ: float):
        self.t = t
        self.iniInventory = iniInventory
        self.preQ = preQ
        self.iniCash = iniCash

    def __str__(self):
        return "t = " + str(self.t) + " " + "，I = " + str(self.iniInventory)\
                      + 'cash = ' + str(self.iniCash) + ", preQ = " + str(self.preQ)

    def __hash__(self):
        return hash(str(self.t) + str(self.iniInventory) + str(self.iniCash) + str(self.preQ))

    def __eq__(self, other):
        return self.t == other.t and self.iniInventory == other.iniInventory\
            and self.iniCash == other.iniCash and self.preQ == other.preQ
  
class StochasticInventory:
    def __init__(self, B: float, price: float, overheadCosts: list, K: float, v: float, h: float, pai: float, \
                 gamma: float, meanD: list, tq: float, r1: float, r2: float, r3:float, U: float):
        self.capacity = B
        self.price = price
        self.overheadCosts = overheadCosts
        self.fixOrderCost = K
        self.variOrderCost = v
        self.holdCost = h
        self.penaCost = pai
        self.demands = meanD
        self.truncationQ = tq
        self.pmf = self.get_pmf()
        self.max_inventory = 200
        self.min_inventory = 0
        self.max_cash = 500
        self.min_cash = -500
        self.r1 = r1      
        self.r2 = r2
        self.r3 = r3
        self.U = U
        self.gamma = gamma
        self.cache_actions = {}


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
        nextInventory = state.iniInventory + state.preQ - demand
        nextInventory = self.max_inventory if self.max_inventory < nextInventory else nextInventory
        nextInventory = self.min_inventory if self.min_inventory > nextInventory else nextInventory
        nextCash = state.iniCash + self.imme_value(state, action, demand)
        nextCash =  self.max_cash if self.max_cash < nextCash else nextCash
        nextCash =  self.min_cash if self.min_cash > nextCash else nextCash
        return State(state.t + 1, nextInventory, nextCash, action)

    def imme_value(self, state:State, action, demand):
        fixCost = self.fixOrderCost if action > 0 else 0
        variCost = self.variOrderCost * action
        nextInventory = max(state.iniInventory + state.preQ - demand, 0)
        nextInventory = self.max_inventory if nextInventory > self.max_inventory else nextInventory
        nextInventory = self.min_inventory if nextInventory < self.min_inventory else nextInventory
        holdingCost = self.holdCost * max(0, nextInventory)
        penaltyCost = self.penaCost * max(0, -nextInventory)
        revenue = self.price * min(demand, state.iniInventory + state.preQ)
        cashbalanceBefore = state.iniCash - fixCost - variCost - holdingCost - self.overheadCosts[state.t - 1]
        interest = 0
        if cashbalanceBefore > 0:
            interest = cashbalanceBefore*self.r1
        elif cashbalanceBefore > - self.U:
            interest = cashbalanceBefore*self.r2
        else:
            interest = -self.U*self.r2 + (cashbalanceBefore + self.U)*self.r3
        salValue = 0
        if state.t == len(self.demands):
            salValue = self.gamma * nextInventory;
        return revenue - fixCost - variCost - holdingCost - penaltyCost + interest + salValue - self.overheadCosts[state.t - 1]
    
    # recursion
    @ lru_cache(maxsize = None)
    def f(self, state:State) -> float:
        bestQValue = -float('inf')
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
            if thisQValue > bestQValue:
                bestQValue = thisQValue
                bestQ = action

        self.cache_actions[str(state)] = bestQ
        return bestQValue


demands = [10, 10, 10, 10]
T = len(demands)
overheadCosts = [50 for t in range(T)]
capacity = 20
fixOrderCost = 0
variOderCost = 1
holdCost = 0
penaCost = 0
price = 10
unitSal = 0.5
truncationQ = 0.9999
leadtime = 1 # lead time is reflected in the state preQ
r1 = 0
r2 = 0.1
r3 = 2
limit = 500 # overdraft limit

start = time.process_time()
lot_sizing = StochasticInventory(capacity, price, overheadCosts, fixOrderCost, variOderCost, holdCost, penaCost, unitSal, demands, truncationQ, r1, r2, r3, limit)
ini_state = State(1, 0, 0, 0)
expect_total_cost = lot_sizing.f(ini_state)
print('****************************************')
print('final expected cash increment is %.2f' % expect_total_cost)
optQ = lot_sizing.cache_actions[str(ini_state)]
print('optimal Q_1 is %.2f' % optQ)
end = time.process_time()
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)