# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:30:17 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: solving the SAA for specific scenarios
    
"""

from gurobipy import *
from gurobipy import LinExpr
from gurobipy import GRB
from gurobipy import Model
import math
import numpy as np
import scipy.stats as st
from math import exp



def mip(ref_s,B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0):
    N = len(ref_s[0])
    T = len(ref_s)
    
    M = 100000
    
    try:
        # Create a new model
        m = Model("Order-loan-mip-refs")

        # Create variables
        Q = [[m.addVar(vtype=GRB.CONTINUOUS) for n in range(N)] for t in
             range(T)]  # ordering quantity in each period for each product
        I = [[m.addVar(vtype=GRB.CONTINUOUS) for n in range(N)] for t in
             range(T)]  # end-of-period inventory in each period for each product
        delta = [[m.addVar(vtype=GRB.BINARY) for n in range(N)] for t in range(T)]  # whether lost-sale not occurs
        g = [[m.addVar(vtype=GRB.CONTINUOUS) for n in range(N)] for t in
             range(T)]  # order-loan quantity in each period for each product

        C = [LinExpr() for t in range(T)]  # LinExpr, end-of-period cash in each period
        R = [[LinExpr() for n in range(N)] for t in
             range(T + delay_length)]  # LinExpr, revenue for each product in each period

        # revenue expression
        for n in range(N):
            for t in range(T + delay_length):
                if t < delay_length:
                    R[t][n] = prices[n] * g[t][n]
                elif t == delay_length:
                    R[t][n] = prices[n] * (
                                ini_I[n] + Q[t - delay_length][n] - I[t - delay_length][n] - g[t - delay_length][n]
                                - g[t - delay_length][n] * (1 + r0) ** delay_length)
                elif t < T:
                    R[t][n] = prices[n] * (
                            g[t][n] + I[t - delay_length - 1][n] + Q[t - delay_length][n] - I[t - delay_length][n] -
                            g[t - delay_length][n] - g[t - delay_length][n] * (1 + r0) ** delay_length)
                else:
                    R[t][n] = prices[n] * (
                            I[t - delay_length - 1][n] + Q[t - delay_length][n] - I[t - delay_length][n] -
                            g[t - delay_length][n] - g[t - delay_length][n] * (1 + r0) ** delay_length)

        # cash flow
        revenue_total = [LinExpr() for t in range(T)]
        vari_costs_total = [LinExpr() for t in range(T)]
        for t in range(T):
            revenue_total[t] = sum([R[t][n] for n in range(N)])
            vari_costs_total[t] = sum([vari_costs[n] * Q[t][n] for n in range(N)])
            if t == 0:
                C[t] = ini_cash + revenue_total[t] - vari_costs_total[t] - overhead_cost[t]
            else:
                C[t] = C[t - 1] + revenue_total[t] - vari_costs_total[t] - overhead_cost[t]

        # objective function
        discounted_cash = LinExpr(0)
        for n in range(N):
            for k in range(delay_length):
                discounted_cash = LinExpr(discounted_cash + R[T + k][n] / (1 + discount_rate) ** (k + 1))
        final_cash = LinExpr(C[T - 1] + discounted_cash)

        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)

        # Add constraints
        # inventory flow
        for n in range(N):
            for t in range(T):
                if t == 0:
                    m.addConstr(I[t][n] <= ini_I[n] + Q[t][n] - ref_s[t][n] + (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n] >= ini_I[n] + Q[t][n] - ref_s[t][n] - (1 - delta[t][n]) * M)
                    m.addConstr(ini_I[n] + Q[t][n] - ref_s[t][n] <= delta[t][n] * M - 0.1)
                    m.addConstr(ini_I[n] + Q[t][n] >= ref_s[t][n] - (1 - delta[t][n]) * M)
                else:
                    m.addConstr(I[t][n] <= I[t - 1][n] + Q[t][n] - ref_s[t][n] + (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n] >= I[t - 1][n] + Q[t][n] - ref_s[t][n] - (1 - delta[t][n]) * M)
                    m.addConstr(I[t - 1][n] + Q[t][n] - ref_s[t][n] <= delta[t][n] * M - 0.1)
                    m.addConstr(I[t - 1][n] + Q[t][n] >= ref_s[t][n] - (1 - delta[t][n]) * M)
                m.addConstr(I[t][n] <= delta[t][n] * M)

        # cash constraint
        for t in range(T):
            if t == 0:
                m.addConstr(
                    ini_cash >= sum([vari_costs[n] * Q[t][n] for n in range(N)]) + overhead_cost[t])  # cash constaints
            else:
                m.addConstr(
                    C[t - 1] >= sum([vari_costs[n] * Q[t][n] for n in range(N)]) + overhead_cost[t])  # cash constaints

        # non-negavtivety of I_t
        for n in range(N):
            for t in range(T):
                m.addConstr(I[t][n] >= 0)

        # order loan quantity less than realized demand
        for n in range(N):
            for t in range(T):
                m.addConstr(g[t][n] <= I[t - delay_length - 1][n] + Q[t - delay_length][n] - I[t - delay_length][n])

        # total order loan limit
        total_loan = LinExpr()
        for n in range(N):
            for t in range(T):
                total_loan += prices[n] * g[t][n]
        m.addConstr(total_loan <= B)

        # solve
        m.optimize()
        print('')

        print('Obj: %g' % m.objVal)

        return [m.objVal, Q[0][0].X, Q[0][1].X, Q[0][2].X]
        
    
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')
        

# SS is the total sample number
def mip_pairS(ref_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    N = len(samples[0])
    T = len(samples)
    S = 2
    SS = len(scenario_permulations)
    
    M = 100000
    
    try:
        # Create a new model
        m = Model("order-loan-saa")
    
        # Create variables
    #    Q0 = [m.addVar(vtype = GRB.CONTINUOUS) for n in range(N)]
        Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)]for n in range(N)] for t in range(T)]
        I = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
        delta = [[[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(N)] for n in range(T)] # whether lost-sale not occurs
        g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
         
        C = [[LinExpr()  for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period
        R = [[[LinExpr()  for s in range(S)] for n in range(N)] for t in range(T + delay_length)]  # LinExpr, revenue for each product in each period
        
        # revenue expression  # check revenue
        for s in range(S):
            for n in range(N):
                for t in range(T + delay_length):
                    if t < delay_length:
                        R[t][n][s] = prices[n] * g[t][n][s]
                    else:
                        if t == delay_length:
                            R[t][n][s] = prices[n]*(ini_I[n]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        elif t < T:
                            R[t][n][s] = prices[n]*(g[t][n][s]+I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        else:        
                            R[t][n][s] = prices[n]*(I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
        
        m.update()
        
        # cash flow   
        revenue_total = [[LinExpr() for s in range(S)] for t in range(T)]
        vari_costs_total = [[LinExpr() for s in range(S)] for t in range(T)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(S):
            for t in range(T):
                revenue_total[t][s] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[t][s] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) # strange
                try:
                    if t == 0:
                        C[t][s] = ini_cash + revenue_total[t][s] - vari_costs_total[t][s] - overhead_cost[t]
                    else:
                        C[t][s] = C[t-1][s] + revenue_total[t][s] - vari_costs_total[t][s]- overhead_cost[t]
                except:
                    print(n)   
        
        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[t][s] / S for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[t][s] / S for s in range(S)])
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = (C[T-1][0] + discounted_cash[0])/ SS + (C[T-1][0] + discounted_cash[0])*(SS-1)/ SS
        expect_discounted_cash = sum([(discounted_cash[s])/ S for s in range(S)])
        
        
        # Add constraints
    #    for s in range(S):
    #        for n in range(N):
    #            for t in range(T):
    #                m.addConstr(Q0 == 0) 
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    demand = samples[t][n][scenario_permulations[ref_range[s]][t]]   # be careful
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                        m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    else:
                        try:
                            m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                            m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                            m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                        except:
                            print(n)
                    m.addConstr(I[t][n][s] <= delta[t][n][s] * M)         
            
        # cash constraint
        for s in range(S):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints
                else:       
                    m.addConstr(C[t - 1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints  
        
        # non-negavtivety of I_t
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)
        
         # order loan quantity less than realized demand
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(g[t][n][s] <= I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s])
                
        # total order loan limit
        total_loan = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    total_loan[s] += prices[n] * g[t][n][s]
        for s in range(S):
            m.addConstr(total_loan[s] <= B)
            
        # first-stage decision
        for s in range(S-1):
            for n in range(N):
                m.addConstr(Q[0][n][s] == Q[0][n][s+1])
        
        for n in range (N):
            for t in range(T):
                if scenario_permulations[ref_range[0]][t] == scenario_permulations[ref_range[0]][t]:
                    m.addConstr(Q[t][n][0] == Q[t][n][1])
                    m.addConstr(I[t][n][0] == I[t][n][1])
                    m.addConstr(g[t][n][0] == g[t][n][1])
                    m.addConstr(delta[t][n][0] == delta[t][n][1])
                else:
                    break
        
        
                        
        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)
                           
        # solve
        m.update()
        m.optimize()
        print('') 

        print('Obj: %g' % m.objVal)

        return m.objVal, Q[0][0][0].X, Q[0][1][0].X, Q[0][2][0].X
        
    
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')


# need revise, for one specific scenario, its value should not be too large
def mip_fixQ0(Q_range, samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    S = len(scenario_permulations)
    N = len(samples[0])
    T = len(samples)
    
    M = 100000
    
    try:
        # Create a new model
        m = Model("order-loan-saa")
    
        # Create variables
    #    Q0 = [m.addVar(vtype = GRB.CONTINUOUS) for n in range(N)]
        Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)]for n in range(N)] for t in range(T)]
        I = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
        delta = [[[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(N)] for n in range(T)] # whether lost-sale not occurs
        g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
         
        C = [[LinExpr()  for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period
        R = [[[LinExpr()  for s in range(S)] for n in range(N)] for t in range(T + delay_length)]  # LinExpr, revenue for each product in each period
        
        # revenue expression  # check revenue
        for s in range(S):
            for n in range(N):
                for t in range(T + delay_length):
                    if t < delay_length:
                        R[t][n][s] = prices[n] * g[t][n][s]
                    else:
                        if t == delay_length:
                            R[t][n][s] = prices[n]*(ini_I[n]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        elif t < T:
                            R[t][n][s] = prices[n]*(g[t][n][s]+I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        else:        
                            R[t][n][s] = prices[n]*(I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
        
        m.update()
        
        # cash flow   
        revenue_total = [[LinExpr() for s in range(S)] for t in range(T)]
        vari_costs_total = [[LinExpr() for s in range(S)] for t in range(T)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(S):
            for t in range(T):
                revenue_total[t][s] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[t][s] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) # strange
                try:
                    if t == 0:
                        C[t][s] = ini_cash + revenue_total[t][s] - vari_costs_total[t][s] - overhead_cost[t]
                    else:
                        C[t][s] = C[t-1][s] + revenue_total[t][s] - vari_costs_total[t][s]- overhead_cost[t]
                except:
                    print(n)   
        
        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[t][s] / S for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[t][s] / S for s in range(S)])
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = sum([(C[T-1][s] + discounted_cash[s])/ S for s in range(S)])
        expect_discounted_cash = sum([(discounted_cash[s])/ S for s in range(S)])
        
        
        # Add constraints
    #    for s in range(S):
    #        for n in range(N):
    #            for t in range(T):
    #                m.addConstr(Q0 == 0) 
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    demand = samples[t][n][scenario_permulations[s][t]]  # be careful
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                        m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    else:
                        try:
                            m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                            m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                            m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                        except:
                            print(n)
                    m.addConstr(I[t][n][s] <= delta[t][n][s] * M)         
            
        # cash constraint
        for s in range(S):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints
                else:       
                    m.addConstr(C[t - 1][s] +10 >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints  
        
        # non-negavtivety of I_t
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)
        
         # order loan quantity less than realized demand
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(g[t][n][s] <= I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s])
                
        # total order loan limit
        total_loan = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    total_loan[s] += prices[n] * g[t][n][s]
        for s in range(S):
            m.addConstr(total_loan[s] <= B)
            
        # first-stage decision
        for s in range(S):
            for n in range(N):
                m.addConstr(Q[0][n][s] == Q_range[n])
                    
        
                        
        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)
                           
        # solve
        m.update()
        
        m.computeIIS()
        m.feasRelax()
        
        m.optimize()
        print('') 
        print('final expected value is: %g' % m.objVal)
        
        
        
        
        return m.objVal
        
    
    
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')

def mip_allS(samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    S = len(scenario_permulations)
    N = len(samples[0])
    T = len(samples)
    
    M = 100000
    
    try:
        # Create a new model
        m = Model("order-loan-saa")
    
        # Create variables
    #    Q0 = [m.addVar(vtype = GRB.CONTINUOUS) for n in range(N)]
        Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)]for n in range(N)] for t in range(T)]
        I = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
        delta = [[[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(N)] for n in range(T)] # whether lost-sale not occurs
        g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
         
        C = [[LinExpr()  for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period
        R = [[[LinExpr()  for s in range(S)] for n in range(N)] for t in range(T + delay_length)]  # LinExpr, revenue for each product in each period
        
        # revenue expression  # check revenue
        for s in range(S):
            for n in range(N):
                for t in range(T + delay_length):
                    if t < delay_length:
                        R[t][n][s] = prices[n] * g[t][n][s]
                    else:
                        if t == delay_length:
                            R[t][n][s] = prices[n]*(ini_I[n]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        elif t < T:
                            R[t][n][s] = prices[n]*(g[t][n][s]+I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        else:        
                            R[t][n][s] = prices[n]*(I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
        
        m.update()
        
        # cash flow   
        revenue_total = [[LinExpr() for s in range(S)] for t in range(T)]
        vari_costs_total = [[LinExpr() for s in range(S)] for t in range(T)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(S):
            for t in range(T):
                revenue_total[t][s] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[t][s] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) # strange
                try:
                    if t == 0:
                        C[t][s] = ini_cash + revenue_total[t][s] - vari_costs_total[t][s] - overhead_cost[t]
                    else:
                        C[t][s] = C[t-1][s] + revenue_total[t][s] - vari_costs_total[t][s]- overhead_cost[t]
                except:
                    print(n)   
        
        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[t][s] / S for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[t][s] / S for s in range(S)])
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = sum([(C[T-1][s] + discounted_cash[s])/ S for s in range(S)])
        expect_discounted_cash = sum([(discounted_cash[s])/ S for s in range(S)])
        
        
        # Add constraints
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    demand = samples[t][n][scenario_permulations[s][t]]  # be careful
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                        m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    else:
                        try:
                            m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                            m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                            m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                        except:
                            print(n)
                    m.addConstr(I[t][n][s] <= delta[t][n][s] * M)         
            
        # cash constraint
        for s in range(S):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints
                else:       
                    m.addConstr(C[t - 1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints  
        
        # non-negavtivety of I_t
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)
        
         # order loan quantity less than realized demand
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(g[t][n][s] <= I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s])
                
        # total order loan limit
        total_loan = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    total_loan[s] += prices[n] * g[t][n][s]
        for s in range(S):
            m.addConstr(total_loan[s] <= B)
            
        # first-stage decision
        for s in range(S-1):
            for n in range(N):
                m.addConstr(Q[0][n][s] == Q[0][n][s+1])
                    
        
                        
        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)
                           
        # solve
        m.update()
        m.optimize()

        print('') 
        print('final expected value is: %g' % m.objVal)
        return m.objVal, Q[0][0][0].X, Q[0][1][0].X, Q[0][2][0].X
    
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')
        


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

  
    
## parameter values
#ini_I = [0, 0, 0]
## prices = [89, 159, 300]
## vari_costs = [70, 60, 60]
#prices = [189, 144, 239]
#vari_costs = [140, 70, 150]
#ini_cash = 22000
#
#T = 6
#overhead_cost = [2000 for t in range(T)]
#booming_demand = [0, 0, 0, 0, 1, 1]
#delay_length = 0
#discount_rate = 0.01
#B = 10000  # total quantity of order loan
#ro = 0.015  # loan rate
#M = 10000
#
#mus = [[3.66, 5.79], [4.13, 5.91], [3.54, 4.96]]
#sigmas = [[0.6, 0.26], [0.66, 0.33], [0.46, 0.18]]
##mus = [[3.66, 5.79], [4.13, 5.91]]
##sigmas = [[0.6, 0.26], [0.66, 0.33]]
#N = len(mus)
#sample_nums = [5, 5, 5, 5, 3, 3]
#trunQuantile = 1
#
#samples = generate_sample(sample_nums, trunQuantile, mus, sigmas, booming_demand[0:T])
#SS = np.prod(sample_nums[0:T]) # number of samples
#arr = []
#for t in range(T):
#    arr.append(range(sample_nums[t]))
#scenario_permulations = list(itertools.product(*arr))
#
#Q_range =[0, 0, 0]
#this_value = mip_allS(samples, scenario_permulations, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro)
#
#print(this_value)