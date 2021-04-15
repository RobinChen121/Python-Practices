# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:18:14 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

from gurobipy import *


def mip_allS(samples, scenarioLink, scenario_probs, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    S = len(scenario_permulations)
    N = len(samples[0])
    T = len(scenario_permulations[0])
    
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
            expect_revenue_total[t] = sum([revenue_total[s][t] * scenario_probs[s] for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[s][t] * scenario_probs[s] for s in range(S)])
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(S)])
        expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(S)])
        
        
        # Add constraints
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[s][t]
                    index2 = booming_demand[t]
                    demand = samples[index2][n][index]  # be careful
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
        # careful, there is no delay_length in this constraint
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    if t == 0:
                        m.addConstr(g[t][n][s] <= ini_I[n] + Q[t][n][s]-I[t][n][s])
                    else:
                        m.addConstr(g[t][n][s] <= I[t-1][n][s]+Q[t][n][s]-I[t][n][s])
                    
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
                    
        # non-anticipativity 
        # s1 与 s 的顺序没啥影响       
        # no need for R, C
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t][n][s1] for s1 in range(S)]) == \
                                Q[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * I[t][n][s1] for s1 in range(S)]) == \
                                I[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * delta[t][n][s1] for s1 in range(S)]) == \
                                delta[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * g[t][n][s1] for s1 in range(S)]) == \
                                g[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                        
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

# SS is the total sample number
def mip_pairS(ref_range, ref_prob, samples, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    N = len(samples[0])
    T = len(scenario_permulations[0])
    S = 2
    
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
        
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = (C[T-1][0] + discounted_cash[0])*ref_prob + (C[T-1][1] + discounted_cash[1])*(1-ref_prob)
        
        
        # Add constraints
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[ref_range[s]][t]
                    index2 = booming_demand[t]
                    demand = samples[index2][n][index]  # be careful
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
        # careful, there is no delay_length in this constraint
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    if t == 0:
                        m.addConstr(g[t][n][s] <= ini_I[n] + Q[t][n][s]-I[t][n][s])
                    else:
                        m.addConstr(g[t][n][s] <= I[t-1][n][s]+Q[t][n][s]-I[t][n][s])
                    
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
                if scenario_permulations[ref_range[0]][t] == scenario_permulations[ref_range[1]][t]:
                    if t < T - 2:
                        m.addConstr(Q[t+1][n][0] == Q[t+1][n][1])
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


# for one scenario
def mip(ref_s, samples, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0):
    N = len(samples[0])
    T = len(scenario_permulations[0])
    
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
                index = scenario_permulations[ref_s][t]
                index2 = booming_demand[t]
                demand = samples[index2][n][index]  # be careful
                if t == 0:
                    m.addConstr(I[t][n] <= ini_I[n] + Q[t][n] - demand  + (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n] >= ini_I[n] + Q[t][n] - demand  - (1 - delta[t][n]) * M)
                    m.addConstr(ini_I[n] + Q[t][n] - demand  <= delta[t][n] * M - 0.1)
                    m.addConstr(ini_I[n] + Q[t][n] >= demand  - (1 - delta[t][n]) * M)
                else:
                    m.addConstr(I[t][n] <= I[t - 1][n] + Q[t][n] - demand  + (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n] >= I[t - 1][n] + Q[t][n] - demand  - (1 - delta[t][n]) * M)
                    m.addConstr(I[t - 1][n] + Q[t][n] - demand  <= delta[t][n] * M - 0.1)
                    m.addConstr(I[t - 1][n] + Q[t][n] >= demand  - (1 - delta[t][n]) * M)
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
        # careful, there is no delay_length in this constraint
        for n in range(N):
            for t in range(T):
                if t == 0:
                    m.addConstr(g[t][n] <= ini_I[n] + Q[t][n]-I[t][n])
                else:
                    m.addConstr(g[t][n] <= I[t-1][n]+Q[t][n]-I[t][n])

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
        
        
def mip_fixQ0(Q_range, samples, scenarioLink, scenario_probs, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, ro):
    S = len(scenario_permulations)
    N = len(samples[0])
    T = len(scenario_permulations[0])
    
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
    
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(S)])
        expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(S)])
        
        
        # Add constraints
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[s][t]
                    index2 = booming_demand[t]
                    demand = samples[index2][n][index]  # be careful
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                        m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    else:
                        try:
                            m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                            m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                            m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand <= delta[t][n][s]* M -0.1) 
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
        # careful, there is no delay_length in this constraint
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    if t == 0:
                        m.addConstr(g[t][n][s] <= ini_I[n] + Q[t][n][s]-I[t][n][s])
                    else:
                        m.addConstr(g[t][n][s] <= I[t-1][n][s]+Q[t][n][s]-I[t][n][s])
                    
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
                    
        # non-anticipativity 
        # s1 与 s 的顺序没啥影响       
        # no need for R, C
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t][n][s1] for s1 in range(S)]) == \
                                Q[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * I[t][n][s1] for s1 in range(S)]) == \
                                I[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * delta[t][n][s1] for s1 in range(S)]) == \
                                delta[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * g[t][n][s1] for s1 in range(S)]) == \
                                g[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                        
        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)
                           
        # solve
        m.update()
        m.optimize()

        print('') 
        print('final expected value is: %g' % m.objVal)
        return m.objVal
    
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')
