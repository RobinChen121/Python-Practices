# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:47:25 2020

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
     scenario programming version for order-loan situation,
     4 periods, 5 scenarios in the each period, running time is 50.15s, value is 33999;
    
"""

from gurobipy import *
from gurobipy import LinExpr
from gurobipy import GRB
from gurobipy import Model
import time
import product
import check_outsample
from scenario_mip import mip_fixQ0


# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 30000

T = 6
overhead_cost = [2000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
N = len(ini_I)
delay_length = 2
discount_rate = 0.003
B = 10000  # total quantity of order loan
r0 = 0.015  # loan rate
M = 10000

# # tree 3
demand_scenarios = [[[134, 28, 46], [246, 58, 58], [87, 24, 43]],
                    [[481, 317, 259], [608, 311, 309], [134, 181, 121]]]
demand_possibility = [[0.103, 0.476, 0.421], [0.266, 0.34, 0.394]]

#tree 2
demand_scenarios = [[[134, 32, 54], [246, 59, 56], [88, 37, 17]], [[345, 269, 481], [341, 302, 611], [156, 123, 184]]]
demand_possibility = [[0.102, 0.694, 0.204], [0.185, 0.556, 0.259]]

# # # tree 1
# demand_scenarios = [[[133, 30, 49], [246, 58, 57], [87, 39, 20]], [[291, 468, 268], [597, 322, 293], [123, 124, 177]]]
# demand_possibility = [[0.1, 0.598, 0.302], [0.286, 0.318, 0.396]]

# demand_scenarios = [[[133.6453,29.4853,40.3997,58.0380,32.7425], [246.6858,56.1574,57.1013,63.2265,58.2849], [30.5152,35.5840,16.1261,83.1273,35.6119]], \
#                     [[548.5844,294.1882,336.2068,183.5184,352.6249], [702.1755,206.3259,378.2353,276.9209,467.1965], [145.1143, 202.3169, 138.8466, 95.8576, 155.0488]]]
# demand_possibility = [[0.1022, 0.2323, 0.1419, 0.1295, 0.3941], [ 0.1098,0.1309,0.5212,0.1199, 0.1181]]


for i in range(2):
        for j in range(3):
            for k in range(3):
                demand_scenarios[i][j][k] = 2 * demand_scenarios[i][j][k]
# overhead_cost = [2 * i for i in overhead_cost]               
                
K = len(demand_possibility[0])  # scenario number in a period
S = K ** T  # total scenario number

# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
for t in range(T):
    slices = round(S * (1 / K) ** (t + 1))  # number of scenario from a node in a period 每个阶段一个结点发出的情景数
    slice_num = round(K ** (t + 1))  # total number of nodes in a period，每一个阶段的结点数
    for i in range(slice_num):
        for j in range(slices * i, slices * (i + 1)):
            for k in range(slices * i, slices * (i + 1)):
                scenarioLink[t][j][k] = 1

# set values for scenario probabilities
scenario_permulations = product.product(range(K), T)
scenario_probs = [0 for s in range(S)]
for s in range(S):
    index = scenario_permulations[s][0]
    scenario_probs[s] = demand_possibility[0][index]
    for i in range(1, len(scenario_permulations[s])):
        index = scenario_permulations[s][i]
        index2 = booming_demand[i]
        scenario_probs[s] = scenario_probs[s] * demand_possibility[index2][index]

# Q0 = [0, 171, 0]
# value = mip_fixQ0(Q0, demand_scenarios, scenarioLink, scenario_probs, scenario_permulations, booming_demand, B, delay_length, prices, vari_costs, overhead_cost, ini_cash, ini_I, discount_rate, r0)
# print(value)

tic = time.time()
try:
    # Create a new model
    m = Model("self-cash-mip")

    # Create variables
    Q = [[[m.addVar(vtype=GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in
        range(T)]  # ordering quantity in each period for each product
    I = [[[m.addVar(vtype=GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in
        range(T)]  # end-of-period inventory in each period for each product
    delta = [[[m.addVar(vtype=GRB.BINARY) for s in range(S)] for n in range(N)] for t in
            range(T)]  # whether lost-sale not occurs
    g = [[[m.addVar(vtype=GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in
        range(T)]  # order-loan quantity in each period for each product

    C = [[LinExpr() for s in range(S)] for t in range(T)]  # LinExpr, end-of-period cash in each period
    R = [[[LinExpr() for s in range(S)] for n in range(N)] for t in
        range(T + delay_length)]  # LinExpr, revenue for each product in each period

    # revenue expression
    for s in range(S):
        for n in range(N):
            for t in range(T + delay_length):
                if t < delay_length:
                    R[t][n][s] = prices[n] * g[t][n][s]
                else:
                    if t == delay_length:
                        R[t][n][s] = prices[n] * (ini_I[n] + Q[t - delay_length][n][s] - I[t - delay_length][n][s] -
                                                  g[t - delay_length][n][s] - g[t - delay_length][n][s] * (
                                                          1 + r0) ** delay_length)
                    elif t < T:
                        R[t][n][s] = prices[n] * (
                                g[t][n][s] + I[t - delay_length - 1][n][s] + Q[t - delay_length][n][s] -
                                I[t - delay_length][n][s] - g[t - delay_length][n][s] - g[t - delay_length][n][s] * (1 + r0) ** delay_length)
                    else:
                        R[t][n][s] = prices[n] * (
                                I[t - delay_length - 1][n][s] + Q[t - delay_length][n][s] - I[t - delay_length][n][
                            s] - g[t - delay_length][n][s] - g[t - delay_length][n][s] * (1 + r0) ** delay_length)

    m.update()
    # cash flow   
    revenue_total = [[LinExpr() for t in range(T)] for s in range(S)]
    vari_costs_total = [[LinExpr() for t in range(T)] for s in range(S)]
    expect_revenue_total = [LinExpr() for t in range(T)]
    expect_vari_costs_total = [LinExpr() for t in range(T)]
    for s in range(S):
        for t in range(T):
            revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
            vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)])
            try:
                if t == 0:
                    C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]
                else:
                    C[t][s] = C[t - 1][s] + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]
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
                discounted_cash[s] = discounted_cash[s] + R[T + k][n][s] / (1 + discount_rate) ** (k + 1)
    final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(S)])
    expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(S)])

    # Set objective
    m.update()
    m.setObjective(final_cash, GRB.MAXIMIZE)

    # Add constraints
    # inventory flow    
    for s in range(S):
        for n in range(N):
            for t in range(T):
                index = scenario_permulations[s][t]
                index2 = booming_demand[t]
                if t == 0:
                    m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] + (
                            1 - delta[t][n][s]) * M)
                    m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] - (
                            1 - delta[t][n][s]) * M)
                    m.addConstr(ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] <= delta[t][n][s] * M - 0.1)
                    m.addConstr(ini_I[n] + Q[t][n][s] >= demand_scenarios[index2][n][index] - (1 - delta[t][n][s]) * M)
                else:
                    try:
                        m.addConstr(I[t][n][s] <= I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] + (
                                1 - delta[t][n][s]) * M)
                        m.addConstr(I[t][n][s] >= I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] - (
                                1 - delta[t][n][s]) * M)
                        m.addConstr(I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] <= delta[t][n][
                            s] * M - 0.1)
                        m.addConstr(I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] >= -(1 -delta[t][n][
                            s]) * M + 0.1)
                    except:
                        print(n)
                m.addConstr(I[t][n][s] <= delta[t][n][s] * M)

            #    m.computeIIS() # this function is only for infeasible model
    #    m.write("model.ilp")

    # cash constraint
    for s in range(S):
        for t in range(T):
            if t == 0:
                m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t])  # cash constaints
            else:
                m.addConstr(C[t - 1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t])  # cash constaints

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

    # non-anticipativity 
    # s1 与 s 的顺序没啥影响       
    # no need for R, C
    for t in range(T):
        for n in range(N):
            for s in range(S):
                if t > 0 and t < T - 1:
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t+1][n][s1] for s1 in range(S)]) == \
                            Q[t+1][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * I[t][n][s1] for s1 in range(S)]) == \
                            I[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * delta[t][n][s1] for s1 in range(S)]) == \
                            delta[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
    
    # first-stage decision
    for s in range(S-1):
        for n in range(N):
            m.addConstr(Q[0][n][s] == Q[0][n][s+1])
            
    # solve
    m.update()
    m.optimize()
    print('')
    print('final expected value is: %g' % m.objVal)
    Q0 = [Q[0][0][0].X, Q[0][1][0].X, Q[0][2][0].X]
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

toc = time.time()
time_pass = toc - tic
print('running time is %.2f' % time_pass)
print('first stage decision: ')


value = m.objVal

Q0=[0.0, 400, 0.0]
out_sample_value = check_outsample.check_value(Q0, B, ini_I, prices, vari_costs, overhead_cost, scenarioLink, scenario_permulations,\
                ini_cash, K, S, N, T, booming_demand, delay_length, discount_rate, r0, M, scenario_probs)

print(Q0)
print('out of sample value2 is %.2f' % out_sample_value)
print('final value is %.2f' % value)



