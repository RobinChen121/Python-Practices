""" 
# @File  : EVPI-orderloan.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/11/4
# @Desc  :  compute the EVPI for the scenario-tree model

"""
import product
import determin_orderLoan
import math

# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 20000

T = 6
overhead_cost = [4000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
N = len(ini_I)
delay_length = 2
discount_rate = 0.01
B = 10000  # total quantity of order loan
r0 = 0.015  # loan rate
M = 10000

# # tree 3
demand_scenarios = [[[134, 28, 46], [246, 58, 58], [87, 24, 43]],
                    [[481, 317, 259], [608, 311, 309], [134, 181, 121]]]
demand_possibility = [[0.103, 0.476, 0.421], [0.266, 0.34, 0.394]]
# tree 2
# demand_scenarios = [[[134, 32, 54], [246, 59, 56], [88, 37, 17]], [[345, 269, 481], [341, 302, 611], [156, 123, 184]]]
# demand_possibility = [[0.102, 0.694, 0.204], [0.185, 0.556, 0.259]]
#tree 1
demand_scenarios = [[[133, 30, 49], [246, 58, 57], [87, 39, 20]], [[291, 468, 268], [597, 322, 293], [123, 124, 177]]]
demand_possibility = [[0.1, 0.598, 0.302], [0.286, 0.318, 0.396]]

K = len(demand_possibility[0])  # scenario number in a period
S = K ** T  # total scenario number

# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
for t in range(T):
    slices = round(S * (1 / K) ** (t + 1))  # number of scenario in a slice
    slice_num = round(K ** (t + 1))  # totoal number of slices
    for i in range(slice_num):
        for j in range(slices * i, slices * (i + 1)):
            for k in range(slices * i, slices * (i + 1)):
                scenarioLink[t][j][k] = 1

# set values for scenario probabilities
scenario_permulations = product.product(range(K), T)
scenario_probs = [0 for s in range(S)]
scenario_values = [0 for s in range(S)]
for s in range(S):
    index = scenario_permulations[s][0]
    scenario_probs[s] = demand_possibility[0][index]
    for i in range(1, len(scenario_permulations[s])):
        index = scenario_permulations[s][i]
        index2 = booming_demand[i]
        scenario_probs[s] = scenario_probs[s] * demand_possibility[index2][index]

for s in range(S):
    mean_demand = [[[0 for i in range(N)] for j in range(2)] for t in range(T)]
    for n in range(N):
        for t in range(T):
            index = scenario_permulations[s][t]
            index2 = booming_demand[t]
            mean_demand[t][index2][n] = demand_scenarios[index2][n][index]
    print(mean_demand)
    _, _, scenario_values[s] = determin_orderLoan.mip(mean_demand, T, booming_demand, ini_cash, overhead_cost, delay_length)

final_value = 0
for s in range(S):
    final_value = final_value + scenario_values[s] * scenario_probs[s]

mean_demands12 = [[[46.5, 77, 38], [338, 389, 144]] for t in range(T)]
Qv, gv, _ = determin_orderLoan.mip(mean_demands12, T, booming_demand, ini_cash, overhead_cost, delay_length) # the solution of the deterministic model

I = [[[0 for s in range(S)] for n in range(N)] for t in range(T)]
R = [[[0 for s in range(S)] for n in range(N)] for t in range(T + delay_length)]

revenue_total = [[0 for t in range(T)] for s in range(S)]
vari_costs_total = [[0 for t in range(T)] for s in range(S)]
expect_revenue_total = [0 for t in range(T)]
expect_vari_costs_total = [0 for t in range(T)]
C = [[0 for s in range(S)] for t in range(T)]
Q = [[0 for n in range(N)] for t in range(T)]

# get inventory flow, cash flow period by period
for s in range(S):
    for n in range(N):
        for t in range(T):
            index = scenario_permulations[s][t]
            index2 = booming_demand[t]
            if t == 0:
                Qbound = max(0, ini_cash / vari_costs[n])
                Q[t][n] = min(Qv[t][n], Qbound)
            else:
                Qbound = max(0, C[t - 1][s] / vari_costs[n])
                Q[t][n] = min(Qv[t][n], Qbound)

            if t == 0:
                I[t][n][s] = max(ini_I[n] + Q[t][n] - demand_scenarios[index2][index][n], 0)
            else:
                I[t][n][s] = max(I[t - 1][n][s] + Q[t][n] - demand_scenarios[index2][index][n], 0)

            if t < delay_length:
                R[t][n][s] = prices[n] * gv[t][n]
            else:
                if t == delay_length:
                    Qbound = max(0, ini_cash / vari_costs[n])
                    Q[t - delay_length][n] = min(Qv[t - delay_length][n], Qbound)
                    R[t][n][s] = prices[n] * (ini_I[n] + Q[t - delay_length][n] - I[t - delay_length][n][s] -
                                              gv[t - delay_length][n] - gv[t - delay_length][n] * (1 + r0) ** delay_length)
                elif t < T:
                    Qbound = max(0, C[t - delay_length - 1][s] / vari_costs[n])
                    Q[t - delay_length][n] = min(Qv[t - delay_length][n], Qbound)
                    R[t][n][s] = prices[n] * (
                            gv[t][n] + I[t - delay_length - 1][n][s] + Q[t - delay_length][n] -
                            I[t - delay_length][n][s] - gv[t - delay_length][n] - gv[t - delay_length][n] * (1 + r0) ** delay_length)
                else:
                    Qbound = max(0, C[t - delay_length - 1][s] / vari_costs[n])
                    Q[t - delay_length][n] = min(Qv[t - delay_length][n], Qbound)
                    R[t][n][s] = prices[n] * (
                            I[t - delay_length - 1][n][s] + Q[t - delay_length][n]- I[t - delay_length][n][s]
                            - gv[t - delay_length][n] - gv[t - delay_length][n] * (1 + r0) ** delay_length)

            revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
            vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n] for n in range(N)])
            if t == 0:
                C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]
            else:
                C[t][s] = C[t - 1][s] + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]

for t in range(T):
    expect_vari_costs_total[t] = sum([vari_costs_total[s][t] * scenario_probs[s] for s in range(S)])
    expect_revenue_total[t] = sum([revenue_total[s][t] * scenario_probs[s] for s in range(S)])

# objective function
discounted_cash = [0 for s in range(S)]
for s in range(S):
    for n in range(N):
        for k in range(delay_length):
            discounted_cash[s] = discounted_cash[s] + R[T + k][n][s] / (1 + discount_rate) ** (k + 1)
expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(S)])
final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(S)])

print('value for solution of determin model in scenarios DV: %.2f' % final_cash)
print('value for solution of perfect information value PV: %.2f' % final_value)