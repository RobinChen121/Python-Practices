"""
3  # @File  : scenario_reduction.py
4  # @Author: Chen Zhen
5  # @Date  : 2020/10/11
6  # @Desc  :  reduce the number of scenarios based on the paper: A two stage stochastic programming model for 
               lot-sizing and scheduling under uncertainty (2016) in CIE.
    
             
"""

import numpy as np
import math
import select_orderloan


# Python function to f.write permutations of a given list
def product(args, repeat):
    pools = [tuple(args)] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return result


def distance(index1, index2, boomingIndex):
    k = boomingIndex
    global demand_scenarios
    item_num = len(demand_scenarios[0])
    set1 = [0 for i in range(item_num)]
    set2 = [0 for i in range(item_num)]
    for i in range(item_num):
        set1[i] = demand_scenarios[k][i][index1]
        set2[i] = demand_scenarios[k][i][index2]


# tree 1
demand_scenarios = [[[134, 17, 40], [246, 62, 57], [84, 58, 28]], [[345, 269, 481], [341, 302, 611], [156, 123, 184]]]
demand_possibility = [[0.103, 0.383, 0.514], [0.185, 0.556, 0.259]]
# # tree 2
# demand_scenarios = [[[133, 30, 49], [246, 58, 57], [87, 39, 20]], [[291, 468, 268], [597, 322, 293], [123, 124, 177]]]
# demand_possibility = [[0.102, 0.598, 0.3], [0.286, 0.318, 0.396]]
# # tree 3
# demand_scenarios = [[[47, 58, 133], [57, 58, 246], [44, 25, 86]], [[249, 314, 472], [316, 296, 596], [125, 178, 123]]]
# demand_possibility = [[0.38, 0.517, 0.103], [0.327, 0.385, 0.289]]

booming_demand = [0, 0, 0, 0, 1, 1]
scenario_num_need = 9  # scenario number after reducing
scenario_selected = []  # index of selected scenario

T = 2
delay_length = 1
K = len(demand_possibility[0])  # scenario number in a period
S = K ** T  # total scenario number
item_num = len(demand_scenarios[0])

# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
for t in range(T):
    slices = round(S * (1 / K) ** (t + 1))  # number of scenario in a slice
    slice_num = round(K ** (t + 1))  # total number of slices
    for i in range(slice_num):
        for j in range(slices * i, slices * (i + 1)):
            for k in range(slices * i, slices * (i + 1)):
                scenarioLink[t][j][k] = 1

# set values for scenario probabilities
scenario_permutations = product(range(K), T)
scenario_probs = [0 for s in range(S)]
for s in range(S):
    index = scenario_permutations[s][0]
    scenario_probs[s] = demand_possibility[booming_demand[0]][index]
    for i in range(1, len(scenario_permutations[s])):
        index = scenario_permutations[s][i]
        scenario_probs[s] = scenario_probs[s] * demand_possibility[booming_demand[i]][index]

K = 1
J = range(S)  # index for unselected scenarios
d = [[0 for s in range(S)] for s in range(S)]
for i in range(S):
    for j in range(i, S):
        for k in range(len(scenario_permutations[0])):
            index1 = scenario_permutations[i][k]
            index2 = scenario_permutations[j][k]
            boomingIndex = booming_demand[k]
            d[i][j] += (scenario_permutations[i][k] - scenario_permutations[j][
                k]) ** 2  # distance(index1, index2, boomingIndex)
        d[i][j] = math.sqrt(d[i][j])
        d[j][i] = d[i][j]
while K <= scenario_num_need:
    if K == 1:
        wd = [0 for s in range(S)]
        for i in range(S):
            for j in range(S):
                wd[i] += scenario_probs[j] * d[j][i]
        l = np.argmin(wd)
        scenario_selected.append(l)
        J = np.delete(J, l, axis=0)
    else:
        m = len(J)
        for i in J:
            for j in J:
                d[j][i] = min(d[j][i], d[j][l])
        wd = [0 for i in range(m)]
        index = 0
        for i in J:
            for j in J:
                wd[index] += scenario_probs[j] * d[j][i]
            index = index + 1
        try:
            l = np.argmin(wd)
        except:
            print(wd)
        scenario_selected.append(J[l])
        J = np.delete(J, l, axis=0)
    K = K + 1

print('scenario index:')
print(scenario_selected)
print('')
print('%d selected demand scenario index for the 1st item:' % scenario_num_need)
for i in scenario_selected:
    indexes = scenario_permutations[i]
    scenario = []
    for j in indexes:
        scenario.append(demand_scenarios[booming_demand[j]][0][j])
    print(scenario)
print('')
# print('%d selected demand scenario index for the 2 item:' % scenario_num_need)
# for i in scenario_selected:
#     indexes = scenario_permutations[i]
#     scenario = []
#     for j in indexes:
#         scenario.append(demand_scenarios[booming_demand[j]][1][j])
#     print(scenario)
# print('')
# print('%d selected demand scenario index for the 3 item:' % scenario_num_need)
# for i in scenario_selected:
#     indexes = scenario_permutations[i]
#     scenario = []
#     for j in indexes:
#         scenario.append(demand_scenarios[booming_demand[j]][2][j])
#     print(scenario)
scenario_selected.sort()
select_orderloan.select_mip(scenario_selected, demand_scenarios, demand_possibility, booming_demand, T, delay_length)
