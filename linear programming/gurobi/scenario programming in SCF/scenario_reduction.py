"""
3  # @File  : scenario_reduction.py
4  # @Author: Chen Zhen
5  # @Date  : 2019/9/17
6  # @Desc  :  reduce the number of scenarios based on the paper: A two stage stochastic programming model for 
               lot-sizing and scheduling under uncertainty (2016) in CIE.
    
             
"""



import numpy as np
import math

# Python function to f.write permutations of a given list 
def product(args, repeat):
    pools = [tuple(args)] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result


#tree 1
demand_scenarios = [[[134, 17, 40], [246, 62, 57], [84, 58, 28]], [[345, 269, 481], [341, 302, 611], [156, 123, 184]]]
demand_possibility = [[0.103, 0.383, 0.514], [0.185, 0.556, 0.259]] 
scenario_num_need = 10  # scenario number after reducing
scenario_selected = [] # index of selected scenario

T = 6
booming_demand = [0, 0, 0, 0, 1, 1]
K = len(demand_possibility) # scenario number in a period
S =  K ** T # total scenario number


# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
for t in range(T):
    slices = round(S * (1 / K)**(t+1)) # number of scenario in a slice
    slice_num = round(K**(t+1))       # totoal number of slices
    for i in range(slice_num):
        for j in range(slices * i, slices * (i + 1)):
            for k in range(slices * i, slices * (i + 1)):
                scenarioLink[t][j][k] = 1
        
# set values for scenario probabilities
scenario_permulations = product(range(K), T)  
scenario_probs = [0 for s in range(S)]
for s in range(S):
        index = scenario_permulations[s][0]
        scenario_probs[s] = demand_possibility[booming_demand[0]][index]
        for i in range(1, len(scenario_permulations[s])):
            index = scenario_permulations[s][i]
            index2 = booming_demand[i]
            scenario_probs[s] = scenario_probs[s] * demand_possibility[index2][index] 

K = 1
J = np.arange(S) # index for unselected scenarios
d = [[0 for s in range(S)] for s in range(S)]
for i in range(S):
    for j in range(i, S):
        for k in range(len(scenario_permulations[0])):
            d[i][j] += (scenario_permulations[i][k] - scenario_permulations[j][k])**2
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
        J = np.delete(J, l, axis = 0)
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
                
        l = np.argmin(wd)
        scenario_selected.append(J[l])
        J = np.delete(J, l, axis = 0)     
    K = K + 1

print('scenario index:')    
print(scenario_selected)
print('')
print('%d selected demand scenario index in each period:' % scenario_num_need)
for i in scenario_selected:
    indexes = scenario_permulations[i]
    scenario = []
    for j in indexes:
        scenario.append(demand_scenarios[booming_demand[j]][j][0])
    print(scenario)
    

        
