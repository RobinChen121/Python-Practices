"""
3  # @File  : scenario_reduction.py
4  # @Author: Chen Zhen
5  # @Date  : 2019/9/17
6  # @Desc  :  reduce the number of scenarios based on the paper: A two stage stochastic programming model for 
               lot-sizing and scheduling under uncertainty (2016) in CIE.
               
               demand follow non-stationary gamma distribution.
               3-item, 6 periods:
                Distribution   & Gamma & Gamma & Gamma  \\
                Scale   & 62.99 & 199.34&124.05\\
                Shape &1.30 & 1.99 &1.46 \\
                Mean &82.14 & 397.33 &181.54 \\
                Variance &5173.98 &79206.22 &22520.71 \\
                Skewness &2.06 &0.41 &1.67 \\
                Kurtosis &7.39 & 1.78 &5.37\\
               
        scenario tree 1: possibility and demand realizations
                    0.105  & 25  & 290  & 109  \\ 
                    0.341  & 58  & 365  & 90  \\ 
                    0.330  & 62  & 134  & 132  \\ 
                    0.106  & 289  & 789  & 273  \\ 
                    0.119  & 74  & 965  & 564  \\ 
                    
           in each period there are five scenarios obtained by matlab
           
           matrix size is too large, not suitable to compute in python
             
"""



import numpy as np
import math

demand_scenarios = [[25, 290, 109], [58, 365, 90], [62, 134, 132], [289, 789, 273], [74, 965, 564]]
demand_probs = [0.105, 0.341, 0.33, 0.106, 0.119]

scenario_num_need = 10  # scenario number after reducing


item_num = 3
horizon_length = 6
demand_relization_num = len(demand_scenarios)
scenario_num = demand_relization_num ** horizon_length
scenario_index = np.zeros((scenario_num, demand_relization_num))
index = 0
for i1 in range(horizon_length):
    for i2 in range(horizon_length):
        for i3 in range(horizon_length):
            for i4 in range(horizon_length):
                for i5 in range(horizon_length):
                    for i6 in range(horizon_length):
                        scenario_index[index, :] = [i1, i2, i3, i4, i5, i6]
                        index = index + 1
        
        
