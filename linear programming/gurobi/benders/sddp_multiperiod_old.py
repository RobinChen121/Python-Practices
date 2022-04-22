# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:25:47 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: test sddp in multi period newsvendor problems,
the state variables are also decision variables in the linear programming models.
    
"""


import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools
import random
from TreeStructure import get_tree_strcture, draw_tree
 
    
def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples


ini_I = 0
ini_cash = 0
price  =  6
vari_cost = 2
sal_value = 1
uni_hold_cost = 0
mean_demands = [10, 10]
sample_nums = [2, 2]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_num = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

samples_detail = [[5, 15], [5, 15]]
scenarios = list(itertools.product(*samples_detail)) 
S = round(scenario_num)
samples= random.sample(scenarios, S) # 不放回抽样，有相同是因为需求期望值小了
node_values, node_index = get_tree_strcture(samples)
# draw_tree(node_values)


z_ub = GRB.INFINITY
z_lb = -GRB.INFINITY
q = 5 # initial ordering quantity for stage 1
Q_ub = 2000 # 第二阶段的最优值，初始默认值
q_s = [[5 for t in range(T)] for s in range(S)] # initial q for stage 2 to T
# q_s 用来记录每次迭代时的给定 x，而 x 表示决策变量订货量

I = [[0.0 for t in range(T)] for s in range(S)]
cash = [[0.0 for t in range(T)] for s in range(S)]
models = [[Model() for t in range(T)] for i in range(S)] # linear master models for stage 1 to T
ExpectQ = [[models[s][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS) for t in range(T-1)] for s in range(S)] # expected Q for stage 1 to T-1
x = [[models[s][t].addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for s in range(S)] # order quantity for stage 2 to T
y = [[models[s][t].addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for s in range(S)] # realized demand

# initial setting: initial inventory and cash for each period;
# initial constaint and objective in the first iteration
for s in range(S):
    for t in range(T):           
        if t == 0: # 这个约束条件要最先添加，因为后面要取第一个约束条件更新
            models[s][t].addConstr(y[s][t] <= ini_I + q)
        else:
            models[s][t].addConstr(y[s][t] <= I[s][t-1] + q_s[s][t-1])
        if t < T-1:
            models[s][t].addConstr(ExpectQ[s][t] <= Q_ub)
        models[s][t].addConstr(y[s][t] <= scenarios[s][t]) 
            
# # initial model settings and cuts
main_m = Model() # linear model at the beginning of stage 1
ExpectQ_values = [[0.0 for t in range(T-1)] for s in range(S)]
main_x = main_m.addVar(vtype = GRB.CONTINUOUS)
mainExpectQ = main_m.addVar(vtype = GRB.CONTINUOUS)
main_m.addConstr(mainExpectQ <= Q_ub)
main_obj = -vari_cost*main_x + mainExpectQ
main_m.setObjective(main_obj, GRB.MAXIMIZE)

# 第一阶段期初的决策是x，之后每个阶段的决策变量是 y, x，T+1 阶段期初的决策变量是y
k = 1
obj_value = 3000 # initial objective value    
while True:
    last_obj_value = obj_value     
    for s in range(S): # 每次迭代更新时也更细各阶段各情景库存资金
        for t in range(T):  
            if t == 0:    # 各阶段的初始资金与库存量
                I[s][t] = max(0, ini_I + q - scenarios[s][t])
                cash[s][t] = ini_cash + price*min(ini_I + q, samples[s][t]) - uni_hold_cost*I[s][t] - vari_cost*q
            else:
                I[s][t] = max(0, I[s][t-1] + q_s[s][t-1] - samples[s][t])
                cash[s][t] = cash[s][t-1] + price*min(I[s][t-1] + q_s[s][t-1] , samples[s][t]) - uni_hold_cost*I[s][t] - vari_cost*q_s[s][t-1]      
    for t in range(T-1, -1, -1):
        obj = [0.0  for i in range(S)]
        g = [0.0 for i in range(S)]
        for s in range(S):
            if t < T-1:
                models[s][t].setObjective(-vari_cost*x[s][t] + price*y[s][t] - uni_hold_cost*(I[s][t-1] + q_s[s][t-1] - y[s][t]) + ExpectQ[s][t], GRB.MAXIMIZE)
            elif t == T-1 and t > 0:
                models[s][t].setObjective(price*y[s][t] + (sal_value - uni_hold_cost)*(I[s][t-1] + q_s[s][t-1]- y[s][t]), GRB.MAXIMIZE)
            else: # t==0 
                models[s][t].setObjective(price*y[s][t] + (sal_value - uni_hold_cost)*(ini_I + q - y[s][t]), GRB.MAXIMIZE)    

            if k > 1: # update objective and some constraints that has q
                if t < T-1: # 更新目标函数可以直接 setObjective
                    models[s][t].setObjective(-vari_cost*x[s][t] + price*y[s][t] - uni_hold_cost*(I[s][t-1] + q_s[s][t] - y[s][t]) + ExpectQ[s][t], GRB.MAXIMIZE) 
                elif t == T-1 and t > 0: 
                    models[s][t].setObjective(price*y[s][t] + (sal_value - uni_hold_cost)*(I[s][t-1] + q_s[s][t]- y[s][t]), GRB.MAXIMIZE)
                else: # t==0 
                    models[s][t].setObjective(price*y[s][t] + (sal_value - uni_hold_cost)*(ini_I + q - y[s][t]), GRB.MAXIMIZE)
                    
                c = models[s][t].getConstrs()[0] #更新约束条件
                if t == 0:
                    c.RHS = ini_I + q
                else:
                    c.RHS = I[s][t-1] + q_s[s][t]
                models[s][t].update()            
            
            models[s][t].Params.LogToConsole = 0    # 不输出求解器计算过程      
            
            models[s][t].optimize()                   
            obj[s] = models[s][t].objVal
            pi = models[s][t].getAttr(GRB.Attr.Pi)
            g[s] = pi[0]*1
            y_value = y[s][t].x
        
        if T == 1: # 需要改 avg，跟该节点后面跟的树枝有关
            avg_obj = sum(obj)/S 
            avg_g = sum(g)/S
        # 此时一个阶段的 s 已经循环完了
        if t > 0:
            last_t_node_num = len(node_index[t-1])
            for j in range(last_t_node_num):    
                node_scenario_index = node_index[t-1][j]
                node_scenario_num = len(node_scenario_index)
                sum_obj = 0
                sum_g = 0
                for k in range(node_scenario_num):
                    s_index = node_index[t-1][j][k]
                    sum_obj = sum_obj + obj[s_index]
                    sum_g = sum_g + g[s_index]
                avg_obj = sum(obj)/node_scenario_num
                avg_g = sum(g)/node_scenario_num
                    
                for k in range(node_scenario_num):
                    s_index = node_index[t-1][j][k]
                    models[s_index][t-1].addConstr(ExpectQ[s_index][t-1] >= avg_obj + avg_g*(x[s_index][t-1] - q_s[s_index][t-1])) # add cut
                    models[s_index][t-1].update()
                    # models[i][t-1].Params.LogToConsole = 0
                    # models[s_index][t-1].write('model.lp')
                    models[s_index][t-1].optimize()

                    ExpectQ_values[s_index][t-1] = ExpectQ[s_index][t-1].x
                    q_s[s_index][t-1] = x[s_index][t-1].x
        else:
            main_m.addConstr(mainExpectQ <= avg_obj + avg_g*(main_x - q)) # add cut
            main_m.update()
            main_m.Params.LogToConsole = 0
            main_m.optimize()   
             
            obj_value = main_m.objVal
            q = main_x.x
            print('')
    
    k = k + 1
    if abs(obj_value - last_obj_value) < 1e-1 and k > 4: # same effect with the above line
        break
    
print('iteration steps are %d' % k)    
print()
print('ordering quantity is %.2f' % main_x.x)
print('expected profit is %.2f' % obj_value)
    
