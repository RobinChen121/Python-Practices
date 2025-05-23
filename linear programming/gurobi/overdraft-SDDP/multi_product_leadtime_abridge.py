#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:18:39 2024

@author: zhenchen

@disp:  business overdraft for lead time in 2 product problem;
may run several times to get a more stable result;

6 periods gurobi can not generate results when r0 is 0.01;

********************
self defined discrete distribution:

mean_demands1 =[30, 30, 30] # higher average demand vs lower average demand
mean_demands2 = [i*0.5 for i in mean_demands1] # higher average demand vs lower average demand
pk1 = [0.25, 0.5, 0.25]
pk2= pk1
xk1 = [mean_demands1[0]-10, mean_demands1[0], mean_demands1[0]+10]
xk2 = [mean_demands2[0]-5, mean_demands2[0], mean_demands2[0]+5]
ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10] # lower margin vs higher margin
MM = len(prices)
unit_salvages = [0.5* vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

SDP: final optimal cash is 91.26875 (441.57 for 0 overhead cost, 464 for 0 overdraft interest rate)
optimal order quantity in the first period is :  Q1 = 40, Q2 = 20;

no enhancement
sample numer is 5 and scenario number is 5 
planning horizon length is T = 3 
final expected total profits after 100 iteration is 98.12/141.23/23.74/97/139/153/94/43/120/181/61/145 (regenerate sample)
ordering Q1 and Q2 in the first period is 40.00 and 20.00
cpu time is 62.115 s
expected lower bound gap is 78.60
lower bound and upper bound gap is 19.89%
confidence interval for expected objective is [-3.95,  161.15];

no enhancement
sample numer is 5 and scenario number is 5 
planning horizon length is T = 3 
final expected total profits after 100 iteration is 69.84/99.54/141.23 (sampling with replacement)
ordering Q1 and Q2 in the first peiod is 20.00 and 20.00
cpu time is 60.263 s
expected lower bound gap is 66.20
lower bound and upper bound gap is 5.21%
confidence interval for expected objective is [23.44,  108.96];

SDDP with abridge:
for abridge, B=2, F=5 (take average for the 5 out of the 10 solutions of forward computing), so compute 10 sub problems in forward;

"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys

sys.path.append("..")
from tree import *


# for gamma demand
# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
mean_demands1 = [30, 30, 30]  # higher average demand vs lower average demand
mean_demands2 = [
    i * 0.5 for i in mean_demands1
]  # higher average demand vs lower average demand
# betas = [2, 0.25] # lower variance vs higher variance
# T = len(mean_demands1)

pk1 = [0.25, 0.5, 0.25]
pk2 = pk1
xk1 = [mean_demands1[0] - 10, mean_demands1[0], mean_demands1[0] + 10]
xk2 = [mean_demands2[0] - 5, mean_demands2[0], mean_demands2[0] + 5]

# cov1 = 0.25 # lower variance vs higher variance
# cov2 = 0.5
# sigmas1 = [cov1*i for i in mean_demands1]
# sigmas2 = [cov2*i for i in mean_demands2]


T = len(mean_demands1)

ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10]  # lower margin vs higher margin
MM = len(prices)
unit_salvages = [0.5 * vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

r0 = 0  # when it is 0.01, can largely slow the compuational speed
r1 = 0.1
r2 = 2  # penalty interest rate for overdraft exceeding the limit, does not affect computation time
U = 500  # overdraft limit

sample_num = 10  # change 1


# for gamma demand
# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
# mean_demands =[30, 15] # higher average demand vs lower average demand
# betas = [2, 0.25] # lower variance vs higher variance


# detailed samples in each period
trunQuantile = 0.9999  # affective to the final ordering quantity
sample_details1 = [[0 for i in range(sample_num)] for t in range(T)]
sample_details2 = [[0 for i in range(sample_num)] for t in range(T)]
for t in range(T):
    # sample_details1[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands1[t], betas[0])
    # sample_details2[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands2[t], betas[1])
    # sample_details1[t] = generate_samples(sample_num, trunQuantile, mean_demands1[t])
    # sample_details2[t] = generate_samples(sample_num, trunQuantile, mean_demands2[t])
    # sample_details1[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands1[t], sigmas1[t])
    # sample_details2[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands2[t], sigmas2[t])
    sample_details1[t] = generate_samples_discrete(sample_num, xk1, pk1)
    sample_details2[t] = generate_samples_discrete(sample_num, xk2, pk2)

# sample_details1 = [[10, 30], [10, 30], [10, 30]] # change 2
# sample_details2 = [[5, 15], [5, 15], [5, 15]]


theta_iniValue = -1000  # initial theta values (profit) in each period
m = Model()  # linear model in the first stage
# decision variable in the first stage model
q1 = m.addVar(vtype=GRB.CONTINUOUS, name="q_1")
q2 = m.addVar(vtype=GRB.CONTINUOUS, name="q_2")
W0 = m.addVar(vtype=GRB.CONTINUOUS, name="w_1^0")
W1 = m.addVar(vtype=GRB.CONTINUOUS, name="w_1^1")
W2 = m.addVar(vtype=GRB.CONTINUOUS, name="w_1^2")
theta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta_2")

m.setObjective(
    overhead_cost[0]
    + vari_costs[0] * q1
    + vari_costs[1] * q2
    + r2 * W2
    + r1 * W1
    - r0 * W0
    + theta,
    GRB.MINIMIZE,
)
m.addConstr(theta >= theta_iniValue * (T))
m.addConstr(W1 <= U)
m.addConstr(
    -vari_costs[0] * q1 - vari_costs[1] * q2 - W0 + W1 + W2
    == overhead_cost[0] - ini_cash
)


# cuts recording arrays
iter_limit = 60
time_limit = 360
N = 10  # sampled number of scenarios in forward computing, change 3
slope_stage1_1 = []
slope_stage1_2 = []
slope_stage1_3 = []
intercept_stage1 = []
slopes1 = []
slopes2 = []
slopes3 = []
intercepts = []
q1_values = []
qpre1_values = []
q2_values = []
qpre2_values = []
q1_values_bridge = []
qpre1_values_bridge = []
q2_values_bridge = []
qpre2_values_bridge = []
W0_values = []
W1_values = []
W2_values = []

iter = 0
time_pass = 0
B = 5  # number of sampled nodes in forwad subproblems for abridge
time_start = time.process_time()
# while iter < iter_num:
while iter < iter_limit:  # time_pass < time_limit:   # or
    slopes1.append([[[0 for m in range(MM)] for n in range(N)] for t in range(T)])
    slopes2.append([[0 for n in range(N)] for t in range(T)])
    slopes3.append([[[0 for m in range(MM)] for n in range(N)] for t in range(T)])
    intercepts.append([[0 for n in range(N)] for t in range(T - 1)])
    q1_values.append([[0 for n in range(N)] for t in range(T)])
    qpre1_values.append([[0 for n in range(N)] for t in range(T)])
    q2_values.append([[0 for n in range(N)] for t in range(T)])
    qpre2_values.append([[0 for n in range(N)] for t in range(T)])

    q1_values_bridge.append([[0 for n in range(B)] for t in range(T)])
    q2_values_bridge.append([[0 for n in range(B)] for t in range(T)])
    qpre1_values_bridge.append([[0 for n in range(B)] for t in range(T)])
    qpre2_values_bridge.append([[0 for n in range(B)] for t in range(T)])

    # sample_scenarios1 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[0], betas[0], T)
    # sample_scenarios2 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[1], betas[1], T)

    # sample_scenarios1 = generate_scenarios(N, sample_num, sample_details1)
    # sample_scenarios2 = generate_scenarios(N, sample_num, sample_details2)

    sample_scenarios1 = generate_scenarios_discrete(N, xk1, pk1, T)
    sample_scenarios2 = generate_scenarios_discrete(N, xk2, pk2, T)

    # sample_scenarios1 = [[10, 10, 10], [10,10, 30], [10, 30, 10], [10,30, 30],[30,10,10],[30,10,30],[30,30,10],[30,30,30]] # change 4
    # sample_scenarios2 = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5,15,15],[15,5,5], [15,5, 15], [15,15,5], [15,15,15]]

    # forward
    if iter > 0:
        m.addConstr(
            theta
            >= slope_stage1_1[-1][0] * (ini_Is[0])
            + slope_stage1_1[-1][1] * (ini_Is[1])
            + slope_stage1_2[-1]
            * (
                ini_cash
                - vari_costs[0] * q1
                - vari_costs[1] * q2
                - r1 * W1
                + r0 * W0
                - r2 * W2
            )
            + slope_stage1_3[-1][0] * q1
            + slope_stage1_3[-1][1] * q2
            + intercept_stage1[-1]
        )
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()

    # if iter == 14:
    #     m.write('iter' + str(iter+1) + '_main2.lp')
    #     m.write('iter' + str(iter+1) + '_main2.sol')
    #     pass

    q1_values[iter][0] = [q1.x for n in range(N)]
    q2_values[iter][0] = [q2.x for n in range(N)]
    q1_values_bridge[iter][0] = [q1.x for n in range(B)]
    q2_values_bridge[iter][0] = [q2.x for n in range(B)]
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    z = m.objVal
    z_values = [
        [0 for t in range(T + 1)] for n in range(N)
    ]  # for computing the feasible cost
    for n in range(N):
        z_values[n][0] = m.objVal - theta.x

    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q1_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="q1_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    q2_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="q2_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    qpre1_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="qpre1_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    qpre2_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="qpre2_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    I1_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="I1_" + str(t + 1) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T)
    ]
    I2_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="I2_" + str(t + 1) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T)
    ]
    cash_forward = [
        [
            m_forward[t][n].addVar(
                lb=-GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name="C_" + str(t + 1) + "^" + str(n + 1),
            )
            for n in range(N)
        ]
        for t in range(T)
    ]
    W0_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="W0_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    W1_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="W1_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    W2_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="W2_" + str(t + 2) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    # B is the quantity of lost sale
    B1_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="B1_" + str(t + 1) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T)
    ]
    B2_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="B2_" + str(t + 1) + "^" + str(n + 1)
            )
            for n in range(N)
        ]
        for t in range(T)
    ]
    theta_forward = [
        [
            m_forward[t][n].addVar(
                lb=-GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name="theta_" + str(t + 3) + "^" + str(n + 1),
            )
            for n in range(N)
        ]
        for t in range(T - 1)
    ]

    I1_forward_values = [[0 for n in range(N)] for t in range(T)]
    B1_forward_values = [[0 for n in range(N)] for t in range(T)]
    I2_forward_values = [[0 for n in range(N)] for t in range(T)]
    B2_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T - 1)]
    W1_forward_values = [[0 for n in range(N)] for t in range(T - 1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T - 1)]

    I1_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    B1_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    I2_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    B2_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    cash_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    W0_forward_values_bridge = [[0 for n in range(B)] for t in range(T - 1)]
    W1_forward_values_bridge = [[0 for n in range(B)] for t in range(T - 1)]
    W2_forward_values_bridge = [[0 for n in range(B)] for t in range(T - 1)]

    for t in range(T):
        for n in range(N):
            demand1 = sample_scenarios1[n][t]
            demand2 = sample_scenarios2[n][t]
            bn = int(n / (N / B))  # affect the initial inventory, cash in this stage
            pass

            if t == 0:
                m_forward[t][n].addConstr(
                    I1_forward[t][n] - B1_forward[t][n] == ini_Is[0] - demand1
                )
                m_forward[t][n].addConstr(
                    I2_forward[t][n] - B2_forward[t][n] == ini_Is[1] - demand2
                )
                m_forward[t][n].addConstr(
                    cash_forward[t][n]
                    + prices[0] * B1_forward[t][n]
                    + +prices[1] * B2_forward[t][n]
                    == ini_cash
                    - overhead_cost[t]
                    - vari_costs[0] * q1_values[-1][t][n]
                    - vari_costs[1] * q2_values[-1][t][n]
                    - r1 * W1_values[-1]
                    + r0 * W0_values[-1]
                    - r2 * W2_values[-1]
                    + prices[0] * demand1
                    + prices[1] * demand2
                )
            else:
                m_forward[t][n].addConstr(
                    I1_forward[t][n] - B1_forward[t][n]
                    == I1_forward_values_bridge[t - 1][bn]
                    + qpre1_values[-1][t - 1][bn]
                    - demand1
                )
                m_forward[t][n].addConstr(
                    I2_forward[t][n] - B2_forward[t][n]
                    == I2_forward_values_bridge[t - 1][bn]
                    + qpre2_values[-1][t - 1][bn]
                    - demand2
                )
                m_forward[t][n].addConstr(
                    cash_forward[t][n]
                    + prices[0] * B1_forward[t][n]
                    + +prices[1] * B2_forward[t][n]
                    == cash_forward_values_bridge[t - 1][bn]
                    - overhead_cost[t]
                    - vari_costs[0] * q1_values_bridge[-1][t][bn]
                    - vari_costs[1] * q2_values_bridge[-1][t][bn]
                    - r1 * W1_forward_values_bridge[t - 1][bn]
                    + r0 * W0_forward_values_bridge[t - 1][bn]
                    - r2 * W2_forward_values_bridge[t - 1][bn]
                    + prices[0] * demand1
                    + prices[1] * demand2
                )

            if t < T - 1:
                m_forward[t][n].addConstr(
                    qpre1_forward[t][n] == q1_values_bridge[-1][t][bn]
                )
                m_forward[t][n].addConstr(
                    qpre2_forward[t][n] == q2_values_bridge[-1][t][bn]
                )
            if t == T - 1:
                m_forward[t][n].setObjective(
                    -prices[0] * (demand1 - B1_forward[t][n])
                    - prices[1] * (demand2 - B2_forward[t][n])
                    - unit_salvages[0] * I1_forward[t][n]
                    - unit_salvages[1] * I2_forward[t][n],
                    GRB.MINIMIZE,
                )
            else:
                m_forward[t][n].setObjective(
                    overhead_cost[t]
                    + vari_costs[0] * q1_forward[t][n]
                    + vari_costs[1] * q2_forward[t][n]
                    - prices[0] * (demand1 - B1_forward[t][n])
                    - prices[1] * (demand2 - B2_forward[t][n])
                    + r2 * W2_forward[t][n]
                    + r1 * W1_forward[t][n]
                    - r0 * W0_forward[t][n]
                    + theta_forward[t][n],
                    GRB.MINIMIZE,
                )

                m_forward[t][n].addConstr(W1_forward[t][n] <= U)
                m_forward[t][n].addConstr(
                    cash_forward[t][n]
                    - vari_costs[0] * q1_forward[t][n]
                    - vari_costs[1] * q2_forward[t][n]
                    - W0_forward[t][n]
                    + W1_forward[t][n]
                    + W2_forward[t][n]
                    == overhead_cost[t + 1]
                )
                m_forward[t][n].addConstr(
                    theta_forward[t][n] >= theta_iniValue * (T - 1 - t)
                )

            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(B):  # N
                        m_forward[t][n].addConstr(
                            theta_forward[t][n]
                            >= slopes1[i][t][nn][0]
                            * (I1_forward[t][n] + qpre1_forward[t][n])
                            + slopes1[i][t][nn][1]
                            * (I2_forward[t][n] + qpre2_forward[t][n])
                            + slopes3[i][t][nn][0] * q1_forward[t][n]
                            + slopes3[i][t][nn][1] * q2_forward[t][n]
                            + slopes2[i][t][nn]
                            * (
                                cash_forward[t][n]
                                - vari_costs[0] * q1_forward[t][n]
                                - vari_costs[1] * q2_forward[t][n]
                                - r2 * W2_forward[t][n]
                                - r1 * W1_forward[t][n]
                                + r0 * W0_forward[t][n]
                            )
                            + intercepts[i][t][nn]
                        )

            # optimize
            m_forward[t][n].Params.LogToConsole = 0
            m_forward[t][n].optimize()
            # if iter == 1 and t == 0:
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp')
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol')
            #     pass

            I1_forward_values[t][n] = I1_forward[t][n].x
            I2_forward_values[t][n] = I2_forward[t][n].x

            B1_forward_values[t][n] = B1_forward[t][n].x
            B2_forward_values[t][n] = B2_forward[t][n].x
            cash_forward_values[t][n] = cash_forward[t][n].x

            if t < T - 1:  # for computing confidence interval
                z_values[n][t + 1] = m_forward[t][n].objVal - theta_forward[t][n].x
            else:
                z_values[n][t + 1] = m_forward[t][n].objVal

            if t < T - 1:
                q1_values[iter][t + 1][n] = q1_forward[t][n].x
                q2_values[iter][t + 1][n] = q2_forward[t][n].x
                qpre1_values[iter][t][n] = qpre1_forward[t][n].x
                qpre2_values[iter][t][n] = qpre2_forward[t][n].x
                W1_forward_values[t][n] = W1_forward[t][n].x
                W0_forward_values[t][n] = W0_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x

        for bb in range(B):
            start = int(bb * N / B)
            end = int((bb + 1) * N / B)
            I1_forward_values_bridge[t][bb] = np.mean(I1_forward_values[t][start:end])
            # if abs(I1_forward_values_bridge[t][bb] - I1_forward_values[t][start : end]) > 1:
            #     pass

            I2_forward_values_bridge[t][bb] = np.mean(I2_forward_values[t][start:end])
            B1_forward_values_bridge[t][bb] = np.mean(B1_forward_values[t][start:end])
            B2_forward_values_bridge[t][bb] = np.mean(B2_forward_values[t][start:end])
            cash_forward_values_bridge[t][bb] = np.mean(
                cash_forward_values[t][start:end]
            )
            if t < T - 1:
                q1_values_bridge[iter][t + 1][bb] = np.mean(
                    q1_values[iter][t + 1][start:end]
                )
                q2_values_bridge[iter][t + 1][bb] = np.mean(
                    q2_values[iter][t + 1][start:end]
                )
                qpre1_values_bridge[iter][t][bb] = np.mean(
                    qpre1_values[iter][t][start:end]
                )
                qpre2_values_bridge[iter][t][bb] = np.mean(
                    qpre2_values[iter][t][start:end]
                )
                W1_forward_values_bridge[t][bb] = np.mean(
                    W1_forward_values[t][start:end]
                )
                W2_forward_values_bridge[t][bb] = np.mean(
                    W2_forward_values[t][start:end]
                )
                W0_forward_values_bridge[t][bb] = np.mean(
                    W0_forward_values[t][start:end]
                )

    # backward
    m_backward = [
        [[Model() for s in range(sample_num)] for n in range(N)] for t in range(T)
    ]
    q1_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="q1_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    qpre1_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="qpre1_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    q2_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="q2_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    qpre2_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="qpre2_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]

    I1_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="I1_" + str(t + 1) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T)
    ]
    I2_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="I2_" + str(t + 1) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T)
    ]

    cash_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="C_" + str(t + 1) + "^" + str(n + 1),
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T)
    ]
    W0_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="W0_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    W1_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="W1_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]
    W2_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="W2_" + str(t + 2) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]

    # B is the quantity of lost sale
    B1_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="B1_" + str(t + 1) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T)
    ]
    B2_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="B2_" + str(t + 1) + "^" + str(n + 1)
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T)
    ]
    theta_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="theta_" + str(t + 3) + "^" + str(n + 1),
                )
                for s in range(sample_num)
            ]
            for n in range(N)
        ]
        for t in range(T - 1)
    ]

    intercept_values = [
        [[0 for s in range(sample_num)] for n in range(N)] for t in range(T)
    ]
    slope1_values = [
        [[[0 for m in range(MM)] for s in range(sample_num)] for n in range(N)]
        for t in range(T)
    ]
    slope2_values = [
        [[0 for s in range(sample_num)] for n in range(N)] for t in range(T)
    ]
    slope3_values = [
        [[[0 for m in range(MM)] for s in range(sample_num)] for n in range(N)]
        for t in range(T)
    ]

    for t in range(T - 1, -1, -1):
        demand_temp = [sample_details1[t], sample_details2[t]]
        demand_all = list(itertools.product(*demand_temp))
        # demand_all2 = [[demand_all[s][0], demand_all[s][1]] for s in range(sample_num)]
        for bn in range(B):
            S = sample_num  # should revise, should be S^2
            for s in range(S):
                demand1 = demand_all[s][0]  # sample_details1[t][s]  #
                demand2 = demand_all[s][1]  # sample_details2[t][s]  #
                # demand1 = sample_details1[t][s]
                # demand2 = sample_details2[t][s]

                if t == T - 1:
                    m_backward[t][bn][s].setObjective(
                        -prices[0] * (demand1 - B1_backward[t][bn][s])
                        - prices[1] * (demand2 - B2_backward[t][bn][s])
                        - unit_salvages[0] * I1_backward[t][bn][s]
                        - unit_salvages[1] * I2_backward[t][bn][s],
                        GRB.MINIMIZE,
                    )
                else:
                    m_backward[t][bn][s].setObjective(
                        overhead_cost[t]
                        + vari_costs[0] * q1_backward[t][bn][s]
                        + vari_costs[1] * q2_backward[t][bn][s]
                        - prices[0] * (demand1 - B1_backward[t][bn][s])
                        - prices[1] * (demand2 - B2_backward[t][bn][s])
                        + r2 * W2_backward[t][bn][s]
                        + r1 * W1_backward[t][bn][s]
                        - r0 * W0_backward[t][bn][s]
                        + theta_backward[t][bn][s],
                        GRB.MINIMIZE,
                    )
                if t == 0:
                    m_backward[t][bn][s].addConstr(
                        I1_backward[t][bn][s] - B1_backward[t][bn][s]
                        == ini_Is[0] - demand1
                    )
                    m_backward[t][bn][s].addConstr(
                        I2_backward[t][bn][s] - B2_backward[t][bn][s]
                        == ini_Is[1] - demand2
                    )
                    m_backward[t][bn][s].addConstr(
                        cash_backward[t][bn][s]
                        == ini_cash
                        - overhead_cost[t]
                        - vari_costs[0] * q1_values[-1][t][bn]
                        - vari_costs[1] * q2_values[-1][t][bn]
                        - r2 * W2_values[-1]
                        - r1 * W1_values[-1]
                        + r0 * W0_values[-1]
                        + prices[0] * (demand1 - B1_backward[t][bn][s])
                        + prices[1] * (demand2 - B2_backward[t][bn][s])
                    )
                else:
                    m_backward[t][bn][s].addConstr(
                        I1_backward[t][bn][s] - B1_backward[t][bn][s]
                        == I1_forward_values_bridge[t - 1][bn]
                        + qpre1_values_bridge[-1][t - 1][bn]
                        - demand1
                    )
                    m_backward[t][bn][s].addConstr(
                        I2_backward[t][bn][s] - B2_backward[t][bn][s]
                        == I2_forward_values_bridge[t - 1][bn]
                        + qpre2_values_bridge[-1][t - 1][bn]
                        - demand2
                    )
                    m_backward[t][bn][s].addConstr(
                        cash_backward[t][bn][s]
                        + prices[0] * B1_backward[t][bn][s]
                        + prices[1] * B2_backward[t][bn][s]
                        == cash_forward_values[t - 1][bn]
                        - overhead_cost[t]
                        - vari_costs[0] * q1_values[-1][t][bn]
                        - vari_costs[1] * q2_values[-1][t][bn]
                        - r2 * W2_forward_values[t - 1][bn]
                        - r1 * W1_forward_values[t - 1][bn]
                        + r0 * W0_forward_values[t - 1][bn]
                        + prices[0] * demand1
                        + prices[1] * demand2
                    )

                if t < T - 1:
                    m_backward[t][bn][s].addConstr(
                        qpre1_backward[t][bn][s] == q1_values[-1][t][bn]
                    )
                    m_backward[t][bn][s].addConstr(
                        qpre2_backward[t][bn][s] == q2_values[-1][t][bn]
                    )
                if t < T - 1:
                    m_backward[t][bn][s].addConstr(W1_backward[t][bn][s] <= U)
                    m_backward[t][bn][s].addConstr(
                        cash_backward[t][bn][s]
                        - vari_costs[0] * q1_backward[t][bn][s]
                        - vari_costs[1] * q2_backward[t][bn][s]
                        - W0_backward[t][bn][s]
                        + W1_backward[t][bn][s]
                        + W2_backward[t][bn][s]
                        == overhead_cost[t + 1]
                    )
                    m_backward[t][bn][s].addConstr(
                        theta_backward[t][bn][s] >= theta_iniValue * (T - 1 - t)
                    )

                # put those cuts in the back
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N):  # N
                            m_backward[t][bn][s].addConstr(
                                theta_backward[t][bn][s]
                                >= slopes1[i][t][nn][0]
                                * (I1_backward[t][bn][s] + qpre1_backward[t][bn][s])
                                + slopes1[i][t][nn][1]
                                * (I2_backward[t][bn][s] + qpre2_backward[t][bn][s])
                                + slopes3[i][t][nn][0] * q1_backward[t][bn][s]
                                + slopes3[i][t][nn][1] * q2_backward[t][bn][s]
                                + slopes2[i][t][nn]
                                * (
                                    cash_backward[t][bn][s]
                                    - vari_costs[0] * q1_backward[t][bn][s]
                                    - vari_costs[1] * q2_backward[t][bn][s]
                                    - r2 * W2_backward[t][bn][s]
                                    - r1 * W1_backward[t][bn][s]
                                    + r0 * W0_backward[t][bn][s]
                                )
                                + intercepts[i][t][nn]
                            )

                # optimize
                m_backward[t][bn][s].Params.LogToConsole = 0
                m_backward[t][bn][s].optimize()

                pi = m_backward[t][bn][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][bn][s].getAttr(GRB.Attr.RHS)

                # if iter == 8 and t == 1 and bn == 0:
                #     m_backward[t][bn][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(bn+1) + '-' + str(s+1) + '-mback.lp')
                #     m_backward[t][bn][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(bn+1) + '-' + str(s+1) + '-mabck.sol')
                #     filename = 'iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(bn+1) + '-' + str(s+1) + '-m.txt'
                #     with open(filename, 'w') as f:
                #         f.write('demand1=' +str(demand1)+'\bn')
                #         f.write('demand2=' +str(demand2)+'\bn')
                #         f.write(str(pi))
                #     pass

                num_con = len(pi)
                if t < T - 1:
                    # important
                    intercept_values[t][bn][s] += (
                        -pi[0] * demand1
                        - pi[1] * demand2
                        + pi[2] * prices[0] * demand1
                        + pi[2] * prices[1] * demand2
                        - pi[2] * overhead_cost[t]
                        - prices[0] * demand1
                        - prices[1] * demand2
                        + overhead_cost[t + 1]
                    )  # + pi[3]*U + pi[4]*overhead_cost[t+1]-pi[5]*theta_iniValue*(T-1-t)
                else:
                    intercept_values[t][bn][s] += (
                        -pi[0] * demand1
                        - pi[1] * demand2
                        + pi[2] * prices[0] * demand1
                        + pi[2] * prices[1] * demand2
                        - pi[2] * overhead_cost[t]
                        - prices[0] * demand1
                        - prices[1] * demand2
                    )
                for sk in range(5, num_con):
                    intercept_values[t][bn][s] += pi[sk] * rhs[sk]

                slope1_values[t][bn][s] = [pi[0], pi[1]]
                slope2_values[t][bn][s] = pi[2]
                if t < T - 1:
                    slope3_values[t][bn][s] = [pi[3], pi[4]]

            avg_intercept = sum(intercept_values[t][bn]) / S
            avg_slope1 = np.mean(np.array(slope1_values[t][bn]), axis=0).tolist()
            avg_slope2 = sum(slope2_values[t][bn]) / S
            avg_slope3 = np.mean(np.array(slope3_values[t][bn]), axis=0).tolist()
            if t == 0:
                slope_stage1_1.append(avg_slope1)
                slope_stage1_2.append(avg_slope2)
                slope_stage1_3.append(avg_slope3)
                intercept_stage1.append(avg_intercept)
                pass
            else:
                slopes1[-1][t - 1][bn] = avg_slope1
                slopes2[-1][t - 1][bn] = avg_slope2
                slopes3[-1][t - 1][bn] = avg_slope3
                intercepts[-1][t - 1][bn] = avg_intercept

    iter += 1
    time_pass = time.process_time() - start

time_end = time.process_time()
print("********************************************")
print("no enhancement")
print("sample numer is %d and scenario number is %d " % (sample_num, N))
print("planning horizon length is T = %d " % T)
print("final expected total profits after %d iteration is %.2f" % (iter, -z))
print(
    "ordering Q1 and Q2 in the first peiod is %.2f and %.2f"
    % (q1_values[iter - 1][0][0], q2_values[iter - 1][0][0])
)
cpu_time = time_end - time_start
print("cpu time is %.3f s" % cpu_time)
z_lb, z_ub, z_mean = compute_ub(z_values)  # for computing confidence interval
lb = -np.mean(np.sum(z_values, axis=1))
print("expected lower bound gap is %.2f" % lb)
gap2 = abs((z + lb) / z)
print("lower bound and upper bound gap is %.2f%%" % (100 * gap2))
print("confidence interval for expected objective is [%.2f,  %.2f]" % (-z_ub, -z_lb))
