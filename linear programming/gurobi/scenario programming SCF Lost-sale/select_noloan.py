""" 
# @File  : select-noloan.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/11/19
# @Desc  : 

"""


import product
import math
from gurobipy import *


def select_mip(scenario_selected, demand_scenarios, demand_possibility, booming_demand, T, delay_length):
    K = len(demand_possibility[0])  # scenario number in a period
    M = len(scenario_selected)  # total selected scenario number
    S = K ** T  # total scenario number
    N = len(demand_scenarios[0])  # number of items

    # parameter values
    ini_I = [0, 0, 0]
    prices = [189, 144, 239]
    vari_costs = [140, 70, 150]
    ini_cash = 20000
    overhead_cost = [2 * i for i in overhead_cost]
    discount_rate = 0.003
    B = 10000  # total quantity of order loan
    ro = 0.015  # loan rate
    MM = 10000

    scenario_permulations = product.product(range(K), T)
    scenario_select_detail = [[0 for t in range(T)] for i in range(M)]
    index = 0
    for i in scenario_selected:
        scenario_select_detail[index] = scenario_permulations[i]
        index = index + 1

    # set values for scenario links: whether scenario i links with scenario j in period t
    # should be checked and revised
    scenarioLink = [[[0 for s in range(M)] for s in range(M)] for t in range(T)]
    for t in range(T):
        for i in range(M):
            for j in range(M):
                if t == 0:
                    if scenario_select_detail[i][t] == scenario_select_detail[j][t]:
                        scenarioLink[t][i][j] = 1
                else:
                    if scenarioLink[t - 1][i][j] == 1 and \
                            scenario_select_detail[i][t] == scenario_select_detail[j][t]:
                        scenarioLink[t][i][j] = 1

    # set values for scenario probabilities
    scenario_probs = [0 for s in range(M)]
    for s in range(M):
        index = scenario_permulations[scenario_selected[s]][0]
        scenario_probs[s] = demand_possibility[booming_demand[0]][index]
        for i in range(1, len(scenario_permulations[scenario_selected[s]])):
            index = scenario_permulations[scenario_selected[s]][i]
            scenario_probs[s] = scenario_probs[s] * demand_possibility[booming_demand[i]][index]

    scenario_probs_all = [0 for s in range(S)]
    for s in range(S):
        index = scenario_permulations[s][0]
        scenario_probs_all[s] = demand_possibility[booming_demand[0]][index]
        for i in range(1, len(scenario_permulations[s])):
            index = scenario_permulations[s][i]
            scenario_probs_all[s] = scenario_probs_all[s] * demand_possibility[booming_demand[i]][index]

    # add probability
    d = [[0 for s in range(S)] for s in range(S)]
    for i in range(S):
        for j in range(i, S):
            for k in range(len(scenario_permulations[0])):
                d[i][j] += (scenario_permulations[i][k] - scenario_permulations[j][k]) ** 2
            d[i][j] = math.sqrt(d[i][j])
            d[j][i] = d[i][j]

    for i in range(S):
        if i not in scenario_selected:
            min_d = 1000000
            min_index = 0
            for j in range(M):
                if d[i][scenario_selected[j]] < min_d:
                    min_d = d[i][scenario_selected[j]]
                    min_index = j
            scenario_probs[min_index] = scenario_probs[min_index] + scenario_probs_all[i]

    try:
        # Create a new model
        m = Model("select-scenario-mip")

        # Create variables
        Q = [[[m.addVar(vtype=GRB.CONTINUOUS) for s in range(M)] for n in range(N)] for t in
             range(T)]  # ordering quantity in each period for each product
        I = [[[m.addVar(vtype=GRB.CONTINUOUS) for s in range(M)] for n in range(N)] for t in
             range(T)]  # end-of-period inventory in each period for each product
        delta = [[[m.addVar(vtype=GRB.BINARY) for s in range(M)] for n in range(N)] for t in
                 range(T)]  # whether lost-sale not occurs

        C = [[LinExpr() for s in range(M)] for t in range(T)]  # LinExpr, end-of-period cash in each period
        R = [[[LinExpr() for s in range(M)] for n in range(N)] for t in
             range(T + delay_length)]  # LinExpr, revenue for each product in each period

        # revenue expression
        for s in range(M):
            for n in range(N):
                for t in range(T + delay_length):
                    if t < delay_length:
                        R[t][n][s] = LinExpr(0)
                    else:
                        if t == delay_length:
                            R[t][n][s] = prices[n] * (ini_I[n] + Q[t - delay_length][n][s] - I[t - delay_length][n][s])
                        else:
                            R[t][n][s] = prices[n] * (
                                    I[t - delay_length - 1][n][s] + Q[t - delay_length][n][s] - I[t - delay_length][n][s])

        m.update()
        # cash flow
        revenue_total = [[LinExpr() for t in range(T)] for s in range(M)]
        vari_costs_total = [[LinExpr() for t in range(T)] for s in range(M)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(M):
            for t in range(T):
                revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)])
                if t == 0:
                    C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]
                else:
                    C[t][s] = C[t - 1][s] + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]

        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[s][t] * scenario_probs[s] for s in range(M)])
            expect_vari_costs_total[t] = sum([vari_costs_total[s][t] * scenario_probs[s] for s in range(M)])

        m.update()

        # objective function
        discounted_cash = [LinExpr() for s in range(M)]
        for s in range(M):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T + k][n][s] / (1 + discount_rate) ** (k + 1)
        final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(M)])
        expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(M)])

        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)

        # Add constraints
        # inventory flow
        for s in range(M):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[s][t]
                    index2 = booming_demand[t]
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] + (
                                1 - delta[t][n][s]) * MM)
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] - (
                                1 - delta[t][n][s]) * MM)
                        m.addConstr(
                            ini_I[n] + Q[t][n][s] - demand_scenarios[index2][n][index] <= delta[t][n][s] * MM - 0.1)
                        m.addConstr(
                            ini_I[n] + Q[t][n][s] >= demand_scenarios[index2][n][index] - (1 - delta[t][n][s]) * MM)
                    else:
                        m.addConstr(
                            I[t][n][s] <= I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] + (
                                    1 - delta[t][n][s]) * MM)
                        m.addConstr(
                            I[t][n][s] >= I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] - (
                                    1 - delta[t][n][s]) * MM)
                        m.addConstr(I[t - 1][n][s] + Q[t][n][s] - demand_scenarios[index2][n][index] <= delta[t][n][
                            s] * MM - 0.1)
                        m.addConstr(I[t - 1][n][s] + Q[t][n][s] >= demand_scenarios[index2][n][index] - (
                                1 - delta[t][n][s]) * MM)
                    m.addConstr(I[t][n][s] <= delta[t][n][s] * MM)

        #    m.computeIIS() # this function is only for infeasible model
        #    m.write("model.ilp")

        # cash constraint
        for s in range(M):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[
                        t])  # cash constaints
                else:
                    m.addConstr(C[t - 1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[
                        t])  # cash constaints

        # non-negativity of I_t
        for s in range(M):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)


        # non-anticipativity
        # s1 与 s 的顺序没啥影响
        # no need for I, R, C
        for t in range(T):
            for n in range(N):
                for s in range(M):
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t][n][s1] for s1 in range(M)]) == \
                                Q[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * I[t][n][s1] for s1 in range(M)]) == \
                                I[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
                    m.addConstr(
                        sum([scenarioLink[t][s1][s] * scenario_probs[s1] * delta[t][n][s1] for s1 in range(M)]) == \
                        delta[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
        
        # first-stage decision
        for s in range(M-1):
            for n in range(N):
                m.addConstr(Q[0][n][s] == Q[0][n][s+1])
        
        # solve
        m.update()
        m.optimize()
        print('')

        # output in txt files
        Qv = [[[0 for s in range(M)] for n in range(N)] for t in
              range(T)]  # ordering quantity in each period for each product
        Iv = [[[0 for s in range(M)] for n in range(N)] for t in
              range(T)]  # end-of-period inventory in each period for each product
        deltav = [[[0 for s in range(M)] for n in range(N)] for t in range(T)]  # whether lost-sale not occurs
        gv = [[[0 for s in range(M)] for n in range(N)] for t in
              range(T)]  # order-loan quantity in each period for each product
        with open('results.txt', 'w') as f:
            f.write('*********************************\n')
            f.write('ordering quantity Q in each scenario:\n')
            for s in range(M):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % Q[t][n][s].X)
                        Qv[t][n][s] = Q[t][n][s].X
                    f.write('\n')
                f.write('\n')
            f.write(
                '***************************************************************************************************************\n')

            f.write('end-of-period inventory I in each scenario:\n')
            for s in range(M):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % I[t][n][s].X)
                        Iv[t][n][s] = I[t][n][s].X
                    f.write('\n')
                f.write('\n')
            f.write('***************************************************************\n')

            f.write('not lost-sale delta in each scenario:\n')
            for s in range(M):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % delta[t][n][s].X)
                        deltav[t][n][s] = delta[t][n][s].X
                    f.write('\n')
                f.write('\n')
            f.write('*********************************\n')

            f.write('revenue R in each scenario:\n')
            for s in range(M):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % R[t][n][s].getValue())
                    f.write('\n')
                f.write('\n')
            f.write('*********************************\n')

            f.write('discounted cash in each scenario:\n')
            for s in range(M):
                f.write('S%d: ' % s)
                f.write('%.1f ' % discounted_cash[s].getValue())
                f.write('\n')
            f.write('*********************************\n')

            f.write('end-of-period cash C in each scenario:\n')
            for s in range(M):
                f.write('S%d:\n' % s)
                for t in range(T):
                    f.write('%.1f ' % C[t][s].getValue())
                f.write('  %.3f:\n' % scenario_probs[s])
                f.write('\n')
            f.write('*********************************\n')

            f.write('expectd Revenue in each period:\n')
            for t in range(T):
                f.write('%.1f ' % expect_revenue_total[t].getValue())
            f.write('\n')
            f.write('varicosts in each period:\n')
            for t in range(T):
                f.write('%.1f ' % expect_vari_costs_total[t].getValue())
            f.write('\n')
            f.write('expected end-of-period cash in each period:\n')
            f.write('%.1f ' % ini_cash)
            expect_cash = [LinExpr() for t in range(T)]
            for t in range(T):
                expect_cash[t] = sum([C[t][s] * scenario_probs[s] for s in range(M)])
                f.write('%.1f ' % expect_cash[t].getValue())
            f.write('\n')
            f.write('final expected discounted cash is: %g\n' % expect_discounted_cash.getValue())
            f.write('final expected cash is: %g' % final_cash.getValue())
        print('final expected value is: %g' % m.objVal)
        return m.objVal
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')
