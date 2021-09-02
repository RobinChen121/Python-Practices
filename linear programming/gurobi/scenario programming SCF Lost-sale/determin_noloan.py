# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:14:24 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
    the mip model for no finance situation when demand are deterministic
    
    lost sales quantity is not a decision variable
    
"""


from gurobipy import *

# parameter values

mean_demands12 = [[[46.5, 77, 38], [338, 389, 144]] for i in range(6)]
booming_demand = [0, 0, 0, 0, 1, 1]
overhead_cost = [2000 for t in range(6)]
ini_cash = 20000
T = 6
delay_length = 2


def mip(mean_demands, T, booming_demand, ini_cash, overhead_cost, delay_length, B, r0, discount_rate):
    ini_I = [0, 0, 0]
    prices = [189, 144, 239]
    vari_costs = [140, 70, 150]

    N = len(ini_I)
    M = 10000

    try:
        # Create a new model
        m = Model("self-cash-mip")

        # Create variables
        Q = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(N)] for n in range(T)] # ordering quantity in each period for each product
        I = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
        delta = [[m.addVar(vtype = GRB.BINARY) for t in range(N)] for n in range(T)] # whether lost-sale not occurs

        C = [LinExpr() for t in range(T)] # LinExpr, end-of-period cash in each period
        R = [[LinExpr() for n in range(N)] for t in range(T + delay_length)] # LinExpr, revenue for each product in each period


        # revenue expression
        for n in range(N):
            for t in range(T + delay_length):
                if t < delay_length:
                    R[t][n]= LinExpr(0)
                else:
                    if t == delay_length:
                        R[t][n]= prices[n] * (ini_I[n] + Q[t-delay_length][n] - I[t-delay_length][n])
                    else:
                        R[t][n]= prices[n] * (I[t-delay_length-1][n] + Q[t-delay_length][n] - I[t-delay_length][n])

        # cash flow
        revenue_total = [LinExpr() for t in range(T)]
        vari_costs_total = [LinExpr() for t in range(T)]
        for t in range(T):
            revenue_total[t] = sum([R[t][n]for n in range(N)])
            vari_costs_total[t] = sum([vari_costs[n] * Q[t][n]for n in range(N)])
            if t == 0:
                C[t] = ini_cash + revenue_total[t] - vari_costs_total[t] - overhead_cost[t] + B
            else:
                C[t] = C[t - 1] + revenue_total[t] - vari_costs_total[t]- overhead_cost[t]

        # objective function
        discounted_cash = LinExpr(0)
        for n in range(N):
            for k in range(delay_length):
                discounted_cash = LinExpr(discounted_cash + R[T+k][n] / (1 + discount_rate)**(k+1))
        final_cash = LinExpr(C[T - 1] + discounted_cash) - B * (1+r0)**T

        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)

        # Add constraints
        # inventory flow
        for n in range(N):
            for t in range(T):
                index2 = booming_demand[t]
                if t == 0:
                    m.addConstr(I[t][n]<= ini_I[n] + Q[t][n]- mean_demands[t][index2][n]+ (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n]>= ini_I[n] + Q[t][n]- mean_demands[t][index2][n]- (1 - delta[t][n]) * M)
                    m.addConstr(ini_I[n] + Q[t][n]- mean_demands[t][index2][n] <= delta[t][n]* M -0.1)
                    m.addConstr(ini_I[n] + Q[t][n]>= mean_demands[t][index2][n] - (1 -delta[t][n])* M)
                else:
                    m.addConstr(I[t][n]<= I[t-1][n]+ Q[t][n]- mean_demands[t][index2][n] + (1 - delta[t][n]) * M)
                    m.addConstr(I[t][n]>= I[t-1][n] + Q[t][n]- mean_demands[t][index2][n] - (1 - delta[t][n]) * M)
                    m.addConstr(I[t-1][n] + Q[t][n]- mean_demands[t][index2][n] <= delta[t][n]* M -0.1)
                    m.addConstr(I[t-1][n] + Q[t][n]>= mean_demands[t][index2][n] - (1 -delta[t][n])* M)
                m.addConstr(I[t][n]<= delta[t][n]* M)

        # cash constraint
        for t in range(T):
            if t == 0:
                m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n]for n in range(N)]) + overhead_cost[t]) # cash constaints
            else:
                m.addConstr(C[t - 1] >= sum([vari_costs[n] * Q[t][n]for n in range(N)]) + overhead_cost[t]) # cash constaints

        # non-negavtivety of I_t
        for n in range(N):
            for t in range(T):
                m.addConstr(I[t][n]>= 0)

        # solve
        m.optimize()
        print('')

        # output
        print('*********************************')
        print('ordering quantity Q:')
        for n in range(N):
            print('item %d:' % n)
            for t in range(T):
                print('%.1f' % Q[t][n].X, end = ' ')
            print('')
        print('*********************************')
        Qv = [[0 for n in range(N)] for t in range(T)]
        for n in range(N):
            for t in range(T):
                Qv[t][n] = Q[t][n].X

        print('end-of-period inventory I:')
        for n in range(N):
            print('item %d:' % n)
            for t in range(T):
                print('%.1f' % I[t][n].X, end = ' ')
            print('')
        print('*********************************')

        print('values of delta:')
        for n in range(N):
            print('item %d:' % n)
            for t in range(T):
                print('%.1f' % delta[t][n].X, end = ' ')
            print('')
        print('*********************************')


        print('revenue R:')
        for n in range(N):
            print('item %d:' % n)
            for t in range(T):
                print('%.1f' % R[t][n].getValue(), end = ' ')
            print('')
        print('*********************************')

        print('total revenue in each period:')
        for t in range(T):
            print('%.1f' % revenue_total[t].getValue(), end = ' ')
        print('\n')

        print('total vari costs in each period:')
        for t in range(T):
            print('%.1f' % vari_costs_total[t].getValue(), end = ' ')
        print('\n')

        if not isinstance(discounted_cash, int):
            print('totoal discounted cash: ')
            print('%.1f\n' % discounted_cash.getValue())

        print('end-of-period cash C:')
        for t in range(T):
            print('%.1f' % C[t].getValue(), end = ' ')
        print('\n')

        print('Obj: %g' % m.objVal)
        Q0 = [Qv[0][0], Qv[0][1], Qv[0][2]]
        return Q0, m.objVal

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

B = 10000
r0 = 0.015
discount_rate = 0.003
mip(mean_demands12, T, booming_demand, ini_cash, overhead_cost, delay_length, B, r0, discount_rate)


