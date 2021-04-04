""" 
# @File  : check_outsample.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/11/8
# @Desc  :  when checking out of sample stability, solution values Q, I, delta are used

"""


# check values for another scenario tree
def check_value(Q, I, delta, g, ini_I, prices, vari_costs, overhead_cost, scenarioLink, scenario_permulations,\
                ini_cash, K, S, N, T, booming_demand, delay_length, discount_rate, r0, M, scenario_probs):
    T = len(Q)
    N = len(prices)

    # 不同情境树中，用不到具体的需求数值了，只用到了情境概率

    # tree 3
    demand_scenarios = [[[134, 28, 46], [246, 58, 58], [87, 24, 43]],
                        [[481, 317, 259], [608, 311, 309], [134, 181, 121]]]
    demand_possibility = [[0.103, 0.476, 0.421], [0.266, 0.34, 0.394]]

    # tree 2
    # demand_scenarios = [[[134, 32, 54], [246, 59, 56], [88, 37, 17]],
    #                     [[345, 269, 481], [341, 302, 611], [156, 123, 184]]]
    # demand_possibility = [[0.102, 0.694, 0.204], [0.185, 0.556, 0.259]]
    #
    # # # tree 1
    demand_scenarios = [[[133, 30, 49], [246, 58, 57], [87, 39, 20]], [[291, 468, 268], [597, 322, 293], [123, 124, 177]]]
    demand_possibility = [[0.1, 0.598, 0.302], [0.286, 0.318, 0.396]]
    
    demand_scenarios = [[[133.6453,29.4853,40.3997,58.0380,32.7425], [246.6858,56.1574,57.1013,63.2265,58.2849], [30.5152,35.5840,16.1261,83.1273,35.6119]], \
                    [[548.5844,294.1882,336.2068,183.5184,352.6249], [702.1755,206.3259,378.2353,276.9209,467.1965], [145.1143, 202.3169, 138.8466, 95.8576, 155.0488]]]
    demand_possibility = [[0.1022, 0.2323, 0.1419, 0.1295, 0.3941], [ 0.1098,0.1309,0.5212,0.1199, 0.1181]]

    C = [[0 for s in range(S)] for t in range(T)]  # LinExpr, end-of-period cash in each period
    R = [[[0 for s in range(S)] for n in range(N)] for t in
         range(T + delay_length)]  # LinExpr, revenue for each product in each period

    for s in range(S):
        index = scenario_permulations[s][0]
        scenario_probs[s] = demand_possibility[booming_demand[0]][index]
        for i in range(1, len(scenario_permulations[s])):
            index = scenario_permulations[s][i]
            index2 = booming_demand[i]
            scenario_probs[s] = scenario_probs[s] * demand_possibility[index2][index]

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
                                I[t - delay_length][n][s] - g[t - delay_length][n][s] - g[t - delay_length][n][
                                    s] * (1 + r0) ** delay_length)
                    else:
                        R[t][n][s] = prices[n] * (
                                I[t - delay_length - 1][n][s] + Q[t - delay_length][n][s] - I[t - delay_length][n][s]
                                - g[t - delay_length][n][s] - g[t - delay_length][n][s] * (1 + r0) ** delay_length)

    revenue_total = [[0 for t in range(T)] for s in range(S)]
    vari_costs_total = [[0 for t in range(T)] for s in range(S)]
    for s in range(S):
        for t in range(T):
            revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
            vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)])
            if t == 0:
                C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]
            else:
                C[t][s] = C[t - 1][s] + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost[t]

    discounted_cash = [0 for s in range(S)]
    for s in range(S):
        for n in range(N):
            for k in range(delay_length):
                discounted_cash[s] = discounted_cash[s] + R[T + k][n][s] / (1 + discount_rate) ** (k + 1)
    final_cash = sum([scenario_probs[s] * (C[T - 1][s] + discounted_cash[s]) for s in range(S)])
    return final_cash

