"""
created on 2025/1/30, 09:46
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description: Markovian uncertainty in constraint coefficients
Initial wealth is 55k, target value in the end of period is 80k; surplus reward is 1, shortage penalty is 4.

Extensive result is -24.693
SDDP result is -24.692 after 400 iterations

"""

from msm import MSP
from solver_detail import SDDP, Extensive
from evaluation import Evaluation,EvaluationTrue
import numpy


def g(random_state, size):
    a = numpy.empty([size,T,2])
    a[:, 0, :] = [[1.2, 1.06]]
    for t_ in range(1, T):
        noise = random_state.multivariate_normal(
            mean = [1.0, 1.0],
            cov = [[0.1, 0], [0, 0.01]],
            size = size,
        )
        a[:, t_, :] = 0.1 * a[:, t_ - 1, :] + noise
    return a

T = 4
AssetMgt = MSP(T = T, sense = -1, bound = 1e5, outputFlag = 0)
AssetMgt.add_MC_uncertainty_continuous(g)
for t in range(T):
    m = AssetMgt[t]
    now, past = m.addStateVars(2, lb = 0, obj = 0)
    if t == 0:
        m.addConstr(now[0] + now[1] == 55)
    if t in [1, 2]:
        m.addConstr(past[0] + past[1] == now[0] + now[1],
            uncertainty_dependent = {past[0]: 0, past[1]: 1})
    if t == 3:
        y = m.addVar(obj = 1)
        w = m.addVar(obj = - 4)
        m.addConstr(past[0] + past[1] - y + w == 80,
            uncertainty_dependent = {past[0]: 0, past[1]: 1})
AssetMgt.discretize(
    n_Markov_states = 25,
    n_sample_paths = 10000,
)
Extensive(AssetMgt).solve()
SDDP(AssetMgt).solve(max_iterations = 400)
# result = Evaluation(AssetMgt)
# result.run(n_simulations=1000)
# resultTrue = EvaluationTrue(AssetMgt)
# resultTrue.run(n_simulations=1000)
