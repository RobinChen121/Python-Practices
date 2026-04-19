"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 19/04/2026, 12:15
@Desc:

"""

import math
import time
from functools import lru_cache

# --------------------------
# Poisson functions
# --------------------------


def poisson_cdf(k, lam):
    cumulative = 0.0
    term = math.exp(-lam)  # P(X=0)
    for i in range(k + 1):
        cumulative += term
        if i < k:
            term *= lam / (i + 1)
    return cumulative


def poisson_pmf(k, lam):
    if k < 0 or lam < 0:
        return 0.0
    if k == 0 and lam == 0:
        return 1.0

    logp = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(logp)


def poisson_quantile(p, lam):
    low = 0
    high = max(100, int(3 * lam))
    while low < high:
        mid = (low + high) // 2
        if poisson_cdf(mid, lam) < p:
            low = mid + 1
        else:
            high = mid
    return low


# --------------------------
# PMF truncation
# --------------------------


def get_pmf_poisson(demands, q):
    pmf = []
    for lam in demands:
        ub = poisson_quantile(q, lam)
        lb = poisson_quantile(1 - q, lam)

        support = []
        for d in range(lb, ub + 1):
            prob = poisson_pmf(d, lam) / (2 * q - 1)
            support.append((d, prob))

        pmf.append(support)
    return pmf


# --------------------------
# State
# --------------------------


class State:
    def __init__(self, period, inventory):
        self.period = period
        self.inventory = inventory

    def __hash__(self):
        return hash((self.period, self.inventory))

    def __eq__(self, other):
        return self.period == other.period and self.inventory == other.inventory


# --------------------------
# Newsvendor DP
# --------------------------


class NewsvendorDP:
    def __init__(
        self,
        T,
        capacity,
        step_size,
        fix_cost,
        var_cost,
        hold_cost,
        penalty_cost,
        max_I,
        min_I,
        pmf,
    ):

        self.T = T
        self.capacity = capacity
        self.step_size = step_size

        self.fix_cost = fix_cost
        self.var_cost = var_cost
        self.hold_cost = hold_cost
        self.penalty_cost = penalty_cost

        self.max_I = max_I
        self.min_I = min_I

        self.pmf = pmf

        self.cache_actions = {}
        self.cache_values = {}

    def feasible_actions(self):
        Q = int(self.capacity / self.step_size)
        return [i * self.step_size for i in range(Q)]

    def transition(self, state, action, demand):
        next_I = state.inventory + action - demand
        next_I = max(self.min_I, min(self.max_I, next_I))
        return State(state.period + 1, next_I)

    def immediate_cost(self, state, action, demand):
        fix = self.fix_cost if action > 0 else 0.0
        vari = action * self.var_cost

        next_I = state.inventory + action - demand
        next_I = max(self.min_I, min(self.max_I, next_I))

        hold = max(self.hold_cost * next_I, 0.0)
        penalty = max(-self.penalty_cost * next_I, 0.0)

        return fix + vari + hold + penalty

    @lru_cache(maxsize=None)
    def recursion(self, state):
        if state in self.cache_values:
            return self.cache_values[state]

        best_val = float("inf")
        best_q = 0.0

        for action in self.feasible_actions():
            val = 0.0

            for demand, prob in self.pmf[state.period - 1]:
                val += prob * self.immediate_cost(state, action, demand)

                if state.period < self.T:
                    new_state = self.transition(state, action, demand)
                    val += prob * self.recursion(new_state)

            if val < best_val:
                best_val = val
                best_q = action

        self.cache_actions[state] = best_q
        self.cache_values[state] = best_val

        return best_val


# --------------------------
# main
# --------------------------


def main():
    T = 40
    mean_demand = 20.0
    demands = [mean_demand] * T

    capacity = 150.0
    step_size = 1.0

    fix_cost = 0.0
    var_cost = 1.0
    hold_cost = 2.0
    penalty_cost = 10.0

    trunc_q = 0.9999
    max_I = 100.0
    min_I = -100.0

    pmf = get_pmf_poisson(demands, trunc_q)

    model = NewsvendorDP(
        T,
        capacity,
        step_size,
        fix_cost,
        var_cost,
        hold_cost,
        penalty_cost,
        max_I,
        min_I,
        pmf,
    )

    s0 = State(1, 0.0)

    t0 = time.time()
    val = model.recursion(s0)
    t1 = time.time()

    print(f"planning horizon = {T}")
    print(f"runtime of Python = {t1 - t0:.4f} s")
    print(f"optimal value = {val}")


if __name__ == "__main__":
    main()
