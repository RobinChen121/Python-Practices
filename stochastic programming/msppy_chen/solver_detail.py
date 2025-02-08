"""
Created on 2025/1/10, 21:48

@author: Zhen Chen.

@Python version: 3.10

@disp:  Different classes of stochastic programming solvers.

"""
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from sm_detail import StochasticModel
from msm import MSLP
from utils.statistics import rand_int, allocate_jobs, compute_CI
from utils.logger import LoggerSDDP, LoggerEvaluation, LoggerComparison
from evaluation import Evaluation, EvaluationTrue
from collections import abc
from numpy.typing import ArrayLike
import time
import gurobipy
import numpy
import math
import multiprocessing
import numbers
import pandas
import pdb


class Extensive:
    """
    Extensive solver class.

    Can solve:

    1. small-scale stage-wise independent finite discrete risk neutral problem;

    2. small-scale Markov chain risk neutral problem.

    Parameters:
    ----------
    msp: A multi-stage stochastic liner program object.

    Attributes
    ----------
    extensive_model:
        The constructed extensive model

    solving_time:
        The time cost in solving extensive model

    construction_time:
        The time cost in constructing extensive model
    """

    def __init__(self, msp: MSLP) -> None:
        self.start = 0  # starting stage
        self.extensive_model = None
        self.MSP = msp
        self.solving_time = None
        self.construction_time = None
        self.total_time = None

    def __getattr__(self, name: any) -> any:
        """
        Called when the default attribute access fails with an AttributeError.

        Args:
            name: the attribute

        Returns:
            The attribute of this model.
        """
        try:
            return getattr(self.extensive_model, name)
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

    def solve(self, log_to_console: bool = False,
              start: int = 0,
              flag_rolling: bool = 0,
              **kwargs) -> tuple:
        """
        Call extensive solver to solve the discretized problem. It will first
        construct the extensive model and then call Gurobi solver to solve it.

        Args:
            log_to_console: whether to log to console
            start: starting stage index
            flag_rolling:  whether using rolling horizon

        Returns:
            a tuple of construct time, solving time and the objective value of the model
        """
        # check whether the continuity of the problem is discretized
        # check whether the number of states and samples are updated
        # discrete problem does not require being discretized
        self.MSP.check_state_and_continuous_discretized()
        self.MSP.check_markov_copy_models_update_nums()

        construction_start_time = time.time()

        self.extensive_model = gurobipy.Model()
        self.extensive_model.ModelSense = self.MSP.sense
        self.start = start

        for k, v in kwargs.items():
            setattr(self.extensive_model.Params, k, v)  # set attribute for an object
        self._construct_extensive(flag_rolling)  # this step is very import for the extensive model solving
        construction_end_time = time.time()
        self.construction_time = construction_end_time - construction_start_time
        solving_start_time = time.time()
        self.extensive_model.Params.LogToConsole = log_to_console
        self.extensive_model.optimize()
        solving_end_time = time.time()
        self.solving_time = solving_end_time - solving_start_time
        self.total_time = self.construction_time + self.solving_time
        print('*' * 30, end = '\n')
        print('the result of extensive model is %.4f' % self.extensive_model.ObjVal)
        return self.construction_time, self.solving_time, self.extensive_model.objVal

    def _get_varname(self):
        if type(self.MSP.models[self.start]) != list:
            names = [var.varname for var in self.MSP.models[self.start].getVars()]
        else:
            names = [var.varname for var in self.MSP.models[self.start][0].getVars()]
        return names

    def _get_first_stage_vars(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            vars_ = {name: self.extensive_model.getVarByName(name + '(0,)')
                     for name in names}
        else:
            vars_ = {name: self.extensive_model.getVarByName(name + '((0,),(0,))')
                     for name in names}
        return vars_

    def _get_first_stage_states(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            states = {name: self.extensive_model.getVarByName(name + '(0,)')
                      for name in names}
        else:
            states = {name: self.extensive_model.getVarByName(name + '((0,),(0,))')
                      for name in names}
        return states

    @property
    def first_stage_solution(self):
        """
        the obtained solution in the first stage

        """
        states = self._get_first_stage_states()
        return {k: v.X for k, v in states.items()}

    @property
    def first_stage_all_solution(self):
        """
        this property is actually same with fist_stage_solution

        """
        vars_ = self._get_first_stage_vars()
        return {k: v.X for k, v in vars_.items()}

    @property
    def first_stage_cost(self):
        vars_ = self._get_first_stage_vars()
        return sum(v.obj * v.X for k, v in vars_.items())

    # noinspection PyTypeChecker
    def _construct_extensive(self, flag_rolling: bool) -> None:
        """
            Construct the extensive model.

            The basic idea of this function is:
            backward from the last stage,
            enumerate all the sample paths and add all the state /local copy variables;
            for each sample path,
            update the corresponding uncertainty,
            copy the variables and constraints,
            and obj coefficients are updated by the obj in adding Variable functions.

        Args:
            flag_rolling: Whether it is rolling horizon computation

        For CTG, it is an alpha added in the constraints:
        alpha + ax + by >= c in minimization problem
        alpha + ax + by <= c in maximization problem
        """
        msp = self.MSP
        T = msp.T
        start = self.start
        n_Markov_states = msp.n_Markov_states
        n_samples = (
            [msp.models[t].n_samples for t in range(T)]
            if n_Markov_states == 1
            else [msp.models[t][0].n_samples for t in range(T)]
        )
        n_states = msp.n_states
        # check if CTG variable is added or not
        # chen: CTG may be the cutting approximation parameter
        initial_model = (
            msp.models[start] if n_Markov_states == 1 else msp.models[start][0]
        )
        flag_CTG = 1 if initial_model.alpha is not None else -1
        # |       stage 0       |        stage 1       | ... |       stage T-1      |
        # |local_copies, states_extensive | local_copies, states_extensive | ... | local_copies, states_extensive |
        # |local_copies,        | local_copies,        | ... | local_copies, states_extensive |
        # extensive formulation only includes necessary variables
        states_extensive = None
        sample_paths = None
        stage_cost = {}
        new_stage_cost = {}
        last_stage_sample_paths = []
        for t in reversed(range(start, T)):
            # enumerate sample paths only necessarily at stage T - 1
            if t == T - 1:
                _, sample_paths = msp.enumerate_sample_paths(t, start, flag_rolling)
                states_extensive = [  # states_extensive is a 2-dimensional list
                    self.extensive_model.addVars(sample_paths)  # all the state vars are added at stage T - 1
                    for _ in range(n_states[t])  # the number of states at one stage is usually 1
                ]
            # local copy states is the previous stage state variables. last_stage_sample_paths corresponds to paths
            # of the previous stage
            if t != start:
                _, last_stage_sample_paths = msp.enumerate_sample_paths(t - 1, start, flag_rolling)
                local_copy_states = [
                    self.extensive_model.addVars(last_stage_sample_paths)
                    for _ in range(n_states[t - 1])
                ]
                if flag_CTG == 1:
                    new_stage_cost = {  # new_stage_cost is a dict
                        last_stage_sample_path: 0
                        for last_stage_sample_path in last_stage_sample_paths
                    }
            else:
                local_copy_states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]
            M = [msp.models[t]] if n_Markov_states == 1 else msp.models[t]
            for j in range(n_samples[t]):
                for k, m in enumerate(M):
                    # copy information from model in scenario j and markov state k
                    # note: m is the original model, not the extensive model
                    # uncertainty realization in obj, lhs or rhs is updated by the following line of code
                    m.update_uncertainty(j)  # m is the model at each stage
                    m.update()  # do not forget to update the model,
                    # or else the following codes can't get the updated uncertainty in obj/rhs/coef
                    # compute sample paths that go through the current node
                    current_sample_paths = (
                        [item
                         for item in sample_paths
                         if item[0][t - start] == j and item[1][t - start] == k
                         ]
                        if n_Markov_states != 1  # the following is for non-Markov problem
                        else [item for item in sample_paths if item[t - start] == j]
                    )

                    controls_ = m.controls  # Trailing Underscore (var_): Used to avoid conflicts with Python keywords
                    states_ = m.states
                    local_copies_ = m.local_copies
                    controls_dict: dict[gurobipy.Var, int] = {v: i for i, v in enumerate(controls_)}
                    states_dict = {v: i for i, v in enumerate(states_)}
                    local_copies_dict = {
                        v: i for i, v in enumerate(local_copies_)
                    }

                    for current_sample_path in current_sample_paths:
                        flag_reduced_name = 0
                        if len(str(current_sample_path)) > 100:  # use when addVar name in the later codes
                            flag_reduced_name = 1  # when the sample path is too long, change the name of variables
                        if t != start:
                            # the sample path that goes through the ancestor node
                            past_sample_path = (
                                current_sample_path[:-1]
                                if n_Markov_states == 1
                                else (
                                    current_sample_path[0][:-1],
                                    current_sample_path[1][:-1],
                                )
                            )
                        else:
                            past_sample_path = current_sample_path

                        current_node_weight = 0
                        weight = 0
                        if flag_CTG == -1 or t == start:
                            weight = msp.discount ** (
                                (t - start)
                            ) * msp.compute_weight_sample_path(
                                current_sample_path, start
                            )
                        else:
                            current_node_weight = msp.compute_current_node_weight(
                                current_sample_path)

                        for i in range(n_states[t]):
                            obj = (
                                states_[i].obj * weight
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            states_extensive[i][current_sample_path].lb = states_[i].lb
                            states_extensive[i][current_sample_path].ub = states_[i].ub
                            states_extensive[i][current_sample_path].obj = obj
                            states_extensive[i][current_sample_path].vtype = states_[i].vtype
                            if flag_reduced_name == 0:  # the names of the previous stage local copies will change
                                states_extensive[i][current_sample_path].varName = states_[
                                                                                       i
                                                                                   ].varName + str(
                                    current_sample_path).replace(
                                    " ", ""
                                )
                            # cost-to-go update when ctg is true
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                        states_extensive[i][current_sample_path]
                                        * states_[i].obj
                                        * current_node_weight
                                )

                        if t == start:  # local_copy_states only have name at stage start
                            for i in range(n_states[t]):
                                local_copy_states[i][current_sample_path].lb = local_copies_[i].lb
                                local_copy_states[i][current_sample_path].ub = local_copies_[i].ub
                                local_copy_states[i][current_sample_path].obj = local_copies_[i].obj
                                local_copy_states[i][current_sample_path].vtype = local_copies_[i].vtype
                                if flag_reduced_name == 0:
                                    local_copy_states[i][current_sample_path].varName = (local_copies_[i].varname +
                                                                                         str(current_sample_path).replace(
                                                                                             " ", ""))
                        else:
                            for i in range(n_states[t]):
                                local_copy_states[i][past_sample_path].lb = local_copies_[i].lb
                                local_copy_states[i][past_sample_path].ub = local_copies_[i].ub
                                local_copy_states[i][past_sample_path].obj = local_copies_[i].obj
                                local_copy_states[i][past_sample_path].vtype = local_copies_[i].vtype
                                if flag_reduced_name == 0:
                                    local_copy_states[i][past_sample_path].varName = (local_copies_[i].varname +
                                                                                      str(past_sample_path).replace(
                                                                                          " ", ""))

                        # copy local variables
                        controls = [gurobipy.Var for _ in range(len(controls_))]
                        for i, var in enumerate(controls_):
                            obj = (
                                var.obj * weight
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            controls[i] = self.extensive_model.addVar(
                                lb=var.lb,
                                ub=var.ub,
                                obj=obj,
                                vtype=var.vtype,
                                name=(
                                    var.varname
                                    + str(current_sample_path).replace(" ", "")
                                    if flag_reduced_name == 0
                                    else ""
                                ),
                            )
                            # cost-to-go update when ctg is true and t is not the starting stage
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                        controls[i] * var.obj * current_node_weight
                                )
                        # self.extensive_model.update()
                        # add constraints
                        if t != T - 1 and flag_CTG == 1:
                            self.extensive_model.addConstr(
                                msp.sense
                                * (
                                        controls[controls_dict[m.getVarByName("alpha")]] - stage_cost[
                                    current_sample_path]
                                )
                                >= 0
                            )
                        # add constraint for each current sample path
                        for constr_ in m.getConstrs():
                            rhs_ = constr_.rhs
                            # getRow() Retrieve the list of variables that participate in a constraint,
                            # and the associated coefficients. The result is returned as a LinExpr object.
                            expr_ = m.getRow(constr_)
                            lhs = gurobipy.LinExpr()
                            for i in range(expr_.size()):
                                # 3 type of vars: state var, local copy var, control var
                                if expr_.getVar(i) in controls_dict.keys():
                                    pos = controls_dict[expr_.getVar(i)]
                                    lhs += expr_.getCoeff(i) * controls[pos]
                                # chen: without deep copy, the following line of code may be wrong
                                elif expr_.getVar(i) in states_dict.keys():
                                    pos = states_dict[expr_.getVar(i)]
                                    lhs += (
                                            expr_.getCoeff(i)
                                            * states_extensive[pos][current_sample_path]
                                    )
                                elif (
                                        expr_.getVar(i) in local_copies_dict.keys()
                                ):
                                    pos = local_copies_dict[expr_.getVar(i)]
                                    if t != start:
                                        lhs += (
                                                expr_.getCoeff(i)
                                                * local_copy_states[pos][past_sample_path]
                                        )
                                    else:
                                        lhs += (
                                                expr_.getCoeff(i)
                                                * local_copy_states[pos][current_sample_path]
                                        )
                            #! end expression loop
                            self.extensive_model.addConstr(  # model is updated in the addVar or addConstr
                                lhs=lhs, sense=constr_.sense, rhs=rhs_
                            )
                            # pass
                        #! end copying the constraints
                    #! end MC loop
                #! end scenarios loop
            #! end scenario loop
            states_extensive = local_copy_states
            if flag_CTG == 1:
                stage_cost = new_stage_cost
            sample_paths = last_stage_sample_paths

        name = 'extensive.lp'
        self.extensive_model.update()
        self.extensive_model.write(name)


class SDDP(object):
    """
    SDDP solver base class.

    Args:
        msp: A multi-stage stochastic program object
        biased_sampling: whether used biased sampling, i,e, sample probabilities are with weights

    Attributes:
        l: float between 0 and 1/array-like of floats between 0 and 1
                The weights of AVaR from stage 2 to stage T.
                If floated, the weight will be assigned to the same number.
                If array-like, must be of length T-1 (for finite horizon problem)
                or T (for infinite horizon problem).
        a: float between 0 and 1/array-like of floats between 0 and 1.
                The quantile parameters in value-at-risk from stage 2 to stage T
                If floated, those parameters will be assigned to the same number.
                If array-like, must be of length T-1 (for finite horizon problem)
                or T (for infinite horizon problem).

        obj_bound: list, objective bound found by the solver at each iteration
        policy_value: list, policy value at each iteration
    """

    def __init__(self, msp: MSLP, biased_sampling = False):
        # the following 3 lines are for regularization setting
        self.rgl_b: float = None
        self.rgl_a: float = None
        self.rgl_norm: str = None

        self.obj_bound: list = []  # objective bound found by the solver at each iteration
        self.policy_value: list = []  # policy value at each iteration
        self.msp: MSLP = msp
        self.forward_T: int = msp.T
        self.cut_T: int = msp.T - 1
        self.cut_type: list[str] = ["B"]
        self.cut_type_list: list[list[str]] = [["B"] for t in range(self.cut_T)]
        self.iteration: int = 0

        # the following 3 lines are for parallel computation setting
        self.n_processes: int = 1
        self.jobs: list = []
        self.n_steps: int = 1

        self.percentile = 95
        self.biased_sampling = biased_sampling

        self.total_time: float = None

        if self.biased_sampling:
            # l: float between 0 and 1/array-like of floats between 0 and 1
            #    The weights of AVaR from stage 2 to stage T.
            #    If floated, the weight will be assigned to the same number.
            #    If array-like, must be of length T-1 (for finite horizon problem)
            #    or T (for infinite horizon problem).
            #
            # a: float between 0 and 1/array-like of floats between 0 and 1.
            #    The quantile parameters in value-at-risk from stage 2 to stage T
            #    If floated, those parameters will be assigned to the same number.
            #    If array-like, must be of length T-1 (for finite horizon problem)
            #    or T (for infinite horizon problem).
            try:
                self.a = self.msp.a
                self.l = self.msp.l
                for t in range(self.msp.T):
                    m = self.msp.models[t]
                    n_samples = m.n_samples
                    # the following 2 attributes will be added to the model when biased sampling
                    m.counts = numpy.zeros(n_samples)
                    # sample weights will be updated by another function
                    # in the backward step if biased sampling
                    m.weights = numpy.ones(n_samples) / n_samples
            except AttributeError:
                raise Exception("Risk averse parameters unset!")

    def __repr__(self):
        return (
            "<{} solver instance, {} processes, {} steps>"
            .format(self.__class__, self.n_processes, self.n_steps)
        )

    def _select_trial_solution(self, state_solution: list[list]) -> list:
        if self:
            return state_solution[:-1]

    def _forward(
            self,
            randomState_instance: numpy.random.RandomState = None,
            sample_path_idx: list | list[list] = None,
            markovian_idx: list = None,
            markovian_samples: list = None,
            solve_true_flag: bool = False,
            query_vars: list = None,
            query_constraints: list = None,
            query_stage_cost_flag: bool = False
    ) -> dict:
        """
        Single forward step.

        Args:
            randomState_instance: a numpy RandomState instance.
            sample_path_idx: Indices of the sample path.
            markovian_idx: markovian uncertainty index
            markovian_samples: the markovian samples
            solve_true_flag: whether solving the true continuous-uncertainty problem
            query_vars: the vars that wants to check(query_vars)
            query_constraints: the constraints that wants to check
            query_stage_cost_flag: whether to query_vars values of individual stage costs

        """
        msp = self.msp
        state_solution = [[] for _ in range(self.forward_T)]
        policy_value = 0 # policy value, i.e., the final objective value of the sampled path
        query_vars = [] if query_vars is None else list(query_vars)
        query_constraints = [] if query_constraints is None else list(query_constraints)
        queryVar_solution = {item: numpy.full(self.forward_T, numpy.nan) for item in query_vars}
        query_constraint_dualValue = {item: numpy.full(self.forward_T, numpy.nan) for item in query_constraints}
        stage_cost = numpy.full(self.forward_T, numpy.nan)
        # time loop
        for t in range(self.forward_T):
            idx, tm_idx = (t, t)  # starting and end stage index in one computation at each stage
            if msp.type == "stage-wise independent":
                m = msp.models[idx]
            else:  # markovian discrete or markovian continuous
                state_index = 0
                if t == 0:
                    m = msp.models[idx][0]
                else:
                    last_state_index = state_index
                    if sample_path_idx is not None:
                        state_index = sample_path_idx[1][t]  # index 1 of sample_paths is the markov state index
                    elif markovian_idx is not None:
                        state_index = markovian_idx[t]
                    else: # random choose one, may be for markov continuous
                        state_index = randomState_instance.choice(
                            range(msp.n_Markov_states[idx]),  # transition_matrix is 3-D matrix
                            p = msp.transition_matrix[tm_idx][last_state_index]
                        )
                    m = msp.models[idx][state_index]
                    if markovian_idx is not None:
                        m.update_uncertainty_dependent(markovian_samples[t])
            if t > 0:
                m.update_link_constrs(state_solution[t - 1])
                # exhaustive evaluation when the sample paths are given
                if sample_path_idx is not None:
                    if msp.type == "stage-wise independent":
                        uncertainty_index = sample_path_idx[t]
                    else:
                        uncertainty_index = sample_path_idx[0][t]
                    m.update_uncertainty(uncertainty_index) # update the uncertainty realization in the model

                # solving the true problem does not happen in the given examples of the quick_start folder
                # true stage-wise independent randomness is infinite and solve for true
                elif m.type == 'continuous' and solve_true_flag: # only use when solving the true continuous-uncertainty problem
                    m.sample_uncertainty(randomState_instance)
                # true stage-wise independent randomness is large and solve for true
                elif m.type == 'discrete' and m.flag_discretized == 1 and solve_true_flag:
                    uncertainty_index = rand_int(
                        k = m.n_samples_discrete,
                        probability = m.probability,
                        randomState_instance = randomState_instance,
                    )
                    m.update_uncertainty(uncertainty_index)

                # other cases include
                # 1: true stage-wise independent randomness is infinite and solve
                # for approximation problem
                # 2: true stage-wise independent randomness is large and solve
                # for approximation problem
                # 3: true stage-wise independent randomness is small. In this
                # case, true problem and approximation problem are the same.

                else:
                    if self.biased_sampling:
                        sampling_probability = m.weights
                    else:
                        sampling_probability = m.probability

                    # the following codes are normally used for stage-wise independent situation
                    uncertainty_index = rand_int(
                        k = m.n_samples,
                        # probability can be none and the default is uniform probabilities
                        probability = sampling_probability,
                        randomState_instance = randomState_instance,
                    )
                    # random choosing a sample and update the uncertainty
                    m.update_uncertainty(uncertainty_index)

            if self.iteration != 0 and self.rgl_a != 0: # for regularization
                m.regularize(self.rgl_center[t], self.rgl_norm, self.rgl_a,
                             self.rgl_b, self.iteration)
            m.optimize()
            if m.status not in [2, 11]: # model is either solved to optimality or terminated by user
                m.write_infeasible_model("forward_failSolved" + str(m.modelName))
            state_solution[t] = msp.get_state_solution(m, t)
            for var in m.getVars():
                if var.varName in query_vars:
                    queryVar_solution[var.varName][t] = var.X
            for constr in m.getConstrs():
                if constr.constrName in query_constraints:
                    query_constraint_dualValue[constr.constrName][t] = constr.PI
            if query_stage_cost_flag:
                stage_cost[t] = msp.get_stage_cost(m, t) / pow(msp.discount, t)
            policy_value += msp.get_stage_cost(m, t)
            if markovian_idx is not None:
                m.update_uncertainty_dependent(msp.Markov_states[idx][markovian_idx[t]])
            if self.iteration != 0 and self.rgl_a != 0:
                m.deregularize()
        # ! time loop
        if query_vars == [] and query_constraints == [] and query_stage_cost_flag is None:
            return {
                'state_solution': state_solution,
                'policy_value': policy_value
            }
        else:
            return {
                'queryVar_solution': queryVar_solution,
                'query_constraint_dualValue': query_constraint_dualValue,
                'stage_cost': stage_cost,
                'state_solution': state_solution,
                'policy_value': policy_value
            }

    def _add_and_store_cuts(
            self, t: int,
            rhs: float | ArrayLike,
            grad: float | ArrayLike,
            cuts: ArrayLike = None,
            cut_type: str = None,
            j: int = None
    ) -> None:
        """
        Store cut information (rhs and grad) to cuts for the j th step, for cut
        type cut_type and for stage t.

        Args:
            t: stage index
            rhs: right hand side of the cut constraint
            grad: gradient of the cut constraint
            cuts: a list of dictionary stores cuts coefficients and rhs.
                  Key of the dictionary is the cut type. Value of the dictionary is
                  the cut coefficients and rhs.
            cut_type: the cut type
            j: the cut adding index, for parallel processing

        """
        msp = self.msp
        if msp.n_Markov_states == 1:
            msp.models[t - 1].add_cut(rhs, grad)
            if cuts is not None:
                cuts[t - 1][cut_type][j][:] = numpy.append(rhs, grad)
        else:
            for k in range(msp.n_Markov_states[t - 1]):
                msp.models[t - 1][k].add_cut(rhs[k], grad[k])
                if cuts is not None:
                    cuts[t - 1][cut_type][j][k][:] = numpy.append(rhs[k], grad[k])
                pass

    def _compute_cuts(self, t: int, m: StochasticModel,
                      objLP_samples: ArrayLike,
                      gradLP_samples: ArrayLike)\
                     -> tuple[float, float] |tuple[ArrayLike, ArrayLike]:
        """
            get the expected value of objectives and gradients of all the samples
        Args:
            t: stage index
            m: the gurobi model at stage t
            objLP_samples: objectives of all the samples at stage t
            gradLP_samples:  gradients of the linked constraints of all the samples at stage t

        Returns:

        """
        msp = self.msp
        if msp.n_Markov_states == 1:
            return m.average(objLP_samples[0], gradLP_samples[0]) # only 1 rows
        objLP_samples = objLP_samples.reshape( # may be not necessary
            msp.n_Markov_states[t] * msp.n_samples[t])
        gradLP_samples = gradLP_samples.reshape(
            msp.n_Markov_states[t] * msp.n_samples[t], msp.n_states[t])
        # for markov, transition probability between states in consecutive stages is considered
        probability_ind = (
            m.probability if m.probability # will update the probability if m.probability is not none
            else numpy.ones(m.n_samples) / m.n_samples
        )
        probability = numpy.einsum('ij,k->ijk', msp.transition_matrix[t],
                                   probability_ind)
        probability = probability.reshape(msp.n_Markov_states[t - 1],
                                          msp.n_Markov_states[t] * msp.n_samples[t])
        objLP = numpy.empty(msp.n_Markov_states[t - 1])
        gradLP = numpy.empty((msp.n_Markov_states[t - 1], msp.n_states[t]))
        for k in range(msp.n_Markov_states[t - 1]):
            objLP[k], gradLP[k] = m.average(objLP_samples, gradLP_samples,
                                             probability[k])
        return objLP, gradLP

    def _backward(self, state_solution: list[float],
                  j: int = None,
                  lock = None,
                  cuts = None) -> None:
        """
            Single backward step of SDDP serially or in parallel.
            Add and store cuts in this step.

        Args:
          state_solution: feasible state solutions obtained from forward step
          j: index of forward sampling in parallel processing
          cuts: a dictionary stores cuts coefficients and rhs.
                Key of the dictionary is the cut type. Value of the dictionary is
                the cut coefficients and rhs.
        """
        msp = self.msp
        for t in range(msp.T - 1, 0, -1):
            if msp.n_Markov_states == 1:
                M, n_Markov_states = [msp.models[t]], 1
            else:
                M, n_Markov_states = msp.models[t], msp.n_Markov_states[t]
            objLP_samples = numpy.empty((n_Markov_states, msp.n_samples[t]))
            gradLP_samples = numpy.empty((n_Markov_states, msp.n_samples[t],
                                      msp.n_states[t]))
            for k, m in enumerate(M):
                m: StochasticModel
                if msp.n_Markov_states != 1:
                    m.update_link_constrs(state_solution[t - 1])
                objLP_samples[k], gradLP_samples[k] = m.solveLP()

                # if self.biased_sampling:
                #     # chen: actually not used, the function is flawed
                #     self._compute_bs_frequency(objLP_samples, m, t)

            objLP, gradLP = self._compute_cuts(t, M[0], objLP_samples, gradLP_samples)
            # according to the cut constraint formula, the following is the difference
            objLP -= numpy.matmul(gradLP, state_solution[t - 1]) # matrix product

            if lock is None:
                self._add_and_store_cuts(t, rhs = objLP, grad = gradLP, cuts = cuts, cut_type = "B", j = j)
            else:
                with lock:
                    self._add_and_store_cuts(t, rhs=objLP, grad=gradLP, cuts=cuts, cut_type="B", j=j)
            # self._add_cuts_additional_procedure(t, objLP, gradLP, objLP_samples,
            #                                     gradLP_samples, state_solution[t - 1], cuts, "B", j)

    def _add_cuts_additional_procedure(*args, **kwargs):
        pass

    def _compute_bs_frequency(self, obj: list[float], m: StochasticModel, t: int):
        """
            This function is flawed and not used.

        Args:
            obj: objectives of different samples. may be wrong, not sure whether it is a list or float
            m: the StochasticModel
            t: stage index
        """
        n_samples = m.n_samples
        if self.iteration > 0:
            objSortedIndex = numpy.argsort(obj)
            tempSum = 0

            for index in objSortedIndex:
                tempSum += m.weights[index]
                if tempSum >= 1 - self.a[t]:
                    obj_kappa = index
                    break

            for k in range(n_samples):
                if obj[k] >= obj[obj_kappa]:
                    m.counts[k] += 1
                m.counts[k] *= 1 - math.pow(0.5, self.iteration)

            countSorted = numpy.sort(m.counts)
            countSortedIndex = numpy.argsort(m.counts)

            kappa = math.ceil((1 - self.a[t]) * n_samples)
            count_kappa = countSorted[kappa - 1]

            upper_orders = countSortedIndex[[i for i in range(n_samples)
                                             if i > kappa - 1]]
            lower_orders = countSortedIndex[[i for i in range(n_samples)
                                             if i < kappa - 1]]

            for k in range(n_samples):
                if m.counts[k] < count_kappa:
                    m.weights[k] = (1 - self.l[t]) / n_samples
                elif m.counts[k] == count_kappa and k in lower_orders:
                    m.weights[k] = (1 - self.l[t]) / n_samples
                elif m.counts[k] == count_kappa and k not in upper_orders:
                    m.weights[k] = ((1 - self.l[t]) / n_samples + self.l[t]
                                    - self.l[t] * (n_samples - kappa) / (self.a[t] * n_samples))
                elif m.counts[k] > count_kappa or k in upper_orders:
                    m.weights[k] = ((1 - self.l[t]) / n_samples
                                    + self.l[t] / (self.a[t] * n_samples))

    def _add_cut_from_multiprocessing_array(self, cuts):
        for t in range(self.cut_T):
            for cut_type in self.cut_type_list[t]:
                for cut in cuts[t][cut_type]:
                    if self.msp.n_Markov_states == 1:
                        self.msp.models[t]._add_cut(rhs=cut[0], gradient=cut[1:])
                    else:
                        for k in range(self.msp.n_Markov_states[t]):
                            self.msp.models[t][k]._add_cut(
                                rhs=cut[k][0], gradient=cut[k][1:])

    def _remove_redundant_cut(self, clean_stages):
        for t in clean_stages:
            M = (
                [self.msp.models[t]]
                if self.msp.n_Markov_states == 1
                else self.msp.models[t]
            )
            for m in M:
                m.update()
                constr = m.cuts
                for idx, cut in enumerate(constr):
                    if cut.sense == '>':
                        cut.sense = '<'
                    elif cut.sense == '<':
                        cut.sense = '>'
                    flag = 1
                    for k in range(m.n_samples):
                        m.update_uncertainty(k)
                        m.optimize()
                        if m.status == 4:
                            m.Params.DualReductions = 0
                            m.optimize()
                        if m.status not in [3, 11]:
                            flag = 0
                    if flag == 1:
                        m._remove_cut(idx)
                    else:
                        if cut.sense == '>':
                            cut.sense = '<'
                        elif cut.sense == '<':
                            cut.sense = '>'
                m.update()

    def _compute_cut_type(self):
        pass

    def _SDDP_single(self) -> list[float]:
        """
        A single serial SDDP step including both
        the forward pass and backward pass.

        Returns:
            Returns the policy value in list format

        """
        # randomState_instance is constructed by the iteration index
        randomState_instance = numpy.random.RandomState(self.iteration)
        temp = self._forward(randomState_instance)
        state_solution = temp['state_solution']
        # this policy_value is a float
        policy_value = temp['policy_value']
        self.rgl_center = state_solution
        pre_state_solution = self._select_trial_solution(state_solution)
        self._backward(pre_state_solution)
        return [policy_value]

    def _SDDP_single_process(self, jobs, lock, cuts):
        """
        Multiple SDDP jobs by single process. policy_value will store the policy values.
        cuts will store the cut information.
        Have not use the lock parameter so far.

        """
        # randomState_instance is constructed by the number of iteration and the index
        # of the first job that the current process does.
        # list can be the seed of the random generator
        randomState_instance = numpy.random.RandomState([self.iteration, jobs[0]])
        for j in jobs:
            temp = self._forward(randomState_instance)
            state_solution = temp['state_solution']
            # policy_value[j] = temp['policy_value']
            # regularization needs to store last state_solution
            # if j == jobs[-1] and self.rgl_a != 0:
            #     for t in range(self.forward_T):
            #         idx = t
            #         for i in range(self.msp.n_states[idx]):
            #             state_solution[t][i] = solution[t][i]
            pre_state_solution = self._select_trial_solution(state_solution)
            self._backward(pre_state_solution, j, lock, cuts)
            pdb.set_trace()

    def _SDDP_multiprocessesing(self):
        """
        Prepare a collection of multiprocessing arrays to store cuts.
        Cuts are stored in the form of:
         Independent case (index: t, cut_type, j):
            {t:{cut_type: [cut_coeffs_and_rhs]}
         Markovian case (index: t, cut_type, j, k):
            {t:{cut_type: [[cut_coeffs_and_rhs]]}
        """

        # if self.msp.n_Markov_states == 1:
        #     cuts = {
        #         t: {
        #             # RawArray: an array allocated from shared memory
        #             cut_type: [multiprocessing.RawArray("d",
        #                                                 [0] * (self.msp.n_states[t] + 1))
        #                        for _ in range(self.n_steps)]
        #             for cut_type in self.cut_type_list[t]}
        #         for t in range(self.cut_T)}
        # else:
        #     cuts = {
        #         t: {
        #             cut_type: [
        #                 [multiprocessing.RawArray("d",
        #                                           [0] * (self.msp.n_states[t] + 1))
        #                  for _ in range(self.msp.n_Markov_states[t])]
        #                 for _ in range(self.n_steps)]
        #             for cut_type in self.cut_type_list[t]}
        #         for t in range(self.cut_T)}
        cuts = None

        # 'd' means decimal
        # policy_value = multiprocessing.Array("d", [0] * self.n_steps)
        policy_value = []
        # lock = multiprocessing.Lock()
        state_solution = None
        # regularization needs to store last state_solution
        # if self.rgl_a != 0:
        #     state_solution = [multiprocessing.Array(
        #         "d", [0] * self.msp.n_states[t])
        #         for t in range(self.forward_T)
        #     ]

        procs = [multiprocessing.Process(
                target = self._SDDP_single_process,
                args = (self.jobs[p], '', cuts),
            ) for p in  range(self.n_processes)]
        for p in range(self.n_processes):
            # self._SDDP_single_process(self.jobs[p], lock, cuts)

            # procs[p] = multiprocessing.Process(
            #     target = self._SDDP_single_process,
            #     args = (self.jobs[p], '', cuts),
            # )
            # pdb.set_trace()
            procs[p].start() # somthing wrong here
        for proc in procs:
            proc.join()

        self._add_cut_from_multiprocessing_array(cuts)
        # regularization needs to store last state_solution
        # if self.rgl_a != 0:
        #     self.rgl_center = [list(item) for item in state_solution]

        return [item for item in policy_value]

    def solve(
            self,
            n_processes: int = 1,
            n_steps: int = 1,
            max_iterations: int = 10000,
            max_stable_iterations: int = 10000,
            max_time: float = 1000000.0,
            tol: float = 0.001,
            freq_evaluations: int = None,
            percentile: int = 95,
            tol_diff: float = float("-inf"),
            freq_comparisons: int = None,
            n_simulations: int = 3000,
            evaluation_true_flag: bool = False,
            query_vars: list = None,
            query_T: int = None,
            query_constraints: list = None,
            query_stage_cost_flag: bool = False,
            query_policy_value_flag: bool = False,
            freq_clean: int | list = None,
            logToFile_flag: bool = True,
            logToConsole_flag: bool = True,
            directory: str = '',
            rgl_norm: str = 'L2',
            rgl_a: float = 0,
            rgl_b: float = 0.95,
    ):
        """
        Solve the discretized problem.

        Args:
          evaluation_true_flag: whether evaluating the true problem
          rgl_b: regulization coefficient b
          rgl_a: regulization coefficient a
          rgl_norm: regularized norm, 'L1' or 'L2'
          directory: the output directory of the logger and csv files

          query_vars: the vars that wants to check(query_vars)
          query_policy_value_flag: whether to query_vars the policy value
          query_stage_cost_flag: whether to query_vars the individual stage costs
          query_constraints: the constraints that wants to check
          query_T: the last stage in querying

          n_processes: int, optional (default=1)
              The number of processes to run in parallel. Run serial SDDP if 1.
              If n_steps is 1, n_processes is coerced to be 1.
          n_steps: int, optional (default=1)
              The number of forward/backward steps to run in each cut iteration.
              It is coerced to be 1 if n_processes is 1.
          max_iterations: int, optional (default=10000)
              The maximum number of iterations to run SDDP.
          max_stable_iterations: int, optional (default=10000)
              The maximum number of iterations to have same deterministic bound
          max_time: the maximum running time
          tol: float, optional (default=1e-3)
              tolerance for convergence of bounds
          freq_evaluations: int, optional (default=None)
              The frequency of evaluating gap on the discretized problem. It will
              be ignored if risk-averse
          percentile: float, optional (default=95)
              The percentile used to compute confidence interval
          tol_diff: float, optional (default=-inf)
              The stabilization threshold
          freq_comparisons: int, optional (default=None)
              The frequency of comparisons of policies
          n_simulations: int, optional (default=10000)
              The number of simluations to run when evaluating a policy
              on the discretized problem
          freq_clean: int/list, optional (default=None)
              The frequency of removing redundant cuts.
              If int, perform cleaning at the same frequency for all stages.
              If listed, perform cleaning at different frequency for each stage;
              must be of length T-1 (the last stage does not have any cuts).
          logToFile_flag: binary, optional (default=1)
              Switch of logging to log file
          logToConsole_flag: binary, optional (default=1)
              Switch of logging to console

        Examples:
        --------
        >>> SDDP().solve(max_iterations = 10, max_time = 10,
            max_stable_iterations = 10)

        Optimality gap based stopping criteria: evaluate the obtained policy
        every freq_evaluations iterations by running n_simulations Monte Carlo
        simulations. If the gap becomes not larger than tol, the algorithm
        will be stopped.
        >>> SDDP().solve(freq_evaluations = 10, n_simulations = 1000, tol = 1e-2)
        Simulation can be turned off; the solver will evaluate the exact expected
        policy value.
        >>> SDDP().solve(freq_evaluation = 10, n_simulations = -1, tol = 1e-2)

        Stabilization based stopping criteria: compare the policy every
        freq_comparisons iterations by computing the CI of difference of the
        expected policy values. If the upper end of CI becomes not larger
        than tol diff, the algorithm will be stopped.
        >>> SDDP().solve(freq_comparisons = 10, n_simulations = 1000, tol = 1e-2)

        """
        msp = self.msp
        if freq_clean is not None:
            if isinstance(freq_clean, (numbers.Integral, numpy.integer)):
                freq_clean = [freq_clean] * (msp.T - 1)
            if isinstance(freq_clean, (abc.Sequence, numpy.ndarray)):
                if len(freq_clean) != msp.T - 1:
                    raise ValueError("freq_clean list must be of length T - 1!")
            else:
                raise TypeError("freq_clean must be int/list instead of {}!"
                                .format(type(freq_clean)))
        if not msp.flag_updated:  # if the model having not been updated
            msp.update()
        stable_iterations = 0
        total_time = 0
        gap = 1.0
        right_end_of_CI = float("inf")
        obj_bound_past = msp.bound
        self.percentile = percentile
        self.rgl_norm = rgl_norm
        self.rgl_a = rgl_a
        self.rgl_b = rgl_b

        # distinguish policy_value_sim from policy_value
        policy_value_sim_past = None

        if n_processes != 1:
            self.n_steps = n_steps
            self.n_processes = min(n_steps, n_processes)
            self.jobs = allocate_jobs(self.n_steps, self.n_processes)

        logger_sddp = LoggerSDDP(
            logToFile_flag = logToFile_flag,
            logToConsole_flag = logToConsole_flag,
            n_processes = self.n_processes,
            percentile = self.percentile,
            directory = directory,
        )
        logger_sddp.header()
        if freq_evaluations is not None:
            logger_evaluation = LoggerEvaluation(
                n_simulations = n_simulations,
                percentile = percentile,
                logToFile_flag = logToFile_flag,
                logToConsole_flag = logToConsole_flag,
                directory = directory,
            )
            logger_evaluation.header()
        if freq_comparisons is not None:
            logger_comparison = LoggerComparison(
                n_simulations = n_simulations,
                percentile = percentile,
                logToFile_flag = logToFile_flag,
                logToConsole_flag = logToConsole_flag,
                directory = directory,
            )
            logger_comparison.header()
        stop_reason = ''
        try:
            while (
                    self.iteration < max_iterations
                    and total_time < max_time
                    and stable_iterations < max_stable_iterations
                    and tol < gap
                    and (tol_diff < right_end_of_CI or right_end_of_CI < 0)
            ):
                start = time.time()

                # self._compute_cut_type() # this line is not used
                if self.n_processes == 1:
                    policy_value = self._SDDP_single()
                else:
                    policy_value = self._SDDP_multiprocessesing()

                m = (
                    msp.models[0]
                    if msp.n_Markov_states == 1
                    else msp.models[0][0]
                )
                m.optimize()
                if m.status not in [2, 11]: # not solved successfully
                    m.write_infeasible_model(
                        "backward_" + str(m.model.modelName) + ".lp"
                    )
                obj_bound = m.objBound
                self.obj_bound.append(obj_bound)
                msp.obj_bound = obj_bound
                CI = ()
                if self.n_processes != 1:
                    CI = compute_CI(policy_value, percentile)
                self.policy_value.append(policy_value)

                if self.iteration >= 1:
                    if obj_bound_past == obj_bound:
                        stable_iterations += 1
                    else:
                        stable_iterations = 0
                self.iteration += 1
                obj_bound_past = obj_bound

                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time

                if self.n_processes == 1:
                    logger_sddp.text(
                        iteration = self.iteration,
                        obj_bound = obj_bound,
                        policy_value = policy_value[0],
                        time = elapsed_time,
                    )
                else:
                    logger_sddp.text(
                        iteration = self.iteration,
                        obj_bound = obj_bound,
                        CI = CI,
                        time = elapsed_time,
                    )
                if (
                        freq_evaluations is not None
                        and self.iteration % freq_evaluations == 0
                        or freq_comparisons is not None
                        and self.iteration % freq_comparisons == 0
                ):
                    directory = '' if directory is None else directory
                    start = time.time()
                    evaluation = Evaluation(msp)
                    evaluation.run(
                        n_simulations = n_simulations,
                        query_vars = query_vars,
                        query_T = query_T,
                        query_constraints = query_constraints,
                        query_stage_cost_flag = query_stage_cost_flag,
                        percentile = percentile,
                        n_processes = n_processes,
                    )
                    if query_policy_value_flag:
                        pandas.DataFrame(evaluation.policy_values).to_csv(directory +
                                                               "iter_{}_policy_value.csv".format(self.iteration))
                    if query_vars is not None:
                        for item in query_vars:
                            evaluation.solution[item].to_csv(directory +
                                                             "iter_{}_{}.csv".format(self.iteration, item))
                    if query_constraints is not None:
                        for item in query_constraints:
                            evaluation.solution_dual[item].to_csv(directory +
                                                                  "iter_{}_{}.csv".format(self.iteration, item))
                    if query_stage_cost_flag:
                        evaluation.stage_cost.to_csv(directory +
                                                     "iter_{}_stage_cost.csv".format(self.iteration))
                    if evaluation_true_flag:
                        evaluationTrue = EvaluationTrue(msp)
                        evaluationTrue.run(
                            n_simulations = n_simulations,
                            query_vars = query_vars,
                            query_T = query_T,
                            query_constraints = query_constraints,
                            query_stage_cost_flag = query_stage_cost_flag,
                            percentile = percentile,
                            n_processes = n_processes,
                        )
                        if query_policy_value_flag:
                            pandas.DataFrame(evaluationTrue.policy_value).to_csv(directory +
                                                                       "iter_{}_policy_value_true.csv".format(self.iteration))
                        if query_vars is not None:
                            for item in query_vars:
                                evaluationTrue.solution[item].to_csv(directory +
                                                                     "iter_{}_{}_true.csv".format(self.iteration, item))
                        if query_constraints is not None:
                            for item in query_constraints:
                                evaluationTrue.solution_dual[item].to_csv(directory +
                                                                          "iter_{}_{}_true.csv".format(self.iteration,
                                                                                                       item))
                        if query_stage_cost_flag:
                            evaluationTrue.stage_cost.to_csv(directory +
                                                             "iter_{}_stage_cost_true.csv".format(self.iteration))
                    elapsed_time = time.time() - start
                    gap = evaluation.gap
                    if n_simulations == -1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            obj_bound=obj_bound,
                            policy_value=evaluation.epv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    elif n_simulations == 1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            obj_bound=obj_bound,
                            policy_value=evaluation.policy_value,
                            gap=gap,
                            time=elapsed_time,
                        )
                    else:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            obj_bound=obj_bound,
                            CI=evaluation.CI,
                            gap=gap,
                            time=elapsed_time,
                        )
                if (
                        freq_comparisons is not None
                        and self.iteration % freq_comparisons == 0
                ):
                    start = time.time()
                    policy_value_sim = evaluation.policy_value
                    if self.iteration / freq_comparisons >= 2:
                        diff = msp.sense * (numpy.array(policy_value_sim_past) - numpy.array(policy_value_sim))
                        if n_simulations == -1:
                            diff_mean = numpy.mean(diff)
                            right_end_of_CI = diff_mean
                        else:
                            diff_CI = compute_CI(diff, self.percentile)
                            right_end_of_CI = diff_CI[1]
                        elapsed_time = time.time() - start
                        if n_simulations == -1:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration - freq_comparisons,
                                diff=diff_mean,
                                time=elapsed_time,
                            )
                        else:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration - freq_comparisons,
                                diff_CI=diff_CI,
                                time=elapsed_time,
                            )
                    policy_value_sim_past = policy_value_sim
                if freq_clean is not None:
                    clean_stages = [
                        t
                        for t in range(1, msp.T - 1)
                        if self.iteration % freq_clean[t] == 0
                    ]
                    if len(clean_stages) != 0:
                        self._remove_redundant_cut(clean_stages)
                # self._clean()
        except KeyboardInterrupt:
            stop_reason = "interruption by the user"
        # SDDP iteration stops
        msp.obj_bound = self.obj_bound[-1]
        if self.iteration >= max_iterations:
            stop_reason = "iteration:{} has reached".format(max_iterations)
        if total_time >= max_time:
            stop_reason = "time:{} has reached".format(max_time)
        if stable_iterations >= max_stable_iterations:
            stop_reason = "stable iteration:{} has reached".format(max_stable_iterations)
        if gap <= tol:
            stop_reason = "convergence tolerance:{} has reached".format(tol)
        if right_end_of_CI <= tol_diff:
            stop_reason = "stabilization threshold:{} has reached".format(tol_diff)

        logger_sddp.footer(reason = stop_reason)
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation.footer()
        if freq_comparisons is not None:
            logger_comparison.footer()
        self.total_time = total_time
        print('*'*50, end = '\n')
        print('the final result of SDDP is %.2f' % msp.obj_bound)

    @property
    def first_stage_solution(self):
        """the obtained solution in the first stage"""
        return (
            {var.varName: var.X for var in self.msp.models[0].getVars()}
            if self.msp.n_Markov_states == 1
            else {var.varName: var.X for var in self.msp.models[0][0].getVars()}
        )

    def plot_bounds(self, start=0, window=1, smooth=0, ax=None):
        """
        plot the evolution of bounds

        Parameters
        ----------
        ax: Matplotlib AxesSubplot instance, optional
            The specified subplot is used to plot; otherwise a new figure is created.

        window: int, optional (default=1)
            The length of the moving windows to aggregate the policy values. If
            length is bigger than 1, approximate confidence interval of the
            policy values and statistical bounds will be plotted.

        smooth: bool, optional (default=0)
            If 1, fit a smooth line to the policy values to better visualize
            the trend of statistical values/bounds.

        start: int, optional (default=0)
            The start iteration to plot the bounds. Set start to other values
            can zoom in the evolution of bounds in most recent iterations.

        Returns
        -------
        matplotlib.pyplot.figure instance
        """
        from msppy.utils.plot import plot_bounds
        return plot_bounds(self.obj_bound, self.policy_value, self.msp.sense, self.percentile,
                           start=start, window=window, smooth=smooth, ax=ax)

    @property
    def bounds(self):
        """dataframe of the obtained bound"""
        df = pandas.DataFrame.from_records(self.policy_value)
        df['obj_bound'] = self.obj_bound
        return df

class SDDiP(SDDP):
    __doc__ = SDDP.__doc__ # parent docstring is not inherited automatically

    def solve(
            self,
            cuts,
            pattern=None,
            relax_stage=None,
            level_step_size=0.2929,
            level_max_stable_iterations=1000,
            level_max_iterations=1000,
            level_max_time=1000,
            level_mip_gap=1e-4,
            level_tol=1e-3,
            *args,
            **kwargs):
        """Call SDDiP solver to solve the discretized problem.

        Parameters
        ----------
        cuts: list
            Entries of the list could be 'B','SB','LG'

        pattern: dict, optional (default=None)
            The pattern of adding cuts can be cyclical or barrier-in.
            See the example below.

        relax_stage: int, optional (default=None)
            All stage models after relax_stage (exclusive) will be relaxed.

        level_step_size: float, optional (default=0.2929)
            Step size for level method.

        level_max_stable_iterations: int, optional (default=1000)
            The maximum number of iterations to have the same deterministic g_*
            for the level method.

        level_mip_gap: float, optional (default=1e-4)
            The MIPGap used when solving the inner problem for the level method.

        level_max_iterations: int, optional (default=1000)
            The maximum number of iterations to run for the level method.

        level_max_time: int, optional (default=1000)
            The maximum number of time to run for the level method.

        level_tol: float, optional (default=1e-3)
            Tolerance for convergence of bounds for the level method.

        Examples
        --------
        >>> SDDiP().solve(max_iterations=10, cut=['SB'])

        The following cyclical add difference cuts. Specifically, for every six
        iterations add Benders' cuts for the first four,
        strengthened Benders' cuts for the fifth,
        and Lagrangian cuts for the last.

        >>> SDDiP().solve(max_iterations=10, cut=['B','SB','LG'],
        ...     pattern={"cycle": (4, 1, 1)})

        The following add difference cuts from certain iterations. Specifically,
        add Benders' cuts from the beginning,
        Strengthened Benders' cuts from the fourth iteration,
        and Lagragian cuts from the fifth iteration.

        >>> SDDiP().solve(max_iterations=10, cut=['B','SB','LG'],
        ...     pattern={'in': (0, 4, 5)})
        """
        if pattern != None:
            if not all(
                len(item) == len(cuts)
                for item in pattern.values()
            ):
                raise Exception("pattern is not compatible with cuts!")
        self.relax_stage = relax_stage if relax_stage != None else self.msp.T - 1
        self.cut_type = cuts
        self.cut_pattern = pattern
        self.level_step_size = level_step_size
        self.level_max_stable_iterations = level_max_stable_iterations
        self.level_max_iterations = level_max_iterations
        self.level_max_time = level_max_time
        self.level_mip_gap = level_mip_gap
        self.level_tol = level_tol
        super().solve(*args, **kwargs)

    def _backward(self, forward_solution, j=None, lock=None, cuts=None):
        MSP = self.msp
        for t in range(MSP.T-1, 0, -1):
            if MSP.n_Markov_states == 1:
                M, n_Markov_states = [MSP.models[t]], 1
            else:
                M, n_Markov_states = MSP.models[t], MSP.n_Markov_states[t]
            objLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            gradLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t],
                MSP.n_states[t]))
            objSBScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            objLGScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            gradLGScen = numpy.empty((n_Markov_states, MSP.n_samples[t],
                MSP.n_states[t]))
            for k, model in enumerate(M):
                if MSP.n_Markov_states != 1:
                    model._update_link_constrs(forward_solution[t-1])
                model.update()
                m = model.relax() if model.isMIP else model
                objLPScen[k], gradLPScen[k] = m.solveLP()
                # SB and LG share the same model
                if (
                    "SB" in self.cut_type_list[t-1]
                    or "LG" in self.cut_type_list[t-1]
                ):
                    m = model.copy()
                    m._delete_link_constrs()
                if "SB" in self.cut_type_list[t-1]:
                    objSBScen[k] = m._solveSB(gradLPScen[k])
                if "LG" in self.cut_type_list[t-1]:
                    objVal_primal = model._solvePrimal()
                    flag_bin = (
                        True if hasattr(self, "n_binaries")
                        else False
                    )
                    objLGScen[k], gradLGScen[k] = m._solveLG(
                        gradLPScen=gradLPScen[k],
                        given_bound=MSP.bound,
                        objVal_primal=objVal_primal,
                        flag_tight = flag_bin,
                        forward_solution=forward_solution[t-1],
                        step_size=self.level_step_size,
                        max_stable_iterations=self.level_max_stable_iterations,
                        max_iterations=self.level_max_iterations,
                        max_time=self.level_max_time,
                        MIPGap=self.level_mip_gap,
                        tol=self.level_tol,
                    )
            #! Markov states iteration ends
            if "B" in self.cut_type_list[t-1]:
                objLP, gradLP = self._compute_cuts(t, m, objLPScen, gradLPScen)
                objLP -= numpy.matmul(gradLP, forward_solution[t-1])
                self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
                self._add_cuts_additional_procedure(t, objLP, gradLP, objLPScen,
                    gradLPScen, forward_solution[t-1], cuts, "B", j)
            if "SB" in self.cut_type_list[t-1]:
                objSB, gradLP = self._compute_cuts(t, m, objSBScen, gradLPScen)
                self._add_and_store_cuts(t, objSB, gradLP, cuts, "SB", j)
                self._add_cuts_additional_procedure(t, objSB, gradLP, objSBScen,
                    gradLPScen, forward_solution[t-1], cuts, "SB", j)
            if "LG" in self.cut_type_list[t-1]:
                objLG, gradLG = self._compute_cuts(t, m, objLGScen, gradLGScen)
                self._add_and_store_cuts(t, objLG, gradLG, cuts, "LG", j)
                self._add_cuts_additional_procedure(t, objLG, gradLG, objLGScen,
                    gradLGScen, forward_solution[t-1], cuts, "LG", j)
        #! Time iteration ends

    def _compute_cut_type_by_iteration(self):
        if self.cut_pattern == None:
            return list(self.cut_type)
        else:
            if "cycle" in self.cut_pattern.keys():
                cycle = self.cut_pattern["cycle"]
                ## decide pos belongs to which interval ##
                interval = numpy.cumsum(cycle) - 1
                pos = self.iteration % sum(cycle)
                for i in range(len(interval)):
                    if pos <= interval[i]:
                        return [self.cut_type[i]]
            if "in" in self.cut_pattern.keys():
                barrier_in = self.cut_pattern["in"]
                cut = []
                for i in range(len(barrier_in)):
                    if self.iteration >= barrier_in[i]:
                        cut.append(self.cut_type[i])
                if "B" in cut and "SB" in cut:
                    cut.remove("B")
                return cut

    def _compute_cut_type_by_stage(self, t, cut_type):
        if t > self.relax_stage or self.MSP.isMIP[t] != 1:
            cut_type = ["B"]
        return cut_type

    def _compute_cut_type(self):
        cut_type_list = [None] * self.cut_T
        cut_type_by_iteration = self._compute_cut_type_by_iteration()
        for t in range(1, self.cut_T+1):
            cut_type_list[t-1] = self._compute_cut_type_by_stage(
                t, cut_type_by_iteration)
        self.cut_type_list = cut_type_list