"""
Created on 2025/1/10, 21:48

@author: Zhen Chen.

@Python version: 3.10

@disp:  Different classes of stochastic programming solvers.

"""
from msm import MSP
from util.statistics import rand_int, allocate_jobs, compute_CI
from util.logger import LoggerSDDP, LoggerEvaluation, LoggerComparison
from evaluation import Evaluation, EvaluationTrue
from collections import abc
import time
import gurobipy
import numpy
import math
import multiprocessing
import numbers
import pandas


class Extensive:
    """
    Extensive solver class.

    Can solve:

    1. small-scale stage-wise independent finite discrete risk neutral problem;

    2. small-scale Markov chain risk neutral problem.

    Parameters:
    ----------
    msp: A multi-stage stochastic program object.

    Attributes
    ----------
    extensive_model:
        The constructed extensive model

    solving_time:
        The time cost in solving extensive model

    construction_time:
        The time cost in constructing extensive model
    """

    def __init__(self, msp: MSP) -> None:
        self.start = 0 # starting stage
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
            setattr(self.extensive_model.Params, k, v) # set attribute for an object
        self._construct_extensive(flag_rolling) # this step is very import for the extensive model solving
        construction_end_time = time.time()
        self.construction_time = construction_end_time - construction_start_time
        solving_start_time = time.time()
        self.extensive_model.Params.LogToConsole = log_to_console
        self.extensive_model.optimize()
        solving_end_time = time.time()
        self.solving_time = solving_end_time - solving_start_time
        self.total_time = self.construction_time + self.solving_time
        print('*'*30)
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
            vars_ = {name:self.extensive_model.getVarByName(name + '(0,)')
                for name in names}
        else:
            vars_ = {name:self.extensive_model.getVarByName(name + '((0,),(0,))')
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
        return {k:v.X for k,v in vars_.items()}

    @property
    def first_stage_cost(self):
        vars_ = self._get_first_stage_vars()
        return sum(v.obj*v.X for k,v in vars_.items())

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
                states_extensive = [ # states_extensive is a 2-dimensional list
                    self.extensive_model.addVars(sample_paths) # all the state vars are added at stage T - 1
                    for _ in range(n_states[t]) # the number of states at one stage is usually 1
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
                    new_stage_cost = { # new_stage_cost is a dict
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
                    m.update_uncertainty(j) # m is the model at each stage
                    m.update() # do not forget to update the model,
                               # or else the following codes can't get the updated uncertainty in obj/rhs/coef
                    # compute sample paths that go through the current node
                    current_sample_paths = (
                        [   item
                            for item in sample_paths
                            if item[0][t - start] == j and item[1][t - start] == k
                        ]
                        if n_Markov_states != 1 # the following is for non-Markov problem
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
                        if len(str(current_sample_path)) > 100: # use when addVar name in the later codes
                            flag_reduced_name = 1 # when the sample path is too long, change the name of variables
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
                            if flag_reduced_name == 0: # the names of the previous stage local copies will change
                                states_extensive[i][current_sample_path].varName = states_[
                                    i
                                ].varName + str(current_sample_path).replace(
                                    " ", ""
                                )
                            # cost-to-go update when ctg is true
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    states_extensive[i][current_sample_path]
                                    * states_[i].obj
                                    * current_node_weight
                                )

                        if t == start: # local_copy_states only have name at stage start
                            for i in range(n_states[t]):
                                local_copy_states[i][current_sample_path].lb = local_copies_[i].lb
                                local_copy_states[i][current_sample_path].ub = local_copies_[i].ub
                                local_copy_states[i][current_sample_path].obj = local_copies_[i].obj
                                local_copy_states[i][current_sample_path].vtype = local_copies_[i].vtype
                                if flag_reduced_name == 0:
                                    local_copy_states[i][current_sample_path].varName = (local_copies_[i].varname +
                                                                                         str(current_sample_path).replace(" ", ""))
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
                                lb = var.lb,
                                ub = var.ub,
                                obj = obj,
                                vtype = var.vtype,
                                name = (
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
                                    controls[controls_dict[m.getVarByName("alpha")]]- stage_cost[current_sample_path]
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
                            self.extensive_model.addConstr( # model is updated in the addVar or addConstr
                                lhs = lhs, sense = constr_.sense, rhs = rhs_
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
    """

    def __init__(self, msp: MSP, biased_sampling = False):
        self.db = [] # best objective bound
        self.pv = [] # policy value
        self.msp = msp
        self.forward_T = msp.T
        self.cut_T = msp.T - 1
        self.cut_type = ["B"]
        self.cut_type_list = [["B"] for t in range(self.cut_T)]
        self.iteration = 0
        self.n_processes = 1
        self.n_steps = 1
        self.percentile = 95
        self.biased_sampling = biased_sampling

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

    def _select_trial_solution(self, forward_solution: list[list]) -> list:
        if self:
            return forward_solution[:-1]

    def _forward(
            self,
            random_state: numpy.random.RandomState = None,
            sample_path_idx: list | list[list] = None,
            markovian_idx: list = None,
            markovian_samples: list = None,
            solve_true: bool = False,
            query: list = None,
            query_dual: list = None,
            query_stage_cost: bool = False
            ) -> dict:
        """
        Single forward step.

        Args:
            random_state: a numpy RandomState instance.
            sample_path_idx: Indices of the sample path.
            markovian_idx: markovian uncertainty index
            markovian_samples: the markovian samples
            solve_true: whether solving the true continuous-uncertainty problem
            query: the vars that wants to check(query)
            query_dual: the constraints that wants to check
            query_stage_cost: whether to query values of individual stage costs.

        """
        msp = self.msp
        forward_solution = [None for _ in range(self.forward_T)]
        pv = 0
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        solution = {item: numpy.full(self.forward_T, numpy.nan) for item in query}
        solution_dual = {item: numpy.full(self.forward_T, numpy.nan) for item in query_dual}
        stage_cost = numpy.full(self.forward_T, numpy.nan)
        # time loop
        for t in range(self.forward_T):
            idx, tm_idx = (t, t)
            if msp.type == "stage-wise independent":
                m = msp.models[idx]
            else: # markovian discrete or markovian continuous
                state_index = 0
                if t == 0:
                    m = msp.models[idx][0]
                else:
                    last_state_index = state_index
                    if sample_path_idx is not None:
                        state_index = sample_path_idx[1][t] # index 1 of sample_paths is the markov state index
                    elif markovian_idx is not None:
                        state_index = markovian_idx[t]
                    else:
                        state_index = random_state.choice(
                            range(msp.n_Markov_states[idx]), # transition_matrix is 3-D matrix
                            p = msp.transition_matrix[tm_idx][last_state_index]
                        )
                    m = msp.models[idx][state_index]
                    if markovian_idx is not None:
                        m.update_uncertainty_dependent(markovian_samples[t])
            if t > 0:
                m._update_link_constrs(forward_solution[t - 1])
                # exhaustive evaluation when the sample paths are given
                if sample_path_idx is not None:
                    if msp.type == "stage-wise independent":
                        scen = sample_path_idx[t]
                    else:
                        scen = sample_path_idx[0][t]
                    m._update_uncertainty(scen)
                # true stagewise independent randomness is infinite and solve
                # for true
                elif m.type == 'continuous' and solve_true:
                    m._sample_uncertainty(random_state)
                # true stagewise independent randomness is large and solve
                # for true
                elif m.type == 'discrete' and m._flag_discrete == 1 and solve_true:
                    scen = rand_int(
                        k=m.n_samples_discrete,
                        probability=m.probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
                # other cases include
                # 1: true stagewise independent randomness is infinite and solve
                # for approximation problem
                # 2: true stagewise independent randomness is large and solve
                # for approximation problem
                # 3: true stagewise independent randomness is small. In this
                # case, true problem and approximation problem are the same.
                else:
                    if self.biased_sampling:
                        sampling_probability = m.weights
                    else:
                        sampling_probability = m.probability

                    scen = rand_int(
                        k=m.n_samples,
                        probability=sampling_probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
            if self.iteration != 0 and self.rgl_a != 0:
                m.regularize(self.rgl_center[t], self.rgl_norm, self.rgl_a,
                             self.rgl_b, self.iteration)
            m.optimize()
            if m.status not in [2, 11]:
                m.write_infeasible_model("forward_" + str(m.modelName))
            forward_solution[t] = msp._get_forward_solution(m, t)
            for var in m.getVars():
                if var.varName in query:
                    solution[var.varName][t] = var.X
            for constr in m.getConstrs():
                if constr.constrName in query_dual:
                    solution_dual[constr.constrName][t] = constr.PI
            if query_stage_cost:
                stage_cost[t] = msp._get_stage_cost(m, t) / pow(msp.discount, t)
            pv += msp._get_stage_cost(m, t)
            if markovian_idx is not None:
                m._update_uncertainty_dependent(msp.Markov_states[idx][markovian_idx[t]])
            if self.iteration != 0 and self.rgl_a != 0:
                m._deregularize()
        # ! time loop
        if query == [] and query_dual == [] and query_stage_cost is None:
            return {
                'forward_solution': forward_solution,
                'pv': pv
            }
        else:
            return {
                'solution': solution,
                'soultion_dual': solution_dual,
                'stage_cost': stage_cost,
                'forward_solution': forward_solution,
                'pv': pv
            }

    def _add_and_store_cuts(
            self, t, rhs, grad, cuts=None, cut_type=None, j=None
    ):
        """Store cut information (rhs and grad) to cuts for the j th step, for cut
        type cut_type and for stage t."""
        msp = self.msp
        if msp.n_Markov_states == 1:
            msp.models[t - 1]._add_cut(rhs, grad)
            if cuts is not None:
                cuts[t - 1][cut_type][j][:] = numpy.append(rhs, grad)
        else:
            for k in range(msp.n_Markov_states[t - 1]):
                msp.models[t - 1][k]._add_cut(rhs[k], grad[k])
                if cuts is not None:
                    cuts[t - 1][cut_type][j][k][:] = numpy.append(rhs[k], grad[k])

    def _compute_cuts(self, t, m, objLPScen, gradLPScen):
        msp = self.msp
        if msp.n_Markov_states == 1:
            return m._average(objLPScen[0], gradLPScen[0])
        objLPScen = objLPScen.reshape(
            msp.n_Markov_states[t] * msp.n_samples[t])
        gradLPScen = gradLPScen.reshape(
            msp.n_Markov_states[t] * msp.n_samples[t], msp.n_states[t])
        probability_ind = (
            m.probability if m.probability
            else numpy.ones(m.n_samples) / m.n_samples
        )
        probability = numpy.einsum('ij,k->ijk', msp.transition_matrix[t],
                                   probability_ind)
        probability = probability.reshape(msp.n_Markov_states[t - 1],
                                          msp.n_Markov_states[t] * msp.n_samples[t])
        objLP = numpy.empty(msp.n_Markov_states[t - 1])
        gradLP = numpy.empty((msp.n_Markov_states[t - 1], msp.n_states[t]))
        for k in range(msp.n_Markov_states[t - 1]):
            objLP[k], gradLP[k] = m._average(objLPScen, gradLPScen,
                                             probability[k])
        return objLP, gradLP

    def _backward(self, forward_solution, j=None, lock=None, cuts=None):
        """Single backward step of SDDP serially or in parallel.

        Parameters
        ----------
        forward_solution:
            feasible solutions obtained from forward step

        j: int
            index of forward sampling

        lock: multiprocessing.Lock

        cuts: dict
            A dictionary stores cuts coefficients and rhs.
            Key of the dictionary is the cut type. Value of the dictionary is
            the cut coefficients and rhs.
        """
        msp = self.msp
        for t in range(msp.T - 1, 0, -1):
            if msp.n_Markov_states == 1:
                M, n_Markov_states = [msp.models[t]], 1
            else:
                M, n_Markov_states = msp.models[t], msp.n_Markov_states[t]
            objLPScen = numpy.empty((n_Markov_states, msp.n_samples[t]))
            gradLPScen = numpy.empty((n_Markov_states, msp.n_samples[t],
                                      msp.n_states[t]))
            for k, m in enumerate(M):
                if msp.n_Markov_states != 1:
                    m._update_link_constrs(forward_solution[t - 1])
                objLPScen[k], gradLPScen[k] = m._solveLP()

                if self.biased_sampling:
                    self._compute_bs_frequency(objLPScen[k], m, t)

            objLP, gradLP = self._compute_cuts(t, m, objLPScen, gradLPScen)
            objLP -= numpy.matmul(gradLP, forward_solution[t - 1])
            self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
            self._add_cuts_additional_procedure(t, objLP, gradLP, objLPScen,
                                                gradLPScen, forward_solution[t - 1], cuts, "B", j)

    def _add_cuts_additional_procedure(*args, **kwargs):
        pass

    def _compute_bs_frequency(self, obj, m, t):

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

    def _SDDP_single(self):
        """A single serial SDDP step. Returns the policy value."""
        # random_state is constructed by number of iteration.
        random_state = numpy.random.RandomState(self.iteration)
        temp = self._forward(random_state)
        solution = temp['forward_solution']
        pv = temp['pv']
        self.rgl_center = solution
        solution = self._select_trial_solution(solution)
        self._backward(solution)
        return [pv]

    def _SDDP_single_process(self, pv, jobs, lock, cuts, forward_solution=None):
        """Multiple SDDP jobs by single process. pv will store the policy values.
        cuts will store the cut information. Have not use the lock parameter so
        far."""
        # random_state is constructed by the number of iteration and the index
        # of the first job that the current process does
        random_state = numpy.random.RandomState([self.iteration, jobs[0]])
        for j in jobs:
            temp = self._forward(random_state)
            solution = temp['forward_solution']
            pv[j] = temp['pv']
            # regularization needs to store last forward_solution
            if j == jobs[-1] and self.rgl_a != 0:
                for t in range(self.forward_T):
                    idx = t
                    for i in range(self.msp.n_states[idx]):
                        forward_solution[t][i] = solution[t][i]
            solution = self._select_trial_solution(solution)
            self._backward(solution, j, lock, cuts)

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
                        m._update_uncertainty(k)
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

    def _SDDP_multiprocessesing(self):
        """Prepare a collection of multiprocessing arrays to store cuts.
        Cuts are stored in the form of:
         Independent case (index: t, cut_type, j):
            {t:{cut_type: [cut_coeffs_and_rhs]}
         Markovian case (index: t, cut_type, j, k):
            {t:{cut_type: [[cut_coeffs_and_rhs]]}
        """
        procs = [None] * self.n_processes
        if self.msp.n_Markov_states == 1:
            cuts = {
                t: {
                    cut_type: [multiprocessing.RawArray("d",
                                                        [0] * (self.msp.n_states[t] + 1))
                               for _ in range(self.n_steps)]
                    for cut_type in self.cut_type_list[t]}
                for t in range(self.cut_T)}
        else:
            cuts = {
                t: {
                    cut_type: [
                        [multiprocessing.RawArray("d",
                                                  [0] * (self.msp.n_states[t] + 1))
                         for _ in range(self.msp.n_Markov_states[t])]
                        for _ in range(self.n_steps)]
                    for cut_type in self.cut_type_list[t]}
                for t in range(self.cut_T)}

        pv = multiprocessing.Array("d", [0] * self.n_steps)
        lock = multiprocessing.Lock()
        forward_solution = None
        # regularization needs to store last forward_solution
        if self.rgl_a != 0:
            forward_solution = [multiprocessing.Array(
                "d", [0] * self.msp.n_states[t])
                for t in range(self.forward_T)
            ]

        for p in range(self.n_processes):
            procs[p] = multiprocessing.Process(
                target=self._SDDP_single_process,
                args=(pv, self.jobs[p], lock, cuts, forward_solution),
            )
            procs[p].start()
        for proc in procs:
            proc.join()

        self._add_cut_from_multiprocessing_array(cuts)
        # regularization needs to store last forward_solution
        if self.rgl_a != 0:
            self.rgl_center = [list(item) for item in forward_solution]

        return [item for item in pv]

    def solve(
            self,
            n_processes: int = 1,
            n_steps: int = 1,
            max_iterations: int = 10000,
            max_stable_iterations: int = 10000,
            max_time: float = 1000000.0,
            tol: float = 0.001,
            freq_evaluations = None,
            percentile: int = 95,
            tol_diff: float = float("-inf"),
            freq_comparisons: int = None,
            n_simulations: int = 3000,
            query: list = None,
            query_T=None,
            query_dual=None,
            query_stage_cost=False,
            query_policy_value=False,
            freq_clean: int | list = None,
            logFile: bool = True,
            logToConsole: bool = True,
            directory: str = '',
            rgl_norm = 'L2',
            rgl_a: float = 0,
            rgl_b: float = 0.95,
    ):
        """
        Solve the discretized problem.

        Args:
          rgl_b:
          rgl_a:
          rgl_norm:
          directory: the output directory of the logger and csv files
          query_policy_value:
          query_stage_cost:
          query_dual:

          query_T:
          query: the vars that wants to check(query)
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

          logFile: binary, optional (default=1)
            Switch of logging to log file

          logToConsole: binary, optional (default=1)
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
        if not msp.flag_update:
            msp.update()
        stable_iterations = 0
        total_time = 0
        a = time.time()
        gap = 1.0
        right_end_of_CI = float("inf")
        db_past = msp.bound
        self.percentile = percentile
        self.rgl_norm = rgl_norm
        self.rgl_a = rgl_a
        self.rgl_b = rgl_b

        # distinguish pv_sim from pv
        pv_sim_past = None

        if n_processes != 1:
            self.n_steps = n_steps
            self.n_processes = min(n_steps, n_processes)
            self.jobs = allocate_jobs(self.n_steps, self.n_processes)

        logger_sddp = LoggerSDDP(
            logFile=logFile,
            logToConsole=logToConsole,
            n_processes=self.n_processes,
            percentile=self.percentile,
            directory = directory,
        )
        logger_sddp.header()
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation = LoggerEvaluation(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_evaluation.header()
        if freq_comparisons is not None:
            logger_comparison = LoggerComparison(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_comparison.header()
        try:
            while (
                    self.iteration < max_iterations
                    and total_time < max_time
                    and stable_iterations < max_stable_iterations
                    and tol < gap
                    and (tol_diff < right_end_of_CI or right_end_of_CI < 0)
            ):
                start = time.time()

                self._compute_cut_type()

                if self.n_processes == 1:
                    pv = self._SDDP_single()
                else:
                    pv = self._SDDP_multiprocessesing()

                m = (
                    msp.models[0]
                    if msp.n_Markov_states == 1
                    else msp.models[0][0]
                )
                m.optimize()
                if m.status not in [2, 11]:
                    m.write_infeasible_model(
                        "backward_" + str(m._model.modelName) + ".lp"
                    )
                db = m.objBound
                self.db.append(db)
                msp.db = db
                if self.n_processes != 1:
                    CI = compute_CI(pv, percentile)
                self.pv.append(pv)

                if self.iteration >= 1:
                    if db_past == db:
                        stable_iterations += 1
                    else:
                        stable_iterations = 0
                self.iteration += 1
                db_past = db

                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time

                if self.n_processes == 1:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        pv=pv[0],
                        time=elapsed_time,
                    )
                else:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        CI=CI,
                        time=elapsed_time,
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
                        n_simulations=n_simulations,
                        query=query,
                        query_T=query_T,
                        query_dual=query_dual,
                        query_stage_cost=query_stage_cost,
                        percentile=percentile,
                        n_processes=n_processes,
                    )
                    if query_policy_value:
                        pandas.DataFrame(evaluation.pv).to_csv(directory +
                                                               "iter_{}_pv.csv".format(self.iteration))
                    if query is not None:
                        for item in query:
                            evaluation.solution[item].to_csv(directory +
                                                             "iter_{}_{}.csv".format(self.iteration, item))
                    if query_dual is not None:
                        for item in query_dual:
                            evaluation.solution_dual[item].to_csv(directory +
                                                                  "iter_{}_{}.csv".format(self.iteration, item))
                    if query_stage_cost:
                        evaluation.stage_cost.to_csv(directory +
                                                     "iter_{}_stage_cost.csv".format(self.iteration))
                    if evaluation_true:
                        evaluationTrue = EvaluationTrue(msp)
                        evaluationTrue.run(
                            n_simulations=n_simulations,
                            query=query,
                            query_T=query_T,
                            query_dual=query_dual,
                            query_stage_cost=query_stage_cost,
                            percentile=percentile,
                            n_processes=n_processes,
                        )
                        if query_policy_value:
                            pandas.DataFrame(evaluationTrue.pv).to_csv(directory +
                                                                       "iter_{}_pv_true.csv".format(self.iteration))
                        if query is not None:
                            for item in query:
                                evaluationTrue.solution[item].to_csv(directory +
                                                                     "iter_{}_{}_true.csv".format(self.iteration, item))
                        if query_dual is not None:
                            for item in query_dual:
                                evaluationTrue.solution_dual[item].to_csv(directory +
                                                                          "iter_{}_{}_true.csv".format(self.iteration,
                                                                                                       item))
                        if query_stage_cost:
                            evaluationTrue.stage_cost.to_csv(directory +
                                                             "iter_{}_stage_cost_true.csv".format(self.iteration))
                    elapsed_time = time.time() - start
                    gap = evaluation.gap
                    if n_simulations == -1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.epv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    elif n_simulations == 1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.pv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    else:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            CI=evaluation.CI,
                            gap=gap,
                            time=elapsed_time,
                        )
                if (
                        freq_comparisons is not None
                        and self.iteration % freq_comparisons == 0
                ):
                    start = time.time()
                    pv_sim = evaluation.pv
                    if self.iteration / freq_comparisons >= 2:
                        diff = msp.sense * (numpy.array(pv_sim_past) - numpy.array(pv_sim))
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
                    pv_sim_past = pv_sim
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
        msp.db = self.db[-1]
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

        b = time.time()
        logger_sddp.footer(reason=stop_reason)
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation.footer()
        if freq_comparisons is not None:
            logger_comparison.footer()
        self.total_time = total_time

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
        return plot_bounds(self.db, self.pv, self.msp.sense, self.percentile,
                           start=start, window=window, smooth=smooth, ax=ax)

    @property
    def bounds(self):
        """dataframe of the obtained bound"""
        df = pandas.DataFrame.from_records(self.pv)
        df['db'] = self.db
        return df
