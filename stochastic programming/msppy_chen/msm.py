#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:49:14 2025

@author: zhen chen

@Python version: 3.10

@disp:  
    multi stage models;
    
"""
import math
from sm_detail import StochasticModel, StochasticModelLG
from utils.statistics import check_Markov_states_and_transition_matrix
from utils.statistics import check_Markov_callable_uncertainty, check_random_state
from utils.exception import MarkovianDimensionError
import numpy
from numpy.typing import ArrayLike
from itertools import product
from collections.abc import Callable
from collections import abc
import numbers
import gurobipy


# noinspection PyUnresolvedReferences
class MSLP:
    """
    A class of multi-stage linear programming model;
    """

    bound = None

    def __init__(self,
                 T: int,
                 bound: float = None,
                 sense: int = 1,
                 outputLogFlag: bool = False,
                 discount: float = 1.0,
                 flag_CTG: bool = False,
                 **kwargs) -> None:
        """
        Initialize the MSP class.

        Args:
            T (int): The number of stages.
            bound (float, optional): A known uniform lower bound or upper bound for each stage problem.
                                     Default value is 1 billion for maximization problem and -1 billion for minimization problem. 
            sense (int, optional): Model optimization sense. 1 means minimization and -1 means maximization. Defaults to 1.
            outputLogFlag (int, optional): enables or disables solver output. Use LogFile and LogToConsole for finer-grain control.
                                        Setting outputLogFlag to 0 is equivalent to setting LogFile to "" and LogToConsole to 0.
            discount (flot, optional): The discount factor used to compute present value.
                                       float between 0(exclusive) and 1(inclusive). Defaults to 1.0.
            flag_CTG: whether setting CTG in the model.

        Returns:
            A class of multi-stage linear programming model

        """
        self.bound = bound  # A known uniform lower bound or upper bound for each stage problem

        if (T < 2
                or discount > 1
                or discount < 0
                or sense not in [-1, 1]
                or outputLogFlag not in [0, 1]):
            raise Exception('Arguments of SDDP construction are not valid!')

        self.flag_CTG = flag_CTG # whether setting CTG in the model
        self.a = 0 # used for AVAR(CVAR)
        self.l = 0 # used for AVAR
        self.n_samples = [] # number of samples for each stage
        self.n_states = [] # number of states for each stage
        self.Markovian_uncertainty_function = None # random generator function

        self.models = None
        self.T = T
        self.sense = sense
        self.discount = discount

        self.measure = 'risk neutral'
        self.type = 'stage-wise independent'
        self._flag_discrete = 0
        self.individual_type = 'original'
        self.Markov_states = None
        self.Markovian_uncertainty = None
        self.transition_matrix = None
        self.dim_Markov_states = {}
        self.n_Markov_states = 1 # the default 1 meaning this class is not Markovian

        self._set_default_bound()
        self._set_model()
        self._set_up_model_attr(sense, outputLogFlag, kwargs)

        self.flag_updated: bool = False # whether the model has been updated
        self.flag_infinity: bool = False # whether

    def _check_first_stage_deterministic(self):
        """
        Ensure the first stage model is deterministic. The First stage model
        is only allowed to have uncertainty with length one.

        """
        m = self.models[0] if type(self.models[0]) != list else self.models[0][0]
        if m.n_samples != 1:
            raise Exception("First stage must be deterministic!")
        # else:
        #     m._update_uncertainty(0)
        #     m.update()

    def __getitem__(self,
                    t: int
                    ) -> StochasticModel:
        """

        Args:
            t: stage index

        Returns:
            StochasticModel class at one stage t
        """
        return self.models[t]
        
    def _set_default_bound(self):
        """
        Set the default bound for this multi-stage model.

        Returns:
            None.

        """
        if self.bound is None:
            self.bound = -1000000000 if self.sense == 1 else 1000000000

    def _set_model(self):
        """
        Set up the detailed gurobi solvable model for each stage

        Returns:
            None.

        """
        self.models = [StochasticModel(name = str(t)) for t in range(self.T)]

    def _set_up_model_attr(self, sense, outputLogFlag, kwargs):
        for t in range(self.T):
            m = self.models[t]
            m.Params.LogToConsole = outputLogFlag
            m.setAttr('modelsense', sense)
            for k, v in kwargs.items():
                m.setParam(k, v)

    def _set_up_link_constrs(self):
        """
            set up the local copies-link constraints for each stage
            # model copies may not be ready while state size may have changed

        """

        for t in range(1, self.T):
            M = (
                self.models[t]
                if type(self.models[t]) == list
                else [self.models[t]]
            )
            for m in M:
                m.set_up_link_constrs()
                m.update()

    def _delete_link_constrs(self):
        """
            delete the local copies-link constraints for each stage
            # model copies may not be ready while state size may have changed

        """
        for t in range(1, self.T):
            M = (
                self.models[t]
                if type(self.models[t]) == list
                else [self.models[t]]
            )
            for m in M:
                m.delete_link_constrs()
                m.update()

    def _set_up_CTG(self):
        """
            add alpha as a decision variable in the model of each stage
        """
        for t in range(self.T - 1):
            self._set_up_CTG_for_t(t)

    def _set_up_CTG_for_t(self, t):
        M = (
            [self.models[t]]
            if type(self.models[t]) != list
            else self.models[t]
        )
        for m in M:
            m.set_up_CTG(discount = self.discount, bound = self.bound)
            m.update()

    def _set_up_probability(self):
        """
        Return uniform measure if no given probability measure

        """
        if self.n_Markov_states == 1:
            probability = [None for _ in range(self.T)]
            for t in range(self.T):
                m = self.models[t]
                if m.probability is not None:
                    probability[t] = m.probability
                else:
                    probability[t] = [
                        1.0/m.n_samples for _ in range(m.n_samples)
                    ]
        else:
            probability = [
                [None for _ in range(self.n_Markov_states[t])]
                for t in range(self.T)
            ]
            for t in range(self.T):
                for k in range(self.n_Markov_states[t]):
                    m = self.models[t][k]
                    if m.probability is not None:
                        probability[t][k] = m.probability
                    else:
                        probability[t][k] = [
                            1.0/m.n_samples for _ in range(m.n_samples)
                        ]
        return probability

    def enumerate_sample_paths(self, T: int, start: int = 0, flag_rolling: bool = 0) -> tuple[int, list]:
        """
        Enumerate all sample paths (three cases: pure stage-wise independent
        , pure Markovian, and mixed type)

        sample paths are actually the index of the sampled scenario in the sample space of each stage.

        Args:
            T: The number of stages
            start: The Starting stage
            flag_rolling: Whether it is rolling horizon computation

        Returns:
            A tuple including the number of sample paths and lists of the detailed sample paths
        """
        if self.n_Markov_states == 1: # meaning this model is not Markovian
            sample_paths = list(
                # The * "unpacks" an iterable, so that each element is passed as a separate argument
                product(*[range(self.models[t].n_samples) for t in range(start, T + 1)]) # get the Carteisian product of several arrays
            )
        else:
            sample_paths = (
                product(*[range(self.models[t][0].n_samples) for t in range(start, T + 1)])
            )
            if flag_rolling == 0:
                Markov_state_paths = (
                    product(*[range(self.n_Markov_states[t])
                        for t in range(start, T + 1)])
                )
            else:
                n_branches = self.n_Markov_states[start + 1]
                Markov_state_paths = ( # if rolling, the sample path actually starts from the next of the start stage
                    zip([0]*n_branches, # zip() function returns a tuple where the items in the two list are paired together
                        *[range(n_branches) for t in range(start + 1, T + 1)])
                    if start < T
                    else [(0,)]
                )
            # n_sample_paths = n_Markov_state_paths * n_sample_paths
            sample_paths = list(product(sample_paths, Markov_state_paths)) # chen: index 1 of sample_paths is the markov state index
        return len(sample_paths), sample_paths

    def add_MC_uncertainty_discrete(
            self,
            Markov_states: list[list[list[float]]],
            transition_matrix: list[list[list[float]]]
        ) -> None:
        """
        Add a Markov chain process -- discrete uncertainty.

        Args:
            Markov_states: Detailed value in matrix form of markov states,
                           The shape of matrix-like must be (p,q) where q is
                           the dimension index of the Markov chain and p is the
                           index of the Markov states.
            transition_matrix: Transition probabilities. Its shape should be compatiable
                               Markov_states.

        Examples:
        --------
        One-dimensional Markov chian

        >>> add_MC_uncertainty(
        ...     Markov_states=[[[0]],[[4],[6]],[[4],[6]]],
        ...     transition_matrix=[
        ...         [[1]],
        ...         [[0.5,0.5]],
        ...         [[0.3,0.7], [0.7,0.3]]
        ...     ]
        ... )

        Three-dimensional Markov chain # chen: a little weird

        >>> add_MC_uncertainty(
        ...     Markov_states=[[[0]],[[4,6,5],[6,3,4]],[[4,6,5],[6,3,4]]],
        ...     transition_matrix=[
        ...         [[1]],
        ...         [[0.5,0.5]],
        ...         [[0.3,0.7], [0.7,0.3]]
        ...     ]
        ... )
        """
        if self.Markovian_uncertainty is not None and self.Markov_states is not None:
            raise ValueError("Markovian uncertainty has already added!")
        info = check_Markov_states_and_transition_matrix(
            Markov_states, transition_matrix, self.T
        )
        self.dim_Markov_states, self.n_Markov_states = info
        self.Markov_states = Markov_states
        self.transition_matrix = [numpy.array(item) for item in transition_matrix]
        self.type = 'Markov-discrete'

    def add_MC_uncertainty_continuous(self, Markovian_uncertainty: Callable):
        """
        Add a Markovian process - continuous uncertainty.

        Args:
            Markovian_uncertainty: A callable sample path generator.
            The callable should take
            numpy.random.randomState and size as its parameters.
            It should return a three-dimensional numpy array
            (n_samples * T * n_states)

        Example:
        -------
        >>> def f(random_state, size):
        ...     a = numpy.empty([size, 3, 2])
        ...     a[:,0,:] = [[0.2, 0.2]]
        ...     for t in range(1, 3):
        ...         a[:, t, :] = (
        ...             0.5 * numpy.array(a[:, t - 1, :])
        ...             + random_state.multivariate_normal(
        ...                 mean = [0, 0],
        ...                 cov = [[0, 1], [1, 0]],
        ...                 size = size,
        ...                )
        ...         )
        ...     return a
        >>> add_Markovian_uncertainty(f, 10)
        """
        if self.Markovian_uncertainty is None or self.Markov_states is None:
            raise ValueError("Markovian uncertainty has already added!")
        self.dim_Markov_states = check_Markov_callable_uncertainty(Markovian_uncertainty, self.T)
        self.Markovian_uncertainty_function = Markovian_uncertainty
        self.type = 'Markov-continuous'

    def check_state_and_continuous_discretized(self):
        """
        Check stage-wise continuous uncertainties are discretized.

        chen: this functions seems not have much effect currently

        """
        m = self.models[0] if type(self.models[0]) != list else self.models[0][0]
        if not m.states: # m.states is empty
            raise Exception("State variables must be set!")
        for t in range(1, self.T):
            ms = self.models[t] if type(self.models[t]) == list else [self.models[t]]
            for m in ms:
                if m.type == "continuous":
                    if m.flag_discrete == 0:
                        raise Exception(
                            "Stage-wise independent continuous uncertainties "+
                            "must be discretized!"
                        )
                    self._individual_type = "discretized"
                else:
                    if m.flag_discretized == 1:
                        self._individual_type = "discretized"

    def check_markov_copy_models_update_nums(self) -> None:
        """
        Check Markovian uncertainties are discretized.
        Copy Stochastic Models for every Markov states.
        Update the number of states and samples in the class.

        """
        if self.type == "Markovian-continuous" and self._flag_discrete == 0:
            raise Exception("Markovian uncertainties must be discretized!")
        if self.type == "Markov-discrete" or (
            self.type == "Markov-continuous" and self._flag_discrete == 1
        ):
            if type(self.models[0]) != list:
                models = self.models
                self.models = [
                    [None for _ in range(self.n_Markov_states[t])]
                    for t in range(self.T)
                ]
                for t in range(self.T):
                    m = models[t]
                    for k in range(self.n_Markov_states[t]):
                        m.update_uncertainty_dependent(self.Markov_states[t][k])
                        m.update()
                        self.models[t][k] = m.copy() # copy the model m, should use deep copy
                                                     # without the deep copy, the markov models
                                                     # in one stage may share same vars or constraints
                    if t == 3:
                        pass

        self.n_states = (
            [self.models[t].n_states for t in range(self.T)]
            if self.type == 'stage-wise independent'
            else [self.models[t][0].n_states for t in range(self.T)]
        )

        self.n_samples = (
            [self.models[t].n_samples for t in range(self.T)]
            if self.type == 'stage-wise independent'
            else [self.models[t][0].n_samples for t in range(self.T)]
        )

    def check_sample_path_dimension(self):
        """
        Check dimension indices of sample path generator are set properly.

        """
        for t in range(self.T):
            M = (
                self.models[t]
                if type(self.models[t]) == list
                else [self.models[t]]
            )
            for m in M:
                if m.Markovian_dim_index:
                    if any(index not in range(self.dim_Markov_states[t])
                            for index in m.Markovian_dim_index):
                        raise MarkovianDimensionError

    def compute_weight_sample_path(self, sample_path: list | list[list], start: int = 0) -> float:
        """
        Compute weight/probability of (going through) a certain sample path.

        Args:
            sample_path: indices of all the realizations in a scenario
            start: the starting stage

        Returns:
            The weight of the sample path

        """
        probability = self._set_up_probability()
        T = (
            start + len(sample_path)
            if self.n_Markov_states == 1
            else start + len(sample_path[0])
        )
        if self.n_Markov_states == 1:
            weight = numpy.prod(
                [probability[t][sample_path[t - start]] for t in range(start, T)]
            )
        else:
            weight = numpy.prod(
                [
                    self.transition_matrix[t][sample_path[1][t - 1 - start]][
                        sample_path[1][t - start] # the transition probability of this markov sample path
                    ]
                    for t in range(start + 1, T)
                ]
            )
            weight *= numpy.prod(
                [   # for Markov situation, the index 1 is the Markov states index, and 0 is the uncertainty index
                    probability[t][sample_path[1][t - start]][sample_path[0][t - start]]
                    for t in range(start, T)
                ]
            )
        return float(weight)

    def compute_current_node_weight(self, sample_path: list | list[list]) -> float:
        """
        Compute the weight/probability of the last node in a sample_path

        Args:
            sample_path: indices of all the realizations in a scenario

        """
        probability = self._set_up_probability()
        t = (
            len(sample_path) - 1
            if self.n_Markov_states == 1
            else len(sample_path[0]) - 1
        )
        if self.n_Markov_states == 1:
            weight = probability[t][sample_path[t]]
        else:
            weight = (
                self.transition_matrix[t][sample_path[1][t - 1]][
                    sample_path[1][t]
                ]
                if t > 0
                else 1
            )
            weight *= probability[t][sample_path[1][t]][sample_path[0][t]]
        return weight

    def discretize(
            self,
            n_samples: int = None,
            random_state: numpy.random.RandomState | int | None = None,
            replace: bool = True,
            n_Markov_states=None,
            method: str = 'SA',
            n_sample_paths=None,
            Markov_states=None,
            transition_matrix=None,
            int_flag=0):
        """
        Discretize Markovian continuous uncertainty by k-means or (robust)
        stochastic approximation.

        Parameters
        ----------
        n_samples: int, optional, default=None
            number of i.i.d. samples to generate for stage-wise independent
            randomness.

        random_state: None | int | instance of RandomState, optional, default=None
            If int, random_state is the seed used by the
            random number generator;
            If RandomState instance, random_state is the
            random number generator;
            If None, the random number generator is the
            RandomState instance used by numpy.random.

        replace: bool, optional, default=True
            Indicates generating i.i.d. samples with/without replacement for
            stage-wise independent randomness.

        n_Markov_states: list | int, optional, default=None
            If list, it specifies different dimensions of Markov state space
            over time. Length of the list should equal length of the Markovian
            uncertainty.
            If int, it specifies dimensions of Markov state space.
            Note: If the uncertainties are int, trained Markov states will be
            rounded to integers, and duplicates will be removed. In such cases,
            there is no guarantee that the number of Markov states is n_Markov_states.

        method: binary, optional, default=0
            'input': the approximating Markov chain is given by user input (
            through specifying Markov_states and transition_matrix)
            'SAA': use k-means to train Markov chain.
            'SA': use stochastic approximation to train Markov chain.
            'RSA': use robust stochastic approximation to train Markov chain.

        n_sample_paths: int, optional, default=None
            number of sample paths to train the Markov chain.

        Markov_states/transition_matrix: matrix-like, optional, default=None
            The user input of approximating Markov chain.
        """
        if n_samples is not None:
            if isinstance(n_samples, (numbers.Integral, numpy.integer)):
                if n_samples < 1:
                    raise ValueError("n_samples should be bigger than zero!")
                n_samples = (
                    [1]
                    +[n_samples] * (self.T-1)
                )
            elif isinstance(n_samples, (abc.Sequence, numpy.ndarray)):
                if len(n_samples) != self.T:
                    raise ValueError(
                        "n_samples list should be of length {} rather than {}!"
                        .format(self.T,len(n_samples))
                    )
                if n_samples[0] != 1:
                    raise ValueError(
                        "The first stage model should be deterministic!"
                    )
            else:
                raise ValueError("Invalid input of n_samples!")
            # discretize stage-wise independent continuous distribution
            random_state = check_random_state(random_state)
            for t in range(1,self.T):
                self.models[t].discretize(n_samples[t],random_state,replace)
        if n_Markov_states is None and method != 'input': return
        if method == 'input' and (Markov_states is None or
            transition_matrix is None): return
        if n_Markov_states is not None:
            if isinstance(n_Markov_states, (numbers.Integral, numpy.integer)):
                if n_Markov_states < 1:
                    raise ValueError("n_Markov_states should be bigger than zero!")
                n_Markov_states = (
                    [1]
                    +[n_Markov_states] * (self.T-1)
                )
            elif isinstance(n_Markov_states, (abc.Sequence, numpy.ndarray)):
                if len(n_Markov_states) != self.T:
                    raise ValueError(
                        "n_Markov_states list should be of length {} rather than {}!"
                        .format(self.T,len(n_Markov_states))
                    )
                if n_Markov_states[0] != 1:
                    raise ValueError(
                        "The first stage model should be deterministic!"
                    )
            else:
                raise ValueError("Invalid input of n_Markov_states!")
        from msppy.discretize import Markovian
        if method in ['RSA','SA','SAA']:
            markovian = Markovian(
                f=self.Markovian_uncertainty,
                n_Markov_states=n_Markov_states,
                n_sample_paths=n_sample_paths,
                int_flag=int_flag,
            )
        if method in ['RSA','SA','SAA']:
            self.Markov_states,self.transition_matrix = getattr(markovian, method)()
        elif method == 'input':
            dim_Markov_states, n_Markov_states = (
                check_Markov_states_and_transition_matrix(
                    Markov_states=Markov_states,
                    transition_matrix=transition_matrix,
                    T=self.T,
                )
            )
            if dim_Markov_states != self.dim_Markov_states:
                raise ValueError("The dimension of the given sample path "
                    +"generator is not the same as the given Markov chain "
                    +"approximation!")
            self.Markov_states = Markov_states
            self.transition_matrix = [numpy.array(item) for item in transition_matrix]
        self._flag_discrete = 1
        self.n_Markov_states = n_Markov_states
        if method in ['RSA','SA','SAA']:
            return markovian

    def get_stage_cost(self, m: StochasticModel, t: int) -> float:
        """
            get the stage cost
        Args:
            t: the stage index
            m: an instance of StochasticModel

        """
        if self.measure == "risk neutral":
            if m.alpha is not None:
                return pow(self.discount, t) * (
                    m.objVal - self.discount*m.alpha.X
                )
            else:
                return pow(self.discount, t) * m.objVal
        else:
            return pow(self.discount,t) * m.getVarByName("stage_cost").X

    @staticmethod
    def get_state_solution(m: StochasticModel, t: int) -> list[float]:
        """
            get the solutions of state variables at one stage
        Args:
            t: the stage index
            m: an instance of StochasticModel

        """
        solution = [0.0 for _ in m.states]
        # avoid numerical issues
        for idx, var in enumerate(m.states):
            if var.vtype in ['B','I']:
                solution[idx] = int(round(var.X))
            else:
                if var.X < var.lb:
                    solution[idx] = var.lb
                elif var.X > var.ub:
                    solution[idx] = var.ub
                else:
                    solution[idx] = var.X
        return solution

    def update(self):
        self._check_first_stage_deterministic()
        self.check_state_and_continuous_discretized()
        self.check_markov_copy_models_update_nums()

        self._set_up_CTG()
        self._set_up_link_constrs()
        self.check_markov_copy_models_update_nums()
        self.flag_updated = 1

    def set_AVaR(self, l: float | ArrayLike, a: float | ArrayLike, method: str = 'indirect') -> None:
        """
        Set linear combination of expectation and conditional value at risk
        (average value at risk) as risk measure

        Args:
            method: 'direct'/'indirect'
                    direct method directly solves the risk-averse problem;
                    indirect method adds additional state variables and transform the
                    risk-averse problem into risk neutral.

            l: float between 0 and 1/array-like of floats between 0 and 1.
               The weights of AVaR from stage 2 to stage T
               If floated, the weight will be assigned to the same number.
               If array-like, must be of length T-1 (for finite horizon problem)
               or T (for infinite horizon problem).

            a: float between 0 and 1/array-like of floats between 0 and 1
               The quantile parameters in value-at-risk from stage 2 to stage T
               If floated, those parameters will be assigned to the same number.
               If array-like, must be of length T-1 (for finite horizon problem)
               or T (for infinite horizon problem).

        Notes:
            Bigger l means more risk-averse, l = 1 means the objective is to fully minimize AVAR.
            smaller a means more risk-averse, the smaller a means larger values of AVAR (to be certified)
        """
        if isinstance(l, (abc.Sequence, numpy.ndarray)):
            if len(l) not in [self.T - 1, self.T]:
                raise ValueError("Length of l must be T-1/T!")
            if not all(1 >= item >= 0 for item in l): # nice coding
                raise ValueError("l must be between 0 and 1!")
            l = [None] + list(l)
        elif isinstance(l, numbers.Number):
            if l > 1 or l < 0:
                raise ValueError("l must be between 0 and 1!")
            l = [None] + [l] * (self.T - 1)
        else:
            raise TypeError("l should be float/array-like instead of {}!".format(type(l)))
        if isinstance(a, (abc.Sequence, numpy.ndarray)):
            if len(a) not in [self.T - 1, self.T]:
                raise ValueError("Length of a must be T-1!")
            if not all(0 <= item <= 1 for item in a):
                raise ValueError("a must be between 0 and 1!")
            a = [None] + list(a)
        elif isinstance(a, numbers.Number):
            if a > 1 or a < 0:
                raise ValueError("a must be between 0 and 1!")
            a = [None] + [a] * (self.T - 1)
        else:
            raise TypeError("a should be float/array-like instead of {}!".format(type(a)))
        self.a = a
        self.l = l

        if method == 'direct':
            from msppy_chen.utils.measure import Expectation_AVaR
            from functools import partial
            for t in range(1, self.T):
                M = (
                    self.models[t]
                    if type(self.models[t]) == list
                    else [self.models[t]]
                )
                for m in M:
                    # change the measure function to be AVAR
                    m.measure = partial(Expectation_AVaR, a = a[t], l = l[t])
            for t in range(self.T):
                M = (
                    self.models[t]
                    if type(self.models[t]) == list
                    else [self.models[t]]
                )
                for m in M:
                    stage_cost = m.addVar(
                        name = "stage_cost",
                        lb = -gurobipy.GRB.INFINITY,
                        ub = gurobipy.GRB.INFINITY,
                    )
                    # For CTG, it is an alpha subtracted in the constraints
                    # and objective function, and the stage cost is computed
                    # as a variable.
                    if self.flag_CTG:
                        self._set_up_CTG()  # add alpha in the model
                    alpha = m.alpha if m.alpha is not None else 0.0
                    # foe ease of get the stage cost in the AVAR model
                    m.addConstr(m.getObjective() - self.discount * alpha == stage_cost)
                    m.update()

        # the indirect may be seldom used
        elif method == 'indirect': # add additional constraints and variables for the risk-averse computation
            self._set_up_CTG()
            self._delete_link_constrs()
            for t in range(self.T):
                M = (
                    self.models[t]
                    if type(self.models[t]) == list
                    else [self.models[t]]
                )
                for m in M:
                    p_now, p_past = m.addStateVar(
                        lb = -gurobipy.GRB.INFINITY,
                        ub = gurobipy.GRB.INFINITY,
                        name = "additional_state",
                    )
                    v = m.addVar(name = "additional_var")
                    m.addConstr(self.sense * (p_now - self.bound) >= 0)
                    z = m.getObjective()
                    stage_cost = m.addVar(
                        name = "stage_cost",
                        lb = -gurobipy.GRB.INFINITY,
                        ub = gurobipy.GRB.INFINITY,
                    )
                    alpha = m.alpha if m.alpha is not None else 0.0
                    if t > 0:
                        if m.uncertainty_obj != {}:
                            m.addConstr(
                                z - self.discount * alpha == stage_cost,
                                uncertainty = m.uncertainty_obj,
                            )
                            m.uncertainty_obj = {}
                            m.setObjective(
                                (1 - l[t])
                                * (
                                        stage_cost
                                        + self.discount * alpha
                                )
                                + l[t] * p_past
                                + self.sense * v * l[t] / a[t]
                            )
                            m.addConstr(
                                v
                                >= (
                                        stage_cost
                                        + self.discount * alpha
                                        - p_past
                                )
                                * self.sense
                            )
                        else: # if no uncertainty in the objective
                              # z is the model's original objective
                            m.addConstr(z - self.discount * alpha == stage_cost)
                            m.setObjective(
                                (1 - l[t]) * z
                                + l[t] * p_past
                                + self.sense * v * l[t] / a[t]
                            )
                            m.addConstr(
                                v
                                >= (z - p_past)
                                * self.sense
                            )
                    else:
                        m.addConstr(z - self.discount * alpha == stage_cost)
                    m.update()
        else:
            raise NotImplementedError
        self.measure = "risk averse"

class MSIP(MSLP):
    n_binaries = []
    precision = 0
    bin_stage = 0

    def _set_model(self):
        self.models = [StochasticModelLG(name = str(t)) for t in range(self.T)]

    def _check_individual_stage_models(self):
        """
        Check state variables are set properly. Check stage-wise continuous
        uncertainties are discretized.

        """
        if not hasattr(self, "bin_stage"):
            self.bin_stage = 0
        M = self.models[0]
        N = (
            self.models[self.bin_stage - 1]
            if self.bin_stage not in [0, self.T]
            else self.models[0]
        )
        if not M.states:
            raise Exception("State variables must be set!")
        if not N.states:
            raise Exception("State variables must be set!")
        n_states_binary_space = M.n_states
        n_states_original_space = N.n_states
        for t in range(self.T):
            m = self.models[t]
            if m.type == "continuous":
                self._individual_type = "continuous"
                if m.flag_discrete == 0:
                    raise Exception(
                        "stage-wise independent continuous uncertainties "
                        + "must be discretized!"
                    )
            if t < self.bin_stage - 1:
                if m.n_states != n_states_binary_space:
                    raise Exception(
                        "state spaces must be of the same dim for all stages!"
                    )
            else:
                if m.n_states != n_states_original_space:
                    raise Exception(
                        "state spaces must be of the same dim for all stages!"
                    )
        if self._type == "Markovian" and self._flag_discrete == 0:
            raise Exception(
                "stage-wise dependent continuous uncertainties "
                + "must be discretized!"
            )
        self.n_states = [self.models[t].n_states for t in range(self.T)]

    def _check_MIP(self):
        self.isMIP = [0] * self.T
        for t in range(self.T):
            if self.models[t].isMIP == 1:
                self.isMIP[t] = 1

    def binarize(self, precision: int = 0, bin_stage: int = 0) -> None:
        """
        Binarize MSIP.

        The number of binary variables at each stage are same.

        Parameters
        ----------
        precision: int, optional (default=0)
            The number of decimal places of accuracy

        bin_stage: int, optional (default=0)
            Stage index, in which all stage models before bin_stage (exclusive) will be binarized.
        """
        # bin_stage should be within [0, self.T]
        self.bin_stage = int(bin_stage)
        self.bin_stage = min(self.bin_stage, self.T)
        precision = int(precision)
        self.precision = 10 ** precision
        # Binarize the model if bin_stage is not 0
        if self.bin_stage != 0:
            self.n_binaries = [] # number of binaries in each stage
        # Check MSIP is qualified for binarization
        for t in range(self.bin_stage):
            n_binaries = []
            m = (
                self.models[t][0]
                if type(self.models[t]) == list
                else self.models[t]
            )
            for x in m.states:
                if ( # binarization must have bounds for the state variables
                    x.lb == -gurobipy.GRB.INFINITY
                    or x.ub == gurobipy.GRB.INFINITY
                ):
                    raise Exception("missing bounds for the state variables!")
                elif x.lb == x.ub:
                    n_binaries.append(1)
                elif x.vtype in ["B", "I"]:
                    n_binaries.append(int(math.log2(x.ub - x.lb)) + 1)
                else:
                    n_binaries.append(
                        int(math.log2(self.precision * (x.ub - x.lb))) + 1
                    )
            if not self.n_binaries: # meaning self.n_binaries is []
                self.n_binaries = n_binaries
            else:
                if self.n_binaries != n_binaries:
                    raise Exception( # this is a
                        "number of binaries should be the same over time for state variables!"
                    )
        # Binarize MSIP
        for t in range(self.bin_stage):
            M = (
                [self.models[t]]
                if self.n_Markov_states == 1
                else self.models[t]
            )
            transition = ( # bool
                True
                if t == self.bin_stage - 1 # last binary stage
                and self.bin_stage not in [0, self.T]
                else False
            )
            for m in M:
                m.binarize(self.precision, self.n_binaries, transition)

    def update(self):
        self._check_MIP()
        super().update()

    def _back_binarize(self):
        if not hasattr(self, "n_binaries"):
            return
        for t in range(self.bin_stage):
            M = (
                [self.models[t]]
                if self.n_Markov_states == 1
                else self.models[t]
            )
            transition = (
                True
                if t == self.bin_stage-1
                and self.bin_stage not in [0, self.T]
                else False
            )
            for m in M:
                m.back_binarize(self.precision)
        self._set_up_link_constrs()
        self.bin_stage = 0
        