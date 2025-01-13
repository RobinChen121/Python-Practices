#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:49:14 2025

@author: zhen chen

@Python version: 3.10

@disp:  
    multi stage models;
    
"""
from sm_detail import StochasticModel
from numpy.typing import ArrayLike
from statistics import check_Markov_states_and_transition_matrix
from statistics import check_Markov_callable_uncertainty
import numpy
from itertools import product
from collections.abc import Callable


class MSP:
    """
    A class of multi-stage programming model;
    """

    def __init__(self,
                 T: int,
                 bound: float = None,
                 sense: int = 1,
                 outputFlag: int = 0,
                 discount: float = 1.0,
                 **kwargs) -> None:
        """
        Initialize the MSP class.

        Args:
            T (int): the number of stages.
            bound (float, optional): A known uniform lower bound or upper bound for each stage problem.
                                     Default value is 1 billion for maximization problem and -1 billion for minimization problem. 
            sense (int, optional): model optimization sense. 1 means minimization and -1 means maximization. Defaults to 1.
            outputFlag (int, optional): Enables or disables gurobi solver output. Defaults to 0.
            discount (flot, optional): The discount factor used to compute present value.
                                       float between 0(exclusive) and 1(inclusive). Defaults to 1.0.

        Returns:
            A class of multi-stage linear programming model

        """
        self.n_samples = [] # number of samples for each stage
        self.n_states = [] # number of states for each stage
        self.Markovian_uncertainty_function = None # random generator function
        self.models = None
        self.T = T
        self.bound = bound
        self.sense = sense
        self.discount = discount

        self.measure = 'risk neutral'
        self._type = 'stage-wise independent'
        self._flag_discrete = 0
        self._individual_type = 'original'
        self.Markov_states = None
        self.transition_matrix = None
        self.dim_Markov_states = {}
        self.n_Markov_states = 1 # the default 1 meaning this class is not Markovian

        self._set_default_bound()
        self._set_model()

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
            sample_paths = list(product(sample_paths, Markov_state_paths)) # chen: I need to fully understand when coming across the corresponding problems
        return len(sample_paths), sample_paths

    def add_MC_uncertainty_discrete(
            self,
            Markov_states: ArrayLike,
            transition_matrix: ArrayLike
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
        if self.Markovian_uncertainty is None or self.Markov_states is None:
            raise ValueError("Markovian uncertainty has already added!")
        info = check_Markov_states_and_transition_matrix(
            Markov_states, transition_matrix, self.T
        )
        self.dim_Markov_states, self.n_Markov_states = info
        self.Markov_states = Markov_states
        self.transition_matrix = [numpy.array(item) for item in transition_matrix]
        self._type = 'Markov-discrete'

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
        self._type = 'Markov-continuous'

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
                    if m.flag_discrete == 1:
                        self._individual_type = "discretized"

    def check_markov_and_update_num_states_samples(self) -> None:
        """
        Check Markovian uncertainties are discretized.
        Copy Stochastic Models for every Markov states.
        Update the number of states and samples in the class.

        """
        if self._type == "Markovian-continuous" and self._flag_discrete == 0:
            raise Exception("Markovian uncertainties must be discretized!")
        if self._type == "Markov-discrete" or (
            self._type == "Markov-continuous" and self._flag_discrete == 1
        ):
            if type(self.models[0]) != list:
                models = self.models
                self.models = [
                    [None for k in range(self.n_Markov_states[t])]
                    for t in range(self.T)
                ]
                for t in range(self.T):
                    m = models[t]
                    for k in range(self.n_Markov_states[t]):
                        m.update_uncertainty_dependent(self.Markov_states[t][k])
                        m.update()
                        self.models[t][k] = m.copy() # copy the model m
        self.n_states = (
            [self.models[t].n_states for t in range(self.T)]
            if self._type == 'stage-wise independent'
            else [self.models[t][0].n_states for t in range(self.T)]
        )
        self.n_samples = (
            [self.models[t].n_samples for t in range(self.T)]
            if self._type == 'stage-wise independent'
            else [self.models[t][0].n_samples for t in range(self.T)]
        )


        