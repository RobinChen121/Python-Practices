"""
Created on 2025/1/11, 17:29 

@author: Zhen Chen.

@Python version: 3.10

@disp:  some statistical functions

"""
import numpy.random
from numpy.typing import ArrayLike
from collections.abc import Callable


def rand_int(k: int | ArrayLike,
             random_state: numpy.random.RandomState,
             probability: ArrayLike = None,
             size: int = None,
             replace: bool = None) -> list:
    """
    Randomly generate samples from range(k) with given
    probability with/without replacement

    Args:
        k: If int, it is range(k), else it is a ArrayLike
        random_state: A instance of Numpy RandomState
        probability: Given probability
        size: The size of the output samples
        replace: sampling with replacement or not

    Returns:
        A list of random samples.

    """
    if probability is None:
        return random_state.randint(low=0, high=k, size=size)
    else:
        return random_state.choice(a=k, p=probability, size=size, replace=replace)


def check_Markov_states_and_transition_matrix(
        Markov_states: ArrayLike,
        transition_matrix: ArrayLike,
        T: int) -> tuple[list, list]:
    """
    Check Markov states and transition matrix are in the right form.

    Args:
        Markov_states: Detailed values of Markov states
        transition_matrix: Transition probability matrix.
        T: The numer of stages

    Returns:
        A tuple of the dimension of Markov states and the number of Markov states.

    Examples:
    --------
    One-dimensional Markov Chain:

    Markov_states=[[[0]],[[4],[6]],[[4],[6]]]
    transition_matrix=[
             [[1]],
             [[0.5,0.5]],
             [[0.3,0.7], [0.7,0.3]]
         ]

    Three-dimensional Markov Chain:
    Markov_states=[[[0]],[[4,6,5],[6,3,4]],[[4,6,5],[6,3,4]]],
    transition_matrix=[
        [[1]],
         [[0.5,0.5]],
        [[0.3,0.7], [0.7,0.3]]
     ]

    """
    n_Markov_states = []
    dim_Markov_states = []
    if len(transition_matrix) < T:
        raise ValueError(
            "The transition_matrix is of length {}, expecting of longer than {}!"
            .format(len(transition_matrix), T)
        )
    if len(Markov_states) < T:
        raise ValueError(
            "The Markov_states is of length {}, expecting of length longer than{}!"
            .format(len(Markov_states), T)
        )
    a = 1
    for t, item in enumerate(transition_matrix):
        if a != numpy.array(item).shape[0]:
            raise ValueError("Invalid transition_matrix!")
        else:
            a = numpy.array(item).shape[1]
            n_Markov_states.append(a)
        for single in item:
            if round(sum(single), 4) != 1:
                raise ValueError("Probability does not sum to one!")
    for t, item in enumerate(Markov_states):
        shape = numpy.array(item).shape
        if shape[0] != n_Markov_states[t]:
            raise ValueError(
                "The dimension of Markov_states is not compatible with \
                the dimension of transition_matrix!"
            )
        dim_Markov_states.append(shape[1])
    return dim_Markov_states, n_Markov_states


def check_Markov_callable_uncertainty(Markovian_uncertainty: Callable, T: int) -> list:
    """
    Check Markovian uncertainty is in the right form. Return
    the dimension of Markov states.

    Args:
        Markovian_uncertainty: numpy.random.RandomState generator
        T: The number of stages.

    """
    dim_Markov_states = []
    if not callable(Markovian_uncertainty):
        raise ValueError("Markovian uncertainty must be callable!")
    try:
        initial = Markovian_uncertainty(numpy.random, 2)
    except TypeError:
        raise TypeError("Sample path generator should always take "
                        + "numpy.random.RandomState and size as its arguments!")
    if not isinstance(initial, numpy.ndarray) or initial.ndim != 3:
        raise ValueError("Sample path generator should always return a three "
                         + "dimensional numpy array!")
    if initial.shape[1] < T:
        raise ValueError("Second dimension of sample path generator expects "
                         + "to be larger than {} rather than {}!".format(T, initial.shape[1]))
    for t in range(T):
        dim_Markov_states.append(initial.shape[2])
    return dim_Markov_states
