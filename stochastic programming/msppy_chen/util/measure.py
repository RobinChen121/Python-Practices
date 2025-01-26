"""
Created on 2025/1/25, 12:01 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""
import numpy
from numpy.typing import ArrayLike
from typing import Tuple

def Expectation(obj: ArrayLike, grad: ArrayLike, p: None | ArrayLike) -> Tuple[float, float]:
    """
        Take expectation for objective and gradients (slopes of the cut constraints)
    Args:
        obj: objectives
        grad: gradients (slopes of the cut constraints)
        p: given probabilities

    Returns:
        expected value of objectives and gradients
    """
    if p is not None:
        return numpy.dot(p, obj), numpy.dot(p, grad)
    else:
        return numpy.mean(obj), numpy.mean(grad, axis = 0) # axis = 0 means column computation

def Expectation_AVaR(obj: ArrayLike,
                     grad: ArrayLike,
                     p: None | ArrayLike,
                     a: float,
                     l: float,
                     sense: int)-> Tuple[float, float]:
    """
        Compute the expectation of the risk-averse situation.
    Args:
        obj: objectives
        grad: gradients (slopes of the cut constraints)
        p: given probabilities
        a: the confidence level of AVAR
        l: the weight used in the risk-averse formula
        sense: 1 or -1, whether minimization or maximization

    Returns:
        the objective and gradient considering AVAR risk-averse
    """
    n_samples, n_states = grad.shape
    if p is None:
        p = numpy.ones(n_samples)/n_samples
    objAvg = numpy.dot(p, obj)
    gradAvg = numpy.dot(p, grad) # 1-d array on the left of dot() is viewed as row vector
    # assert(type(gradAvg) == list and len(gradAvg) == len(p))
    objSortedIndex = numpy.argsort(obj)
    if sense == -1:
        objSortedIndex = objSortedIndex[::-1]
    # store the index of 1 - alpha percentile (quantile) #
    tempSum = 0
    for index in objSortedIndex:
        tempSum += p[index]
        if tempSum >= 1 - a: # when this 'a' is smaller, AVAR is larger
            kappa = index
            break

    # kappa = objSortedIndex[int((1 - a) * sampleSize)]
    # the following 2 lines of codes are the original formulas in EJOR(2013)
    # objLP = (1 - l)*objAvg + l(obj_kappa + 1/a*avg((obj - obj[kappa])+))
    # gradLP = (1 - l)*gradAvg + l*(grad_kappa + 1/a*avg((obj - grad[kamma]))

    # the following is the weighted average of means and VaR
    objLP = (1 - l)*objAvg + l*obj[kappa]
    gradLP = (1 - l) * gradAvg + l * grad[kappa]

    # the last term of the original formula
    gradTerm = numpy.zeros((n_samples, n_states))
    objTerm = numpy.zeros(n_samples)
    for j in range(n_samples):
        if sense*(obj[j] - obj[kappa]) >= 0:
            gradTerm[j] = sense * (grad[j] - grad[kappa])
            objTerm[j] = sense * (obj[j] - obj[kappa])

    objLP += sense*l*numpy.dot(p, objTerm) / a
    gradLP += sense*l*numpy.dot(p, gradTerm) / a
    return objLP, gradLP
