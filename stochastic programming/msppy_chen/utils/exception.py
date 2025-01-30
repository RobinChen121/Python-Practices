#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:17:03 2025

@author: zhen chen

@Python version: 3.10

@disp:  self defined exception classes.
    
    
"""

class SampleSizeError(Exception):
    """
    Exception class to raise if uncertainty of different sample sizes are
    added to the model.
    """
    def __init__(self, modelName, dimensionality, uncertainty, dimension):
        super().__init__("Dimensionality of stochasticModel {} is {} but dimension of the uncertainty {} is {}".format(
                modelName, dimensionality, uncertainty, dimension))
        
class DistributionError(Exception):
    """
    Exception class to raise if continuous distribution is not added
    properly
    """
    def __init__(self, arg: bool = True, return_data: bool= True):
        if not arg:
            Exception.__init__(
                self,
                "Continuous distribution should always take \
                numpy.random.RandomState as its single argument.",
            )
        if not return_data:
            Exception.__init__(
                self,
                "Uni-variate distribution should always return a number; \
                Multivariate distribution should always return an array-like.",
            )

class MarkovianDimensionError(Exception):
    """
    Exception class to raise if dim index is not specified
    properly

    """
    def __init__(self):
        Exception.__init__(
            self,
            "Dimension indices of Markovian uncertainties not set properly."
        )