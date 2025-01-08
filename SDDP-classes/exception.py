#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:17:03 2025

@author: zhenchen

@Python version: 3.10

@disp:  self defined exception classes.
    
    
"""

class SampleSizeError(Exception):
    """Exception class to raise if uncertainty of different sample sizes are
    added to the model."""
    def __init__(self, modelName, dimensionality, uncertainty, dimension):
        super().__init__("Dimensionality of stochasticModel {} is {} but dimension of the uncertainty {} is {}".format(
                modelName, dimensionality, uncertainty, dimension))
                modelName, dimensionality, uncertainty, dimensio),
        
