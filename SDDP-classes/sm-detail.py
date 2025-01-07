#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:48:12 2025

@author: zhenchen

@Python version: 3.10

@disp:  detailed stochastic model solvable by gurobi;
    
    
"""

import gurobipy


class StochasticModel():
    """
    the detailed programming model solvable by gurobi;
    
    """
    
    def __init__(self, name: str = '', env: object = None):
        """
        


        Args:
            name (str, optional): Name of new model. Defaults to ''.
            env (object, optional): Environment in which to create the model. 
                                    Creating your own environment (using the Env constructor) gives you more control
                                    (for example, to solve your model on a specific Compute Server). 
                                    It can make your program more verbose, though, 
                                    so we suggest that you use the default environment unless you know that
                                    you need to control your own environments. Defaults to None.

        Returns:
            Initialize a gurobi model.

        """
        self._model = gurobipy.Model(name = name, env = env)
