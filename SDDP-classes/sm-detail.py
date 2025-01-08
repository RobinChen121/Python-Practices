#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:48:12 2025

@author: zhenchen

@Python version: 3.10

@disp:  detailed stochastic model solvable by gurobi;
    
    
"""

from gurobipy import *
from numpy.typing import ArrayLike
from collections.abc import Callable


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
        self.states = [] # states varaibles in the model
        self.local_copies = [] # local copies for state varaibles in the model
        self.num_states = 0 # number of state variables in the model
        self.num_samples = 0 # number of discrete uncertainties
        self._type = None # type of the true problem: continuous/discrete
        
    def _check_uncertainty(self):
        if isinstance(uncertainty, abc.Mapping):
            pass
    
    def addStateVar(self,
                    lb: float = 0.0,
                    ub: float = float('inf'), 
                    obj: float = 0.0,
                    vtype: str = GRB.CONTINUOUS, 
                    name: str = '',
                    column: gurobipy.Columm = None,
                    uncertainty: ArrayLike | Callable = None,
                    uncertainty_dependent: ArrayLike = None
                    ):
        """
        Generalize Gurobi's addVar() function.
        Speciallyfor adding the state varaibles in the multi-stage stochastic models.

        Args:
            lb (float, optional): Lower bound for the variable. Defaults to 0.0.
            ub (float, optional): Upper bound for the variable. Defaults to float('inf').
            obj (float, optional): Objective coefficient for the variable. Defaults to 0.0.
            vtype (str, ptional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
                                    or 'C' for continuous, 'B' for binary, 'I' for integer, 'S' for semi-continuous, or 'N' for semi-integer)).
                                    Defaults to GRB.CONTINUOUS.
            name (str, optional): Name for the variable. Defaults to ''.
            column (gurobi.Column, optional): gurobi Column object that indicates the set of constraints in which the new variable participates, and the associated coefficients. 
                                     Defaults to None.
            uncertainty (ArrayLike | Callable, optional): Default to None.
                If it is ArrayLike, it is for discrete uncertainty and it is the scenarios (uncertainty realizatoins) of stage-wise independent uncertain objective
                coefficients.
                If it is a Callable function, it is for continous uncertainty and it is a multivariate random variable generator of stage-wise
                independent uncertain objective coefficients. It must take numpy RandomState as its only argument.
            uncertainty_dependent (ArrayLike): Default to None.
                The location index in the stochastic process generator of stage-wise dependent uncertain objective coefficients.
                For Markov uncertainty.
                
            Examples
            --------
            >>> now,past = model.addStateVar(ub=2.0, uncertainty=[1,2,3])

            >>> def f(random_state):
            ...     return random_state.normal(0, 1)
            >>> now,past = model.addStateVar(ub=2.0, uncertainty=f)

            Markovian objective coefficient:
            >>> now,past = model.addStateVar(ub=2.0, uncertainty_dependent=[1,2])
            
        Returns:
            the created stata varaible and the corresponding local copy variable.

        """
        state = self._model.addVar(lb = lb,
                                   ub = ub,
                                   obj = obj,
                                   vtype = vtype,
                                   name = name,
                                   column = column)
        local_copy = self._model.addVar(lb = lb,
                                        ub = ub,
                                        name = name + '_local_copy')
        self._model.update()
        
        self.states += [state] # append the state to the model
        self.local_copies += [local_copy] 
        self.num_states += 1
        
        return state, local_copy
        
        
        
