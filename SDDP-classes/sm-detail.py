#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:48:12 2025

@author: zhenchen

@Python version: 3.10

@disp:  detailed stochastic model at a stage solvable by gurobi;
    
    
"""

from gurobipy import *
from numpy.typing import ArrayLike
from collections.abc import Callable, Mapping, Sequence
import numpy as np
from exception import SampleSizeError


class StochasticModel():
    """
    the detailed programming model for a stage solvable by gurobi;
    
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
        self.num_samples = 1 # number of discrete uncertainties
        self._type = None # type of the true problem: continuous/discrete       
        self.probability = None # probabilities for the discrete uncertainties
        
        # (discretized) uncertainties
        # stage-wise independent discrete uncertainties
        self.uncertainty_rhs = {} # {} is dic format, uncertainty is in the right hand side of the constraints
        self.uncertainty_coef = {} # uncertainty is in the constraint coefficients
        self.uncertainty_obj = {} # uncertainty is in the objecgtive coefficients
        
        # true uncertainties
        # stage-wise independent true continuous uncertainties
        self.uncertainty_rhs_continuous = {}
        self.uncertainty_coef_continuous = {}
        self.uncertainty_obj_continuous = {}
        self.uncertainty_mix_continuous = {}
        # stage-wise independent true discrete uncertainties
        self.uncertainty_rhs_discrete = {}
        self.uncertainty_coef_discrete = {}
        self.uncertainty_obj_discrete = {}
        
        # indices of stage-dependent uncertainties
        self.uncertainty_rhs_dependent = {}
        self.uncertainty_coef_dependent = {}
        self.uncertainty_obj_dependent = {}
        
       
    def _check_uncertainty(self, uncertainty: ArrayLike | Callable, flag_dict: bool, dimension: int):
        """
        Check whether the input unertainty is in correct format. 

        Args:
            uncertainty (ArrayLike | Callable): The uncertainty.
            flag_dict (bool): Whether the uncertainty is in a dictionary data structure.
            dimension (int): The dimension of the uncertainty.
        
        ------
        for discrete uncertainty:
            
        Uncertainty added by addVar must be a array-like (flag_dict = 0, dimension = 1).
        
        
        for continuous uncertainty:
            
        Uncertainty added by addVar must be a callable that generates a single
        number (flag_dict = 0, dimensino = 1).

        Returns:
            Return the uncertainty in correct format.

        """       
        if isinstance(uncertainty, (Sequence, np.ndarray)): # whether it is a list or sequence, for discrete uncertainty          
            if dimension == 1:
                if uncertainty.ndim != 1:
                    raise ValueError("dimension of the scenarios is {} while  dimension of the added object is 1!"
                        .format(uncertainty.ndim))
                try:
                    uncertainty = [float(item) for item in uncertainty]
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
                uncertainty = list(uncertainty)
            else: # dimension more than 1
                if uncertainty.shape[1] != dimension:
                    raise ValueError("dimension of the scenarios should be {} while \
                                     dimension of the added object is {}!"
                        .format(dimension, uncertainty.ndim))
                try:
                    uncertainty = np.array(uncertainty, dtype='float64')
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
                uncertainty = [list(item) for item in uncertainty]
                
                if self._type is None: # if it is None
                    self._type = "discrete"
                    self.n_samples = len(uncertainty)
                else:
                    if self._type != "discrete": # meaning it is continous uncertainty
                        raise SampleSizeError(
                            self._model.modelName,
                            "infinite",
                            uncertainty,
                            len(uncertainty)
                        )
                    if self.n_samples != len(uncertainty):
                        raise SampleSizeError(
                            self._model.modelName,
                            self.n_samples,
                            uncertainty,
                            len(uncertainty)
                        )
            
        elif isinstance(uncertainty, Mapping): # whether it is an instance of dict type or similar type
            uncertainty = dict(uncertainty)
            
        return uncertainty
        
        
    def _check_uncertainty_dependent(self):
        """
        

        Returns:
            None.

        """
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
        
        if uncertainty is not None:
            uncertainty  = self._check_uncertainty(uncertainty, 0, 1) 
            # uncertainty in the state is reflected in the objective coefficient of this state variable
            if callable(uncertainty): # if uncertainty is generated by a callable function
                self.uncertainty_obj_continuous[state] = uncertainty # add this uncertainty to the dicts
            else:
                self.uncertainty_obj[state] = uncertainty # add this uncertainty to the dicts
        
        if uncertainty_dependent is not None: # for Markov uncertainty in the objective coefficients
            uncertainty_dependent = self._check_uncertainty_dependent(uncertainty_dependent, 0, 1)
            self.uncertainty_obj_dependent[state] = uncertainty_dependent
                    
        return state, local_copy
        
        
        
