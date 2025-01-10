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
            A class of multi stage linear programming model;

        """
        self.T = T
        self.bound = bound
        self.sense = sense

        self.measure = 'risk neutral'
        self.type = 'stage-wise independent'

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

        