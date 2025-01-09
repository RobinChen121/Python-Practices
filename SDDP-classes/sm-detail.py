#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:48:12 2025

@author: zhen chen

@Python version: 3.10

@disp:  detailed stochastic model at a stage solvable by gurobi;
    
    
"""

import gurobipy
from numpy.typing import ArrayLike
from collections.abc import Callable, Mapping, Sequence
import numpy
from exception import SampleSizeError, DistributionError
from numbers import Number


# noinspection PyUnresolvedReferences
class StochasticModel:
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
        self._model = gurobipy.Model(name=name, env=env)
        self.states = []  # states variables in the model
        self.local_copies = []  # local copies for state variables in the model

        self.num_states = 0  # number of state variables in the model
        self.num_samples = 1  # number of discrete uncertainties
        self._type = None  # type of the true problem: continuous/discrete
        self.probability = None  # probabilities for the discrete uncertainties

        # (discrete) uncertainties
        # stage-wise independent discrete uncertainties
        self.uncertainty_rhs = {}  # {} is dic format, uncertainty is in the right hand side of the constraints
        self.uncertainty_coef = {}  # uncertainty is in the constraint coefficients
        self.uncertainty_obj = {}  # uncertainty is in the objective coefficients

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

        # collection of all specified dim indices of Markovian uncertainties
        self.Markovian_dim_index = []

    def _check_uncertainty(self,
                           uncertainty: ArrayLike | Mapping | Callable,
                           flag_dict: bool,
                           dimension: int
                           ) -> ArrayLike | Mapping | Callable:

        """
        Check whether the input uncertainty is in correct format.

        flag_dict = 0 only happens for addConstr() and add_continuous_uncertainty()

        Args:
            uncertainty (ArrayLike | Mapping | Callable): The uncertainty.
            flag_dict (bool): Whether the uncertainty is in a dictionary data structure.
            dimension (int): The dimension of the uncertainty.
        
        ------
        for discrete uncertainty: (array-like or array-like in the dict)         
        Uncertainty added by addVar must be an array-like (flag_dict = 0, dimension = 1);
        
        Uncertainty added by addConstrs and addVars must be a multidimensional
        array-like (flag_dict = 0, dimension > 1),
        The multidimensional array-like has the shape (a,b), where a should be
        the dimension of the object added indicated by dimension (>1) and b
        should be the number of samples;
        
        Uncertainty added by addConstr must be a dictionary. Value of the
        dictionary must be a callable that generates a single number
        (flag_dict=1, list_dim=1).

        for continuous uncertainty: (callable or callable in the dict)           
        Uncertainty added by addVar must be a callable that generates a single
        number (flag_dict = 0, dimension = 1).
        
        Uncertainty added by addConstr must be a dictionary. Value of the
        dictionary must be a callable that generates a single number
        (flag_dict = 1, dimension = 1).

        Uncertainty added by addConstrs and addVars must be a callable that
        generates an array-like (flag_dict = 0, dimension > 1)
        The generated array-like has the shape (a,b), where a should the
        dimension of the object added indicated by list_dim (>1) and b should be
        the number of samples.
        
        -------------------------
        All callable should take numpy RandomState as its only argument;
        The true problem must be either continuous or discrete. Hence, once a
        continuous uncertainty has been added, discrete uncertainty is no longer
        accepted, vice versa.

        Returns:
            Return the uncertainty in correct format.

        """
        if isinstance(uncertainty, (Sequence, numpy.ndarray)):  # whether it is array-like, for discrete uncertainty
                                                                # list is a sequence, but is not a numpy ndarray
            uncertainty = numpy.array(uncertainty) # ArrayLike objects can be converted to numpy arrays
            if dimension == 1:
                if uncertainty.ndim != 1:
                    raise ValueError("dimension of the scenarios is {} while  dimension of the added object is 1!"
                                     .format(uncertainty.ndim))
                try:
                    uncertainty = [float(item) for item in uncertainty]
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
                uncertainty = list(uncertainty)
            else:  # dimension more than 1
                if uncertainty.shape[1] != dimension:
                    raise ValueError("dimension of the scenarios should be {} while \
                                     dimension of the added object is {}!"
                                     .format(dimension, uncertainty.shape[1]))
                try:
                    uncertainty = numpy.array(uncertainty, dtype = 'float64')
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
                uncertainty = [list(item) for item in uncertainty]

                if self._type is None:  # if it is None
                    self._type = "discrete"
                    self.n_samples = len(uncertainty)
                else:
                    if self._type != "discrete":  # meaning it is continuous uncertainty
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
        elif isinstance(uncertainty, Mapping):  # whether it is an instance of dict type or similar dict type
            if not flag_dict:
                raise TypeError("wrong uncertainty format!")
            uncertainty = dict(uncertainty)
            for key, value in uncertainty.items():
                if callable(value):  # dict with callable function
                    if self._type is None:
                        self._type = "continuous"
                    else:
                        # already added uncertainty
                        if self._type != "continuous":
                            raise SampleSizeError(
                                self._model.modelName,
                                self.n_samples,
                                uncertainty,
                                "infinite"
                            )
                    try:
                        value(numpy.random)
                    except TypeError:
                        raise DistributionError(arg = False)
                    try:
                        numpy.array(value(numpy.random), dtype = 'float64')
                    except (ValueError, TypeError):
                        raise DistributionError(return_data = False)
                else:  # dict but not callable
                    try:
                        value = numpy.array(value, dtype = 'float64')
                    except ValueError:
                        raise ValueError("Scenarios must only contains numbers!")
                    if value.ndim != 1: # if it is a dict format but not callable, it should be 1 dimension.
                        raise ValueError(
                            "dimension of the distribution is {} while \
                            dimension of the added object should be {}!"
                            .format(value.ndim, 1)
                        )
                    uncertainty[key] = list(value)

                    if self._type is None:
                        # add uncertainty for the first time
                        self._type = "discrete"
                        self.n_samples = len(value)
                    else:
                        # already added uncertainty
                        if self._type != "discrete":
                            raise SampleSizeError(
                                self._model.modelName,
                                "infinite",
                                {key: value},
                                len(value)
                            )
                        if self.n_samples != len(value):
                            raise SampleSizeError(
                                self._model.modelName,
                                self.n_samples,
                                {key: value},
                                len(value)
                            )
        elif isinstance(uncertainty, Callable):  # whether it is a callable function
            try:
                sample = uncertainty(numpy.random)
            except TypeError:
                raise DistributionError(arg = False)
            if dimension == 1:
                try:
                    float(sample)
                except (ValueError, TypeError):
                    raise DistributionError(return_data = False)
            else:
                try:
                    sample = [float(item) for item in sample]
                except (ValueError, TypeError):
                    raise DistributionError(return_data = False)
                if dimension != len(uncertainty(numpy.random)):
                    raise ValueError(
                        "dimension of the distribution is {} while \
                        dimension of the added object is {}!"
                        .format(len(uncertainty(numpy.random)), dimension)
                    )
            if self._type is None:
                # add uncertainty for the first time
                self._type = "continuous"
            else:
                # already added uncertainty
                if self._type != "continuous":
                    raise SampleSizeError(
                        self._model.modelName,
                        self.n_samples,
                        uncertainty,
                        "infinite"
                    )
        else:
            raise TypeError("wrong uncertainty format!")
        return uncertainty

    def _check_uncertainty_dependent(self,
                                     uncertainty_dependent: ArrayLike | Mapping | int,
                                     flag_dict: bool,
                                     dimension: int
                                     ) -> ArrayLike | Mapping | int:


        """
        Make sure the input uncertainty location index is in the correct form.

        flag_dict = True only happens for addConstr() and add_continuous_uncertainty()

        Args:
            uncertainty_dependent (ArrayLike | Mapping | int): The dependent uncertainty.
            flag_dict (bool): Whether the dependent uncertainty is in a dictionary data structure.
            dimension (int): The dimension of the dependent uncertainty.
        
        Returns:
            A copied uncertainty to avoid making changes to mutable object
            given by the users.

        
        Check data structure
        --------------------

        Uncertainty_dependent added by addConstr must be a dictionary. Value of the
        dictionary must be an int (flag_dict = 1, dimension = 1).

        Uncertainty_dependent added by addVar must be an int (flag_dict = 0, dimension = 1).

        Uncertainty_dependent added by addConstrs and addVars must be an array-like of int
        (flag_dict = 0, dimension > 1). The length of the array-like
        should equal dimension.
        
        """
        if isinstance(uncertainty_dependent, Mapping):  # if it is dict type or similar dict type
            if flag_dict == 0:
                raise TypeError("wrong uncertainty_dependent format!")
            for key, value in uncertainty_dependent.items():
                try:
                    value = int(value)
                    uncertainty_dependent[key] = value
                except (TypeError, ValueError):
                    raise ValueError("location index of individual component \
                                     of uncertainty_dependent must be integer!")
                self.Markovian_dim_index.append(value)

        elif isinstance(uncertainty_dependent, (Sequence, numpy.ndarray)):  # if it is array-like
            uncertainty_dependent = list(uncertainty_dependent)
            if len(uncertainty_dependent) != dimension:
                raise ValueError(
                    "dimension of the scenario is {} while \
                    dimension of added object is {}!"
                    .format(len(uncertainty_dependent), dimension)
                )
            self.Markovian_dim_index += uncertainty_dependent  # extend the list of dim index

        elif isinstance(uncertainty_dependent, Number):  # if it is a number
            uncertainty_dependent = int(uncertainty_dependent)
            if dimension != 1:
                raise ValueError(
                    "dimension of the scenario is 1 while \
                    dimension of added object is {}!"
                    .format(dimension)
                )
            self.Markovian_dim_index.append(uncertainty_dependent)
        else:
            raise TypeError("wrong uncertainty_dependent format")
        return uncertainty_dependent

    # noinspection PyArgumentList
    def addStateVar(self,
                    lb: float = 0.0,
                    ub: float = float('inf'),
                    obj: float = 0.0,
                    vtype: str = gurobipy.GRB.CONTINUOUS,
                    name: str = '',
                    column: gurobipy.Columm = None,
                    uncertainty: Callable | ArrayLike | Mapping = None,
                    uncertainty_dependent: int | ArrayLike | Mapping = None
                    ) -> tuple[gurobipy.Var, gurobipy.Var]:
        """
        Generalize Gurobi's addVar() function.
        Speciallyfor adding the state varaibles in the multi-stage stochastic models.
        if having uncertainty, uncertainty happens in the objective coefficient of this varaible.

        Args:
            lb (float, optional): Lower bound for the variable. Defaults to 0.0.
            ub (float, optional): Upper bound for the variable. Defaults to float('inf').
            obj (float, optional): Objective coefficient for the variable. Defaults to 0.0.
            vtype (str, optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
                                    or 'C' for continuous, 'B' for binary, 'I' for integer, 'S' for semi-continuous, or 'N' for semi-integer)).
                                    Defaults to GRB.CONTINUOUS.
            name (str, optional): Name for the variable. Defaults to ''.
            column (gurobi.Column, optional): gurobi Column object that indicates the set of constraints in which the new variable participates, and the associated coefficients. 
                                     Defaults to None.
            uncertainty (ArrayLike | Callable | Mapping, optional): Default to None.
                If it is ArrayLike, it is for discrete uncertainty, and it is the scenarios (uncertainty realizations) of stage-wise independent uncertain objective
                coefficients.
                If it is Mapping, it can be discrete or continuous uncertainty depending on whether the value in the Mapping item can be callable.
                If it is a Callable function, it is for continous uncertainty, and it is a random variable generator of stage-wise
                independent uncertain objective coefficients. It must take numpy RandomState as its only argument.
            uncertainty_dependent (int | ArrayLike | mapping): Default to None.
                The location index in the stochastic process generator of stage-wise dependent uncertain objective coefficients.
                For Markov uncertainty.
            
        Returns:
            the created stata varaible and the corresponding local copy variable.

        Examples:
        --------
            stage-wise independent discrete uncertain objective coefficient:
            >>> now, past = model.addStateVar(ub = 2.0, uncertainty = [1, 2, 3])

            stage-wise independent continuous uncertain objective coefficient
            >>> def f(random_state):
            ...     return random_state.normal(0, 1)
            >>> now, past = model.addStateVar(ub = 2.0, uncertainty = f)

            Markovian objective coefficient:
            >>> now, past = model.addStateVar(ub = 2.0, uncertainty_dependent = [1, 2])

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

        self.states += [state]  # append the state to the model
        self.local_copies += [local_copy]
        self.num_states += 1

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, False, 1)
            # uncertainty in the state is reflected in the objective coefficient of this state variable
            if callable(uncertainty):  # if uncertainty is generated by a callable function
                self.uncertainty_obj_continuous[state] = uncertainty  # add this uncertainty to the dicts
            else:
                self.uncertainty_obj[state] = uncertainty  # add this uncertainty to the dicts
        if uncertainty_dependent is not None:  # for Markov uncertainty in the objective coefficients
            uncertainty_dependent = self._check_uncertainty_dependent(uncertainty_dependent, False, 1)
            self.uncertainty_obj_dependent[state] = uncertainty_dependent

        return state, local_copy

    def addVar(self,
                lb: float = 0.0,
                ub: float = float('inf'),
                obj: float = 0.0,
                vtype: str = 'C',
                name: str = "",
                column: gurobipy.Column = None,
                uncertainty: Callable | ArrayLike | Mapping = None,
                uncertainty_dependent: int | ArrayLike | Mapping = None
               ) -> gurobipy.Var:
        """

        Args:
            lb (float, optional): Lower bound for the variable. Defaults to 0.0.
            ub (float, optional): Upper bound for the variable. Defaults to float('inf').
            obj (float, optional): Objective coefficient for the variable. Defaults to 0.0.
            vtype (str, optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
                                 or 'C' for continuous, 'B' for binary, 'I' for integer, 'S' for semi-continuous, or 'N' for semi-integer)).
                                 Defaults to GRB.CONTINUOUS.
            name (str, optional): Name for the variable. Defaults to ''.
            column (gurobi.Column, optional): gurobi Column object that indicates the set of constraints in which the new variable participates, and the associated coefficients.
                                 Defaults to None.
            uncertainty (ArrayLike | Callable | Mapping, optional): Default to None.
                   If it is ArrayLike, it is for discrete uncertainty, and it is the scenarios (uncertainty realizations) of stage-wise independent uncertain objective
                   coefficients.
                   If it is Mapping, it can be discrete or continuous uncertainty depending on whether the value in the Mapping item can be callable.
                   If it is a Callable function, it is for continous uncertainty, and it is a random variable generator of stage-wise
                   independent uncertain objective coefficients. It must take numpy RandomState as its only argument.
            uncertainty_dependent (int | ArrayLike | Mapping): Default to None.
                The location index in the stochastic process generator of stage-wise dependent uncertain objective coefficients.
                For Markov uncertainty.

        Returns:
            A gurobi Var object.

        Examples
        --------
        stage-wise independent discrete uncertain objective coefficient:
        >>> newVar = model.addVar(ub = 2.0, uncertainty = [1, 2, 3])

        stage-wise independent continuous uncertain objective coefficient
        >>> def f(random_state):
        ...     return random_state.normal(0, 1)
        ... newVar = model.addVar(ub = 2.0, uncertainty = f)

        Markovian objective coefficient
        >>> newVar = model.addVar(ub = 2.0, uncertainty_dependent = [1])
        """
        var = self._model.addVar(
            lb = lb, ub = ub, obj = obj, vtype = vtype, name = name, column = column
        )
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, False, 1)
            if callable(uncertainty):
                self.uncertainty_obj_continuous[var] = uncertainty
            else:
                self.uncertainty_obj[var] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(uncertainty_dependent, False, 1)
            self.uncertainty_obj_dependent[var] = uncertainty_dependent

        return var

    def addVars(
            self,
            *indices: list,
            lb: float = 0.0,
            ub: float = float('inf'),
            obj: float = 0.0,
            vtype: str = 'C',
            name: str = "",
            uncertainty:  Callable | ArrayLike | Mapping = None,
            uncertainty_dependent: int | ArrayLike | Mapping = None
            ) -> gurobipy.tupledict:
        """
        Generalize gurobipy.addVars() to incorporate uncertainty in the objective function.

        Args:
            *indices: Indices for accessing the new variables.
            lb: (optional) Lower bound(s) for new variables.
            ub: (optional) Upper bound(s) for new variables.
            obj: (optional) Objective coefficient(s) for new variables.
            vtype:  (optional) Variable type(s) for new variables.
            name:  (optional) Names for new variables. The given name will be subscribed by the index of the generator expression.
            uncertainty: (optional) If it is ArrayLike, it is for discrete uncertainty, and it is the scenarios (uncertainty realizations) of stage-wise independent uncertain objective
                   coefficients.
                   If it is Mapping, it can be discrete or continuous uncertainty depending on whether the value in the Mapping item can be callable.
                   If it is a Callable function, it is for continous uncertainty, and it is a multivariate random variable generator of stage-wise
                   independent uncertain objective coefficients. It must take numpy RandomState as its only argument.
            uncertainty_dependent: (optional) The location index in the stochastic process generator of stage-wise dependent uncertain objective coefficients.
                For Markov uncertainty.

        Returns:
            gurobi New tupledict object that contains the new variables as values, using the provided indices as keys.

        Examples:
        --------
        stage-wise independent discrete uncertain objective coefficients:
        >>> newVars = model.addVars(3, ub = 2.0, uncertainty = [[2,4,6], [3,5,7]]) # 3 variables, each column in the uncertainty are all the realizations of one variable

        >>> newVars = model.addVars(
        ...     [(1,2),(2,1)],
        ...     ub=2.0,
        ...     uncertainty=[[2,4], [3,5], [4,6]]
        ... ) # create 2 variables x[1, 2] and x[2, 1]

        stage-wise independent continuous uncertain objective coefficients:
        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0], [0,100]]
        ...     )
        >>> newVars = model.addVars(
        ...     2,
        ...     ub = 2.0,
        ...     uncertainty = f
        ... )

        Markovian objective coefficients:
        >>> newVars = model.addVars(
        ...     2,
        ...     ub = 2.0,
        ...     uncertainty_dependent = [1,2]
        ... )
        """
        var = self._model.addVars(
            *indices, lb = lb, ub = ub, obj = obj, vtype = vtype, name = name
        )
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, False, len(var))
            if callable(uncertainty):
                self.uncertainty_obj_continuous[
                    tuple(var.values())
                ] = uncertainty
            else:
                self.uncertainty_obj[tuple(var.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, False, len(var)
            )
            self.uncertainty_obj_dependent[
                tuple(var.values())
            ] = uncertainty_dependent

        return var
