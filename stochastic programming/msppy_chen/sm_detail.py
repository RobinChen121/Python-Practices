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
from collections.abc import Callable, Mapping, Sequence, Generator
import numpy
from utils.exception import SampleSizeError, DistributionError
from numbers import Number
import utils.copy as deepcopy
from utils.measure import Expectation
from utils.statistics import rand_int, check_random_state


# noinspection PyUnresolvedReferences,PyRedeclaration
class StochasticModel:
    """
    The detailed programming model for a stage solvable by gurobi.

    Attributes:
        _model: the gurobi model
        type: type of the true problem
        flag_discretized:  whether the true problem has been discretized
        uncertainty_coef_continuous: key is (constraint, var), value is the random generator function
        local_copies: local copies for state variables in the model, eg: for inventory problem, it's I_{t-1}

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
        self.type: str = None  # type of the true problem: continuous/discrete
        self.flag_discretized = 0 #  whether the true problem has been discretized

        self.states = []  # states variables in the model, eg: for inventory problem, it's I_t
        self.local_copies: list = []  # local copies for state variables in the model, eg: for inventory problem, it's I_{t-1}

        self.n_states = 0  # number of state variables in the model
        self.n_samples = 1  # number of discrete uncertainties
        self.probability = None # probabilities for the discrete uncertainties

        # cutting planes approximation of recourse variable alpha
        self.alpha = None # it is a gurobi variable

        self.cuts = []
        self.link_constrs = [] # the linked constraints

        # stage-wise independent discrete uncertainty realizations
        # they are dicts
        self.uncertainty_rhs = {}  # key is the constraint, value is the uncertainty right hand side values
        self.uncertainty_coef = {}  # uncertainty is in the constraint coefficients
                                    # key is (constraint, var), value is the uncertainty values
        self.uncertainty_obj = {}  # uncertainty is in the objective coefficients
                                   # key is the var, value is the uncertainty values

        # stage-wise independent continuous uncertainties
        # they are dicts, their values are the values of the uncertainty
        self.uncertainty_rhs_continuous: dict = {} # key is the constraint, value is the random generator function
        self.uncertainty_coef_continuous: dict = {} # key is (constraint, var), value is the random generator function
        self.uncertainty_obj_continuous: dict = {} # key is the var, value is the random generator function
        self.uncertainty_mix_continuous: dict = {} # seems useless currently

        # indices of stage-dependent uncertainties
        # they are dicts, their values are the indices of the uncertainty
        self.uncertainty_rhs_dependent = {} # key is the constraint, value is the dependent index
        self.uncertainty_coef_dependent = {} # key is (constraint, var), value is the dependent index
        self.uncertainty_obj_dependent = {} # key is the var, value is the dependent index

        # the followings are actually useless in the programming
        # it seems to record the discretization of the continuous uncertainties
        self.uncertainty_rhs_discrete = {}
        self.uncertainty_coef_discrete = {}
        self.uncertainty_obj_discrete = {}

        # collection of all specified dim indices of Markovian uncertainties
        self.Markovian_dim_index = []

        # risk measure, it is a function
        self.measure = Expectation

    def __getattr__(self, name: any) -> any:
        """
        Called when the default attribute access fails with an AttributeError.

        Args:
            name: the attribute

        Returns:
            the attribute of this model.
        """
        try:
            return getattr(self._model, name) # getattr() is gurobi's function to query the value of an attribute
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

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

                if self.type is None:  # if it is None
                    self.type = "discrete"
                    self.n_samples = len(uncertainty)
                else:
                    if self.type != "discrete":  # meaning it is continuous uncertainty
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
                    if self.type is None:
                        self.type = "continuous"
                    else:
                        # already added uncertainty
                        if self.type != "continuous":
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

                    if self.type is None:
                        # add uncertainty for the first time
                        self.type = "discrete"
                        self.n_samples = len(value)
                    else:
                        # already added uncertainty
                        if self.type != "discrete":
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
                    [float(item) for item in sample]
                except (ValueError, TypeError):
                    raise DistributionError(return_data = False)
                if dimension != len(uncertainty(numpy.random)):
                    raise ValueError(
                        "dimension of the distribution is {} while \
                        dimension of the added object is {}!"
                        .format(len(uncertainty(numpy.random)), dimension)
                    )
            if self.type is None:
                # add uncertainty for the first time
                self.type = "continuous"
            else:
                # already added uncertainty
                if self.type != "continuous":
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

    # @property is a built-in decorator used to turn a class method into a property.
    # This allows you to define methods that can be accessed like attributes.
    @property
    def controls(self):
        """
        Get control variables that are not state names or local copy variable names

        """
        model_vars = self._model.getVars() # getVars() is a gurobi function to get a list of all variables in the model
        states_name = [state.varName for state in self.states]
        local_copies_name = [
            local_copy.varName for local_copy in self.local_copies
        ]
        return [
            var
            for var in model_vars
            if var.varName not in states_name + local_copies_name
        ]

    def sample_uncertainty(self, randomState_instance: any = None) -> None:
        """
            update the uncertainty for the true problem.
            this function is actually not used in the quick_start examples.

        Args:
            randomState_instance: an instance of numpy.random.RandomState

        """
        # Sample stage-wise independent true continuous uncertainty
        randomState_instance = check_random_state(randomState_instance) # make sure randomState_instance is a numpy.random.RandomState instance
        if self.uncertainty_coef_continuous is not None:
            for (
                (constr, var),
                dist,
            ) in self.uncertainty_coef_continuous.items():
                self._model.chgCoeff(constr, var, dist(randomState_instance))
        if self.uncertainty_rhs_continuous is not None:
            for constr_tuple, dist in self.uncertainty_rhs_continuous.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr("RHS", list(constr_tuple), dist(randomState_instance))
                else:
                    constr_tuple.setAttr("RHS", dist(randomState_instance))
        if self.uncertainty_obj_continuous is not None:
            for var_tuple, dist in self.uncertainty_obj_continuous.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), dist(randomState_instance))
                else:
                    var_tuple.setAttr("Obj", dist(randomState_instance))
        if self.uncertainty_mix_continuous is not None:
            for keys, dist in self.uncertainty_mix_continuous.items():
                sample = dist(randomState_instance)
                for index, key in enumerate(keys):
                    if type(key) == gurobipy.Var:
                        key.setAttr("Obj", sample[index])
                    elif type(key) == gurobipy.Constr:
                        key.setAttr("RHS", sample[index])
                    else:
                        self._model.chgCoeff(key[0], key[1], sample[index])

    def update_link_constrs(self, fwdSolution: float) -> None:
        """
            updated the linked constraints(an auxiliary variable equals the state variable, i.e., local copy,
             solution of previous stage)
        Args:
            fwdSolution: the value of this stage's state(local copy)
             variable from the solution of previous stage model
        """
        self._model.setAttr("RHS", self.link_constrs, fwdSolutionn)


    def update_uncertainty(self, k):
        """
        Update the corresponding uncertainty realizations in the rhs, obj coef or const coef

        Args:
            k: the k_th realization of the uncertainty
        """
        # Update model with the k^th stage-wise independent discrete uncertainty
        if self.uncertainty_coef is not None:
            for (constr, var), value in self.uncertainty_coef.items():
                self._model.chgCoeff(constr, var, value[k])
        if self.uncertainty_rhs is not None:
            for constr_tuple, value in self.uncertainty_rhs.items():
                if type(constr_tuple) == tuple: # meaning multi constraints
                    self._model.setAttr("RHS", list(constr_tuple), value[k]) # setAttr() function can set the attributes for a list of objects
                else:
                    constr_tuple.setAttr("RHS", value[k])
        if self.uncertainty_obj is not None:
            for var_tuple, value in self.uncertainty_obj.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), value[k])
                else:
                     var_tuple.setAttr("Obj", value[k])
        # self._model.update()

    def update_uncertainty_discrete(self, k):
        """
        This function seems to update the discretization for continuous uncertainty.
        But basically not used in the programming.

        Args:
            k: the k_th realization of the uncertainty
        """
        # update model with the k^th stage-wise independent true discrete uncertainty
        if self.uncertainty_coef_discrete is not None:
            for (constr, var), value in self.uncertainty_coef_discrete.items():
                self._model.chgCoeff(constr, var, value[k])
        if self.uncertainty_rhs_discrete is not None:
            for constr_tuple, value in self.uncertainty_rhs_discrete.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr("RHS", list(constr_tuple), value[k])
                else:
                    constr_tuple.setAttr("RHS", value[k])
        if self.uncertainty_obj_discrete is not None:
            for var_tuple, value in self.uncertainty_obj_discrete.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), value[k])
                else:
                    var_tuple.setAttr("Obj", value[k])

    # noinspection PyArgumentList
    def update_uncertainty_dependent(self, Markov_state: ArrayLike) -> None:
        """
        Update model with the detailed Markov states values

        Args:
            Markov_state: the detailed values of markov states
        """
        if self.uncertainty_coef_dependent is not None:
            for (constr,var), value in self.uncertainty_coef_dependent.items(): # the value is the index
                self._model.chgCoeff(constr, var, Markov_state[value]) # change the coefficient of one var in a constraint
        if self.uncertainty_rhs_dependent is not None:
            for constr_tuple, value in self.uncertainty_rhs_dependent.items():
                # setAttr(): two arguments (i.e., setAttr(attrname, newvalue)) to set a model attribute;
                # three arguments (i.e., setAttr(attrname, objects, newvalues)) to
                # set attribute values for a list or dict of model objects
                # (Var objects, Constr objects, etc.)
                if type(constr_tuple) == tuple:
                    self._model.setAttr( # change the value of one or more attributes
                        "RHS",
                        list(constr_tuple),
                        [Markov_state[i] for i in value]
                    )
                else:
                    constr_tuple.setAttr("RHS", Markov_state[value])
        if self.uncertainty_obj_dependent is not None:
            for var_tuple, value in self.uncertainty_obj_dependent.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr(
                        "Obj",
                        list(var_tuple),
                        [Markov_state[i] for i in value]
                    )
                else:
                    var_tuple.setAttr("Obj", Markov_state[value])

    def _discretize(self, n_samples: int,
                    randomState_instance: numpy.random.RandomState,
                    replace: bool = True) -> None:
        """
        Discretize the stage-wise independent continuous uncertainties.

        chen: This function is actually not used in the stochastic programming.

        Parameters
        ----------
        n_samples: The number of samples to generate uniformly from the distribution

        randomState_instance:  A RandomState instance.

        replace: (optional) Whether the sample is with or without replacement.
        """
        if self.flag_discretized == 1:
            return
        # Discretize the continuous true problem
        if self.type == "continuous":
            self.n_samples = n_samples
            # sort the recorded uncertainty in the dict
            # and recorded them in another dict
            for key, dist in sorted( # sorted the dict, default is sorting by key
                                     # dist it a random generator function
                self.uncertainty_rhs_continuous.items()
            ):
                self.uncertainty_rhs[key] = [
                    dist(randomState_instance) for _ in range(self.n_samples)
                ]
            for key, dist in sorted(self.uncertainty_obj_continuous.items()):
                self.uncertainty_obj[key] = [
                    dist(randomState_instance) for _ in range(self.n_samples)
                ]
            for key, dist in sorted(self.uncertainty_coef_continuous.items()):
                self.uncertainty_coef[key] = [
                    dist(randomState_instance) for _ in range(self.n_samples)
                ]
            for keys, dist in sorted(self.uncertainty_mix_continuous.items()):
                for i in range(self.n_samples):
                    sample = dist(randomState_instance) # dist is a numpy random generator
                    for index, key in enumerate(keys):
                        if type(key) == gurobipy.Var:
                            if key not in self.uncertainty_obj.keys():
                                self.uncertainty_obj[key] = sample # [sample[index]]
                            else:
                                self.uncertainty_obj[key].append(sample) # sample[index]
                        elif type(key) == gurobipy.Constr:
                            if key not in self.uncertainty_rhs.keys():
                                self.uncertainty_rhs[key] = sample
                            else:
                                self.uncertainty_rhs[key].append(sample)
                        else:
                            if key not in self.uncertainty_coef.keys():
                                self.uncertainty_coef[key] = sample
                            else:
                                self.uncertainty_coef[key].append(sample)
        # Discretize discrete true problem
        else:
            if n_samples > self.n_samples:
                raise Exception("n_samples should be smaller than the total number of samples!")
            for key, samples in sorted(self.uncertainty_rhs_discrete.items()):
                # numpy.random.choice does not work on multidimensional arrays
                choiced_indices = rand_int(
                    self.n_samples,
                    randomState_instance,
                    size = n_samples,
                    probability = self.probability,
                    replace = replace,
                )
                self.uncertainty_rhs[key] = [
                    samples[index]
                    for index in choiced_indices
                ]
            for key, samples in sorted(self.uncertainty_obj_discrete.items()):
                choiced_indices = rand_int(
                    self.n_samples,
                    randomState_instance,
                    size = n_samples,
                    probability = self.probability,
                    replace = replace,
                )
                self.uncertainty_obj[key] = [
                    samples[index]
                    for index in choiced_indices
                ]
            for key, samples in sorted(self.uncertainty_coef_discrete.items()):
                choiced_indices = rand_int(
                    self.n_samples,
                    randomState_instance,
                    size = n_samples,
                    probability = self.probability,
                    replace = replace,
                )
                self.uncertainty_coef[key] = [
                    samples[index]
                    for index in choiced_indices
                ]
            self.n_samples_discrete = self.n_samples
            self.n_samples = n_samples
        self.flag_discretized = 1

    def addStateVar(self,
                    lb: float = 0.0,
                    ub: float = float('inf'),
                    obj: float = 0.0,
                    vtype: str = gurobipy.GRB.CONTINUOUS,
                    name: str = '',
                    column: gurobipy.Column = None,
                    uncertainty: Callable | ArrayLike | Mapping = None,
                    uncertainty_dependent: int | ArrayLike | Mapping = None
                    ) -> tuple[gurobipy.Var, gurobipy.Var]:
        """
        Generalize Gurobi's addVar() function.
        Speciallyfor adding the state varaibles in the multi-stage stochastic models.

        Uncertainty using this function is in the objective coefficient of this varaible.

        Args:
            lb (float, optional): Lower bound for the variable. Defaults to 0.0.
            ub (float, optional): Upper bound for the variable. Defaults to float('inf').
            obj (float, optional): Objective coefficient for the variable. Defaults to 0.0.
            vtype (str, optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
                                    or 'C' for continuous, 'B' for binary, 'I' for integer, 'S' for semi-continuous, or 'N' for semi-integer).
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
                                        name = name + '_pre')
        self._model.update()

        self.states += [state]  # append the state to the model
        self.local_copies += [local_copy]
        self.n_states += 1

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
    
    def addStateVars(self,
                     *indices: int,
                     lb: float = 0.0,
                     ub: float = float('inf'),
                     obj: float = 0.0,
                     vtype: str = 'C',
                     name: str = "",
                     uncertainty: Callable | ArrayLike | Mapping = None,
                     uncertainty_dependent: int | ArrayLike | Mapping = None
                     ) -> tuple[gurobipy.tupledict, gurobipy.tupledict]:
        """
        Add multi state variables in the model. Generalize gurobipy.addVars() to
        incorporate uncertainty in the objective function. The corresponding
        local copy variables will also be added in the model.

        Uncertainty using this function is in the coefficient of the vars in the objective function.

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
            state (gurobipy.tupledict): state varaibles.
            local_copy (gurobipy.tupledict): corresponding local copy variables.


        Examples:
        --------
        stage-wise independent discrete uncertain objective coefficients:
        >>> now,past = model.addStateVars(
        ...     2,
        ...     ub = 2.0,
        ...     uncertainty = [[2,4],[3,5]]
        ... )
        >>> now,past = model.addStateVars(
        ...     [(1,2),(2,1)],
        ...     ub = 2.0,
        ...     uncertainty = [[2,4],[3,5]]
        ... )

        stage-wise independent continuous uncertain objective coefficients:
        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0],[0,1]]
        ...     )
        >>> now, past = model.addStateVars(2, ub = 2.0, uncertainty = f)

        Markovian objective coefficients
        >>> now, past = model.addStateVars(2, ub = 2.0, uncertainty_dependent = [1,2])
        """
        state = self._model.addVars(
            *indices, lb = lb, ub = ub, obj = obj, vtype = vtype, name = name
        )
        local_copy = self._model.addVars(
            *indices, lb = lb, ub = ub, name = name + "_pre"
        )
        self._model.update()
        self.states += state.values()
        self.local_copies += local_copy.values()
        self.n_states += len(state)

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, False, len(state))
            if callable(uncertainty):
                self.uncertainty_obj_continuous[tuple(state.values())] = uncertainty
            else:
                self.uncertainty_obj[tuple(state.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, False, len(state))
            self.uncertainty_obj_dependent[tuple(state.values())] = uncertainty_dependent

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
        Add decision vararables to the model. Generalize gurobi's addVar() to incorporate uncertainties.
        Uncertainty using this function is in the coefficient of this var in the objective function.

        Args:
            lb (float, optional): Lower bound for the variable. Defaults to 0.0.
            ub (float, optional): Upper bound for the variable. Defaults to float('inf').
            obj (float, optional): Objective coefficient for the variable. Defaults to 0.0.
            vtype (str, optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
                                 or 'C' for continuous, 'B' for binary, 'I' for integer, 'S' for semi-continuous, or 'N' for semi-integer).
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
                self.uncertainty_obj[var] = uncertainty # gurobi var can be the key in the dict

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(uncertainty_dependent, False, 1)
            self.uncertainty_obj_dependent[var] = uncertainty_dependent

        return var

    def addVars(
            self,
            *indices: int,
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
        Uncertainty of this function is in the coefficient of the vars in the objetive function.

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
                # var  is gurobi tupledict type
                # it has values() attribute like a dict
                self.uncertainty_obj_continuous[tuple(var.values())] = uncertainty
            else:
                self.uncertainty_obj[tuple(var.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, False, len(var))
            self.uncertainty_obj_dependent[tuple(var.values())] = uncertainty_dependent

        return var

    def addConstr(self,
                  constr: any,
                  sense: str = None,
                  rhs: float | gurobipy.Var | gurobipy.LinExpr = None,
                  name: str = "",
                  uncertainty: Callable | ArrayLike | Mapping = None,
                  uncertainty_dependent: int | ArrayLike | Mapping = None,
                 ) -> gurobipy.Constr:
        """
        Add a constraint to the model. Generalize gurobipy.addConstr()
        to incorporate uncertainty in a constraint.

        Uncertainty using this function is in the RHS or coefficients of the constraint.

        uncertainty or uncertainty_dependent are all in dict format.

        Args:
            constr: gurobipy TempConstr argument or can be the left hand side expression of the constraint.
            rhs: Right-hand side for the new constraint. Can be a constant, a Var, or a LinExpr.
            sense: Sense for the new constraint (GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL).
            name: (optional) Name for new constraint.
            uncertainty: (optional) If it is ArrayLike, it is for discrete uncertainty, and it is the scenarios (uncertainty realizations) of stage-wise
                   independent uncertain constraint coefficient and RHS.
                   If it is Mapping, it can be discrete or continuous uncertainty depending on whether the value in the Mapping item can be callable.
                   If it is a Callable function, it is for continuous uncertainty, and it is a multivariate random variable generator of stage-wise
                   independent uncertain constraint coefficient and RHS. It must take numpy RandomState as its only argument.
            uncertainty_dependent: (optional) The location index in the stochastic process generator of stage-wise dependent uncertain constraint coefficient and RHS.
                For Markov uncertainty.

        Returns:
            New constraint object.

        Examples:
        --------
        stage-wise independent finite discrete uncertain rhs/constraint coefficient:

        >>> newConstr = model.addConstr(
        ...     new + past == 3.0,
        ...     uncertainty = {'rhs': [1,2,3], new: [3,4,5]}
        ... )

        The above example dictates scenarios of RHS to be [1,2,3] and
        coefficient of new to be [3,4,5].

        stage-wise independent continuous uncertain rhs/constraint coefficient:

        >>> def f(random_state):
        ...     return random_state.normal(0, 1)
        >>> newConstr = model.addConstr(
        ...     ub = 2.0,
        ...     uncertainty = {new: f},
        ...     uncertainty_dependent = {'rhs': [1]}
        ... )

        The above constraint contains a stage-wise independent uncertain
        constraint coefficient and a Markovian RHS.
        """
        constr = self._model.addLConstr(constr, sense = sense, rhs= rhs, name = name)
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, flag_dict = True, dimension = 1)
            for key, value in uncertainty.items():
                # key can be a gurobipy.Var or "rhs"
                # Append constr to the key
                if type(key) == gurobipy.Var: # meaning uncertainty is in the coefficient of the constraint
                    if callable(value):
                        self.uncertainty_coef_continuous[(constr, key)] = value
                    else:
                        self.uncertainty_coef[(constr, key)] = value
                elif type(key) == str and key.lower() == "rhs": # meaning uncertainty is in the rhs
                    if callable(value):
                        self.uncertainty_rhs_continuous[constr] = value
                    else:
                        self.uncertainty_rhs[constr] = value
                else:
                    raise ValueError("wrong uncertainty key!")

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, flag_dict = True, dimension = 1 )
            for key, value in uncertainty_dependent.items(): # value is the index of the dependent uncertainty
                # key can be a gurobipy.Var or "rhs"
                # Append constr to the key
                if type(key) == gurobipy.Var: # meaning uncertainty is in the coefficient of the constraint
                    if not key in self._model.getVars():
                        raise ValueError("wrong uncertainty key!")
                    self.uncertainty_coef_dependent[(constr, key)] = value
                elif type(key) == str and key.lower() == "rhs": # meaning uncertainty is in the rhs of the constraint
                    self.uncertainty_rhs_dependent[constr] = value
                else:
                    raise ValueError("wrong uncertainty key!")

        return constr

    def addConstrs(self,
                   generator: Generator,
                   name: str = '',
                   uncertainty: Callable | ArrayLike | Mapping = None,
                   uncertainty_dependent: int | ArrayLike | Mapping = None,
                   ) -> gurobipy.tupledict:
        """
        Add multiple constraints to a model using a Python generator expression.

        Generalize gurobipy.addConstrs() to incorporate uncertainty on the RHS of the constraints.

        If you want to add constraints with uncertainties on coefficients,
        use addConstr() instead and add those constraints one by one.

        Args:
            generator: A generator expression, where each iteration produces a constraint.
            name: (optional) Name pattern for new constraints.
                  The given name will be subscribed by the index of the generator expression.
            uncertainty: (optional) If it is ArrayLike, it is for discrete uncertainty, and it is the scenarios (uncertainty realizations) of stage-wise
                   independent uncertain constraint coefficient and RHS.
                   If it is Mapping, it can be discrete or continuous uncertainty depending on whether the value in the Mapping item can be callable.
                   If it is a Callable function, it is for continuous uncertainty, and it is a multivariate random variable generator of stage-wise
                   independent uncertain constraint coefficient and RHS. It must take numpy RandomState as its only argument.
            uncertainty_dependent: (optional) The location index in the stochastic process generator of stage-wise dependent uncertain constraint coefficient and RHS
                For Markov uncertainty.

        Returns:
            A Gurobi tupledict that contains the newly created constraints,
            indexed by the values generated by the generator expression.


        Examples:
        --------
        >>> new = model.addStateVars(2, ub = 2.0)
            past = model.addStateVars(2, ub = 2.0)

        stage-wise independent discrete uncertain RHSs:

        >>> newConstrs = model.addConstrs((new[i] + past[i] == 0 for i in range(2)),
        ...     uncertainty = [[1,2], [2,3]]
        ... )

        The above example dictates scenarios of RHSs to be [1,2] and [2,3]

        stage-wise independent continuous uncertain RHSs:

        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0],[0,100]]
        ...     )
        >>> newConstrs = model.addConstrs(
        ...        (new[i] + past[i] == 0 for i in range(2)),
        ...        uncertainty = f
        ... )

        Markovian uncertain RHSs:

        >>> newConstrs = model.addConstrs(
        ...     (new[i] + past[i] == 0 for i in range(2)),
        ...     uncertainty_dependent = [0,1],
        ... )
        """
        constr = self._model.addConstrs(generator, name = name)
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, flag_dict = False, dimension = len(constr))
            if callable(uncertainty): # values() get all the constraints since constrs is a tupledict
                self.uncertainty_rhs_continuous[tuple(constr.values())] = uncertainty
            else:
                self.uncertainty_rhs[tuple(constr.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(uncertainty_dependent, flag_dict = False,
                                                                      dimension = len(constr))
            self.uncertainty_rhs_dependent[tuple(constr.values())] = uncertainty_dependent

        return constr

    def average(self, objLP_samples: ArrayLike,
                gradLP_samples: ArrayLike,
                probability: list = None) -> tuple[float, float]:
        """
            get the expectation of objectives and gradients
        Args:
            objLP_samples: objectives of all the samples
            gradLP_samples: gradients of all the samples
            probability: given probabilities of all the samples

        """
        p = self.probability if probability is None else probability
        return self.measure(
            obj = objLP_samples,
            grad = gradLP_samples,
            p = p)

    def copy(self):
        """
        Create a deepcopy of a stochastic model.
        The deepcopy() in the copy module is not suitable.

        Returns:
            The copied StochasticModel object.
        """
        cls = self.__class__ # get the class of the current instance
        result = cls.__new__(cls) # create a new instance of the given class, uninitialized
        # copy the internal Gurobi model
        result._model = self._model.copy() # the vars () and constraints in the initial model will be copied
        for attribute, value in self.__dict__.items(): # __dict__ get all the attributes in the function of __init__()
            # for mutable data types like list, dict, set, should use deep copy
            if attribute == "_model":
                pass
            else:
                # firstly copy all attributes and set their values to None
                # setattr is a python built-in function
                setattr(result, attribute, None)
                dict_ = {'target': result, 'attribute': attribute, 'value': value}
                # this is to copy the variables in the targeted model's attributes.
                # it links the attributes to the vars and constraints in the _model,
                # otherwise the copied attributes will have different vars and
                # constraints with those in the _model although they have same names.
                # copy all uncertainties
                if attribute.startswith("uncertainty"):
                    setattr(result, attribute, {})
                    if attribute.startswith("uncertainty_rhs"):
                        deepcopy.copy_uncertainty_rhs(**dict_) # ** is for dictionary unpacking
                    elif attribute.startswith("uncertainty_coef"):
                        deepcopy.copy_uncertainty_coef(**dict_)
                    elif attribute.startswith("uncertainty_obj"):
                        deepcopy.copy_uncertainty_obj(**dict_)
                    elif attribute.startswith("uncertainty_mix"):
                        deepcopy.copy_uncertainty_mix(**dict_)
                    else:
                        raise Exception("alien uncertainties added!")
                # copy all other variables (some variables already copied by the _model.copy()).
                elif attribute in ["states", "local_copies", "alpha"]:
                    deepcopy.copy_vars(**dict_)
                # copy all constraints
                elif attribute in ["cuts", "link_constrs"]:
                    deepcopy.copy_constrs(**dict_)
                # copy probability measure
                elif attribute == "probability":
                    result.probability = None if value is None else list(value)
                else:
                    setattr(result, attribute, value)

        return result

    def set_up_link_constrs(self):
        if not self.link_constrs:
            self.link_constrs = list(
                self._model.addConstrs(
                    (var == var.lb for var in self.local_copies),
                    name = "link_constrs",
                ).values()
            )

    def delete_link_constrs(self):
        if self.link_constrs:
            for constr in self.link_constrs:
                self._model.remove(constr)
            self.link_constrs = []

    def set_up_CTG(self, discount: float, bound: float):
        """
            add a new variable alpha in the model
        Args:
            discount: discount factor in the multi-stage model
            bound: bound for the alpha value
        """
        # if it's a minimization problem, we need a lower bound for alpha
        # For CTG, it is an alpha added in the constraints:
        #         alpha + ax + by >= c in minimization problem
        #         alpha + ax + by <= c in maximization problem
        if self.modelsense == 1:
            if self.alpha is None:
                self.alpha = self._model.addVar(
                    lb = bound,
                    ub = gurobipy.GRB.INFINITY,
                    obj = discount,
                    name = "alpha")

        # if it's a maximization problem, we need an upper bound for alpha
        else:
            if self.alpha is None:
                self.alpha = self._model.addVar(
                    ub = bound,
                    lb = -gurobipy.GRB.INFINITY,
                    obj = discount,
                    name = "alpha"
                )

    def delete_CTG(self):
        if self.alpha is not None:
            self._model.remove(self.alpha)
            self.alpha = None

    def set_probability(self,
                        probability: ArrayLike) -> None:
        """
        Set probability measure of discrete scenarios.

        Args:
            probability: Array-like probability of scenarios. Default is uniform measure
                         [1/n_samples for _ in range(n_samples)].

                         Length of the list must equal to the length of uncertainty.
                         The order of the list must match with the order of uncertainty list.

        Returns:
            None.

        Examples:
        --------
        >>> newVar = model.addVar(ub = 2.0, uncertainty = [1, 2, 3])
        >>> model.setProbability([0.2, 0.3, 0.4])
        """
        self.probability = list(probability)
        if len(probability) != self.n_samples:
            raise ValueError("probability tree != compatible with scenario tree")

    def solveLP(self) -> tuple[float, float]:
        """
            solve the backward model in one stage
        Returns:
            the objective and dual values of the linked constraints
        """
        objLP_sample = numpy.empty(self.n_samples)
        gradLP_sample = numpy.empty((self.n_samples, self.n_states))
        for k in range(self.n_samples):
            self._update_uncertainty(k)
            self.optimize()
            if self._model.status not in [2, 11]:
                self.write_infeasible_model("backward_failedSolved" + str(self._model.modelName))
            objLP_sample[k] = self.objVal
            gradLP_sample[k] = self.getAttr("Pi", self.link_constrs)
        return objLP_sample, gradLP_sample

    def update(self) -> None:
        """
        Process any pending model modifications.
        """
        self._model.update()

    def write_infeasible_model(self, text):
        self._model.write('./' + text + ".lp")
        raise Exception(
            "infeasibility caught; check complete recourse condition!"
        )