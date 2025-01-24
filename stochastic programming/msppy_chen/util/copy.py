"""
Created on 2025/1/23, 0:20 

@author: Zhen Chen.

@Python version: 3.10

@disp:  deep copy for some mutable data structure

"""
from numpy.typing import ArrayLike
import gurobipy


def copy_uncertainty_rhs(target, attribute: str, value: ArrayLike) -> None:
    """
    Copy rhs uncertainty (attribute, value) to target

    Args:
        target: the targeted Stochastic Model
        attribute: the name of the attribute
        value: the attribute values

    """
    result = {}
    for constr_tuple, value in value.items():
        if type(constr_tuple) == tuple:
            s = [
                target._model.getConstrByName(x.constrName)
                for x in constr_tuple
            ]
            result[tuple(s)] = value
        else:
            s = target._model.getConstrByName(constr_tuple.constrName)
            result[s] = value
    setattr(target, attribute, result)

def _copy_uncertainty_coef( target, attribute, value):
    """Copy coef uncertainty (attribute, value) to target"""
    result = {}
    for key, value in value.items():
        constr = target._model.getConstrByName(key[0].constrName)
        var = target._model.getVarByName(key[1].varName)
        result[(constr, var)] = value
    setattr(target, attribute, result)

def _copy_uncertainty_obj(target, attribute, value):
    """Copy obj uncertainty (attribute, value) to target"""
    result = {}
    for var_tuple, value in value.items():
        if type(var_tuple) == tuple:
            s = [target._model.getVarByName(x.varName) for x in var_tuple]
            result[tuple(s)] = value
        else:
            s = target._model.getVarByName(var_tuple.varName)
            result[s] = value
    setattr(target, attribute, result)

def _copy_uncertainty_mix(target, attribute, value):
    """Copy mixed uncertainty (attribute, value) to target"""
    result = {}
    for keys, dist in value.items():
        s = []
        for key in keys:
            if type(key) == gurobipy.Var:
                s.append(target._model.getVarByName(key.varName))
            elif type(key) == gurobipy.Constr:
                s.append(target._model.getConstrByName(key.constrName))
            else:
                constr = target._model.getConstrByName(key[0].constrName)
                var = target._model.getVarByName(key[1].varName)
                s.append((constr, var))
        result[tuple(s)] = dist
    setattr(target, attribute, result)

def _copy_vars(target, attribute, value):
    """
     copy vars (attribute, value) to target

     """
    if type(value) == list:
        result = [target._model.getVarByName(x.varName) for x in value]
    else:
        result = (
            target._model.getVarByName(value.varName)
            if value is not None
            else None
        )
    setattr(target, attribute, result)

def _copy_constrs(target, attribute, value):
    """Copy constrs (attribute, value) to target"""
    if type(value) == list:
        result = [
            target._model.getConstrByName(x.constrName) for x in value
        ]
    else:
        result = target._model.getConstrByName(value.constrName)
    setattr(target, attribute, result)
