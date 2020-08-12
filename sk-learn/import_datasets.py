# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:15:44 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

from sklearn import datasets

iris = datasets.load_iris()
cancer = datasets.load_breast_cancer()
cancer_data = cancer['data']
caner_target = cancer['target']
caner_names = cancer['feature_names']





