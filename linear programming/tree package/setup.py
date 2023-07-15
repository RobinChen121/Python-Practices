#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:21:27 2023

@author: zhenchen

@disp:  
    
    
"""

from setuptools import setup, find_packages

setup(
    name = 'scenario_tree',
    version = '0.2',
    author='Zhen Chen',
    author_email='15011074486@163.com',
    description = 'Get the sceanrio tree stucture',
    packages = find_packages(),
    py_modules=['tree'],
    install_requires=[]
)