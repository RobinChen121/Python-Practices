# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:51:26 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import matplotlib.pyplot as plt
import math


sampleSimNum = 200
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']
simAvgServiceRate = [0.54, 0.58, 0.58, 0.64, 0.55, 0.66, 0.65, 0.44, 0.23, 0.69]
error1 = [1.96*math.sqrt(i*(1-i)/sampleSimNum) for i in simAvgServiceRate]
simExtendServiceRate = [0.68, 0.69, 0.6, 0.68, 0.64, 0.64, 0.66, 0.6, 0.59, 0.64]
error2 = [1.96*math.sqrt(i*(1-i)/sampleSimNum) for i in simExtendServiceRate]

saaObj = [0.51, 0.67, 0.17, 0.6, 0.1, 0.8, 0.24, 0.23, 0.31, 0.35]
extendObj = [0.42, 0.6, 0.22, 0.53, 0.14, 0.58, 0.23, 0.21, 0.19, 0.25]

plt.figure()
plt.plot(demandPatterns, simAvgServiceRate, 'o-', markersize = 8, label='SAA service rate')
plt.plot(demandPatterns,simExtendServiceRate, 'o-', markersize = 8, label='ExtendSAA service rate')
plt.plot(demandPatterns, saaObj, '*-', markersize = 8, label='SAA objective')
plt.plot(demandPatterns, extendObj, '*-', markersize = 8, label='ExtendSAA objective')
plt.axhline(y=0.6, xmin=0.01, xmax=0.99, ls='--',  label = 'required service rate')
plt.ylim([0, 1])
plt.legend()
plt.title('3025 scenarios')
plt.show()

simAvgServiceRate = [0.71, 0.43, 0.55, 0.67, 0.57, 0.64, 0.61, 0.44, 0.23, 0.69]
saaObj = [0.31, 0.56, 0.21, 0.59, 0.75, 0.8, 0.21, 0.23, 0.31, 0.35]
simExtendServiceRate = [0.82, 0.75, 0.75, 0.63, 0.61, 0.64, 0.71, 0.6, 0.59, 0.64]
extendObj = [0.2, 0.58, 0.15, 0.45, 0.18, 0.73, 0.33, 0.21, 0.19, 0.25]