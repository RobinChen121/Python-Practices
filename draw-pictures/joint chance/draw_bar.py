# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:49:35 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: draw bars for the joint chance results with confidence intervals
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np

sampleSimNum = 200
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']
simAvgServiceRate = [0.54, 0.58, 0.58, 0.64, 0.55, 0.66, 0.65, 0.44, 0.23, 0.69]
error1 = [1.96*math.sqrt(i*(1-i)/sampleSimNum) for i in simAvgServiceRate]
simExtendServiceRate = [0.68, 0.69, 0.6, 0.68, 0.64, 0.64, 0.66, 0.6, 0.59, 0.64]
error2 = [1.96*math.sqrt(i*(1-i)/sampleSimNum) for i in simExtendServiceRate]
index1 = np.arange(len(demandPatterns))
barWidth = 0.4
index2= index1 + barWidth  # 女生条形图的横坐标

plt.bar(index1, height=simAvgServiceRate, width = barWidth, edgecolor = 'black', yerr=error1, capsize=2, color='r', alpha=0.5, label='SAA service rate')
plt.bar(index2, height=simExtendServiceRate, width = barWidth, edgecolor = 'black', yerr=error2, capsize=2, color='g', alpha=0.5, label='ExtendSAA service rate')


plt.ylim([0, 1])
plt.xticks(index1 + barWidth/2, demandPatterns)
plt.axhline(y=0.6, xmin=0.01, xmax=0.99, ls='--', color='k', label = 'required service rate')
plt.legend()
plt.title('3025 scenarios')
plt.show()


plt.figure()
saaObj = [0.51, 0.67, 0.17, 0.6, 0.1, 0.8, 0.24, 0.23, 0.31, 0.35]
extendObj = [0.42, 0.6, 0.22, 0.53, 0.14, 0.58, 0.23, 0.21, 0.19, 0.25]
plt.bar(index1, height=saaObj, width = barWidth, edgecolor = 'black', yerr=error1, capsize=2, color='r', alpha=0.5, label='SAA objective')
plt.bar(index2, height=extendObj, width = barWidth, edgecolor = 'black', yerr=error2, capsize=2, color='g', alpha=0.5, label='ExtendSAA objective')
plt.xticks(index1 + barWidth/2, demandPatterns)
plt.legend()
plt.show()
