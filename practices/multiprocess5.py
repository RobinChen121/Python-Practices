#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:04:37 2025

@author: zhenchen

@Python version: 3.10

@disp:  
    
    
"""

from multiprocessing import Queue

colors = ['red', 'green', 'blue', 'black']
cnt = 1
# instantiating a queue object
queue = Queue()
print('pushing items to queue:')
for color in colors:
    print('item no: ', cnt, ' ', color)
    queue.put(color)
    cnt += 1

print('\npopping items from queue:')
cnt = 0
while not queue.empty():
    print('item no: ', cnt, ' ', queue.get())
    cnt += 1