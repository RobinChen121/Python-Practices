#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:20:40 2024

@author: zhen chen

@disp:  test multi processing in python

    
    
"""

import time

from multiprocessing import Pool


def f(x):
    return x*x

if __name__ == '__main__':
    
    loop_nums = [10, 1000, 100000, 1000000, 4000000]
    for loop_num in loop_nums:
        input_data = range(loop_num)
        
        print('loop number is {:,}'.format(loop_num))
        start = time.process_time()
        with Pool() as p:        
            p.map(f, input_data)
        end = time.process_time()
        print('running time with parallel computing is %.4fs' % (end - start))
        
        start = time.process_time()
        for i in range(loop_num):
            f(input_data[i])
        end = time.process_time()
        print('running time without parallel computing is %.4fs\n' % (end - start))